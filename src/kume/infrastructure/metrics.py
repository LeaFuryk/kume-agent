from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from langchain_community.callbacks.openai_info import TokenType, get_openai_token_cost_for_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from kume.domain.metrics import EmbeddingMetric, IngestionMetric, LLMCallMetric, RequestMetrics, ToolExecutionMetric

logger = logging.getLogger("kume.metrics")


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost using LangChain's built-in OpenAI pricing table."""
    try:
        input_cost = get_openai_token_cost_for_model(model, input_tokens, token_type=TokenType.PROMPT)
        output_cost = get_openai_token_cost_for_model(model, output_tokens, token_type=TokenType.COMPLETION)
        return input_cost + output_cost
    except ValueError:
        logger.warning("Unknown model %r — cost will be reported as $0.00", model)
        return 0.0


def _metrics_to_dict(metrics: RequestMetrics, user_name: str | None = None) -> dict[str, Any]:
    data = asdict(metrics)
    data["start_time"] = metrics.start_time.isoformat()
    data["end_time"] = metrics.end_time.isoformat() if metrics.end_time else None
    if user_name:
        data["user_name"] = user_name
    data["total_cost_usd"] = metrics.total_cost_usd
    data["total_latency_ms"] = metrics.total_latency_ms
    data["total_input_tokens"] = metrics.total_input_tokens
    data["total_output_tokens"] = metrics.total_output_tokens
    data["total_embedding_tokens"] = sum(e.token_count for e in metrics.embeddings)
    data["total_chunks_ingested"] = sum(i.chunks_created for i in metrics.ingestions)
    return data


class MetricsCollector:
    """Request-scoped metric aggregation."""

    def start_request(self, telegram_id: int, user_name: str | None = None) -> None:
        self._request_id = str(uuid.uuid4())
        self._telegram_id = telegram_id
        self._user_name = user_name
        self._start_time = datetime.now(UTC)
        self._llm_calls: list[LLMCallMetric] = []
        self._tool_executions: list[ToolExecutionMetric] = []
        self._embeddings: list[EmbeddingMetric] = []
        self._ingestions: list[IngestionMetric] = []

    def record_llm_call(self, metric: LLMCallMetric) -> None:
        self._llm_calls.append(metric)

    def record_tool_execution(self, metric: ToolExecutionMetric) -> None:
        self._tool_executions.append(metric)

    def record_embedding(self, metric: EmbeddingMetric) -> None:
        self._embeddings.append(metric)

    def record_ingestion(self, metric: IngestionMetric) -> None:
        self._ingestions.append(metric)

    def end_request(self) -> RequestMetrics:
        metrics = RequestMetrics(
            request_id=self._request_id,
            telegram_id=self._telegram_id,
            start_time=self._start_time,
            end_time=datetime.now(UTC),
            llm_calls=list(self._llm_calls),
            tool_executions=list(self._tool_executions),
            embeddings=list(self._embeddings),
            ingestions=list(self._ingestions),
        )
        logger.info(
            "request_metrics",
            extra={"metrics": _metrics_to_dict(metrics, getattr(self, "_user_name", None))},
        )
        return metrics


class MetricsCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that captures per-call metrics."""

    def __init__(self, collector: MetricsCollector) -> None:
        self.collector = collector
        self._llm_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}
        self._tool_names: dict[str, str] = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:
        self._llm_start_times[str(run_id)] = time.monotonic()

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        start = self._llm_start_times.pop(str(run_id), time.monotonic())
        latency = (time.monotonic() - start) * 1000
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        model = response.llm_output.get("model_name", "unknown") if response.llm_output else "unknown"
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = _compute_cost(model, input_tokens, output_tokens)
        self.collector.record_llm_call(
            LLMCallMetric(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency,
                purpose="orchestrator",
            )
        )

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any, **kwargs: Any) -> None:
        self._tool_start_times[str(run_id)] = time.monotonic()
        self._tool_names[str(run_id)] = serialized.get("name", "unknown")

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        start = self._tool_start_times.pop(str(run_id), time.monotonic())
        latency = (time.monotonic() - start) * 1000
        tool_name = self._tool_names.pop(str(run_id), "unknown")
        self.collector.record_tool_execution(ToolExecutionMetric(tool_name=tool_name, latency_ms=latency, success=True))

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        start = self._tool_start_times.pop(str(run_id), time.monotonic())
        latency = (time.monotonic() - start) * 1000
        tool_name = self._tool_names.pop(str(run_id), "unknown")
        self.collector.record_tool_execution(
            ToolExecutionMetric(tool_name=tool_name, latency_ms=latency, success=False)
        )


# ---------------------------------------------------------------------------
# Reasoning logger — logs the agent's reasoning chain (content, not metrics)
# ---------------------------------------------------------------------------

_reasoning_logger = logging.getLogger("kume.reasoning")


class ReasoningCallbackHandler(BaseCallbackHandler):
    """Logs the agent's reasoning chain: tool calls, tool results, and LLM decisions.

    Designed for development — shows what the model is thinking and doing.
    """

    def __init__(self, user_name: str | None = None) -> None:
        self._user_name = user_name
        self._tool_names: dict[str, str] = {}
        self._tool_inputs: dict[str, str] = {}

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")
        self._tool_names[str(run_id)] = tool_name
        self._tool_inputs[str(run_id)] = input_str
        _reasoning_logger.info("\u2502 \U0001f527 Tool call: %s(%s)", tool_name, input_str)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        tool_name = self._tool_names.pop(str(run_id), "unknown")
        self._tool_inputs.pop(str(run_id), None)
        output_str = str(output) if output else "(empty)"
        _reasoning_logger.info(
            "\u2502   \u2514\u2500 \u2705 %s result:\n%s",
            tool_name,
            _indent(output_str),
        )

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        tool_name = self._tool_names.pop(str(run_id), "unknown")
        self._tool_inputs.pop(str(run_id), None)
        _reasoning_logger.info(
            "\u2502   \u2514\u2500 \u274c %s FAILED: %s",
            tool_name,
            str(error)[:300],
        )

    def log_user_message(self, user_message: str, user_name: str | None = None) -> None:
        name_display = user_name or self._user_name or "anonymous"
        _reasoning_logger.info(
            "\u2502 \U0001f464 %s: %s",
            name_display,
            user_message,
        )

    def log_response(self, response: str) -> None:
        _reasoning_logger.info(
            "\u2502 \U0001f916 Kume: %s",
            response,
        )


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))
