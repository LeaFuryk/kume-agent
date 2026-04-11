from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from kume.domain.metrics import LLMCallMetric, RequestMetrics, ToolExecutionMetric

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        logging.getLogger("kume.metrics").warning("Unknown model %r — cost will be reported as $0.00", model)
        return 0.0
    return (input_tokens * pricing[0] / 1000) + (output_tokens * pricing[1] / 1000)


def _metrics_to_dict(metrics: RequestMetrics) -> dict[str, Any]:
    data = asdict(metrics)
    data["start_time"] = metrics.start_time.isoformat()
    data["end_time"] = metrics.end_time.isoformat() if metrics.end_time else None
    data["total_cost_usd"] = metrics.total_cost_usd
    data["total_latency_ms"] = metrics.total_latency_ms
    data["total_input_tokens"] = metrics.total_input_tokens
    data["total_output_tokens"] = metrics.total_output_tokens
    return data


class MetricsCollector:
    """Request-scoped metric aggregation."""

    def start_request(self, telegram_id: int) -> None:
        self._request_id = str(uuid.uuid4())
        self._telegram_id = telegram_id
        self._start_time = datetime.now(UTC)
        self._llm_calls: list[LLMCallMetric] = []
        self._tool_executions: list[ToolExecutionMetric] = []

    def record_llm_call(self, metric: LLMCallMetric) -> None:
        self._llm_calls.append(metric)

    def record_tool_execution(self, metric: ToolExecutionMetric) -> None:
        self._tool_executions.append(metric)

    def end_request(self) -> RequestMetrics:
        metrics = RequestMetrics(
            request_id=self._request_id,
            telegram_id=self._telegram_id,
            start_time=self._start_time,
            end_time=datetime.now(UTC),
            llm_calls=list(self._llm_calls),
            tool_executions=list(self._tool_executions),
        )
        logger = logging.getLogger("kume.metrics")
        logger.info("request_metrics", extra={"metrics": _metrics_to_dict(metrics)})
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
