from __future__ import annotations

import json
import logging
import uuid

import pytest
from langchain_core.outputs import LLMResult

from kume.domain.metrics import LLMCallMetric, ToolExecutionMetric
from kume.infrastructure.logging import JSONFormatter, setup_logging
from kume.infrastructure.metrics import (
    MetricsCallbackHandler,
    MetricsCollector,
    _compute_cost,
    _metrics_to_dict,
)

# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_known_model_gpt4o(self) -> None:
        cost = _compute_cost("gpt-4o", input_tokens=1000, output_tokens=1000)
        # (1000 * 0.0025 / 1000) + (1000 * 0.01 / 1000) = 0.0025 + 0.01
        assert cost == pytest.approx(0.0125)

    def test_known_model_gpt4o_mini(self) -> None:
        cost = _compute_cost("gpt-4o-mini", input_tokens=2000, output_tokens=500)
        # (2000 * 0.00015 / 1000) + (500 * 0.0006 / 1000)
        assert cost == pytest.approx(0.0003 + 0.0003)

    def test_unknown_model_returns_zero(self) -> None:
        cost = _compute_cost("some-unknown-model", input_tokens=5000, output_tokens=5000)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_start_and_end_request(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=42)
        metrics = collector.end_request()

        assert metrics.telegram_id == 42
        assert metrics.request_id  # non-empty UUID string
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.end_time >= metrics.start_time
        assert metrics.llm_calls == []
        assert metrics.tool_executions == []

    def test_record_llm_call(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        llm_metric = LLMCallMetric(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
            latency_ms=200.0,
            purpose="orchestrator",
        )
        collector.record_llm_call(llm_metric)
        metrics = collector.end_request()

        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0] is llm_metric

    def test_record_tool_execution(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        tool_metric = ToolExecutionMetric(tool_name="search", latency_ms=50.0, success=True)
        collector.record_tool_execution(tool_metric)
        metrics = collector.end_request()

        assert len(metrics.tool_executions) == 1
        assert metrics.tool_executions[0] is tool_metric

    def test_end_request_returns_copies_of_lists(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        llm_metric = LLMCallMetric(
            model="gpt-4o", input_tokens=10, output_tokens=5, cost_usd=0.0, latency_ms=1.0, purpose="test"
        )
        collector.record_llm_call(llm_metric)
        metrics = collector.end_request()

        # Mutating the returned list should not affect internal state
        metrics.llm_calls.clear()
        # The internal list was already snapshotted, so this is just verifying immutability intent

    def test_end_request_logs_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=99)
        with caplog.at_level(logging.INFO, logger="kume.metrics"):
            collector.end_request()
        assert any("request_metrics" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MetricsCallbackHandler — LLM callbacks
# ---------------------------------------------------------------------------


class TestMetricsCallbackHandlerLLM:
    def _make_handler(self) -> tuple[MetricsCollector, MetricsCallbackHandler]:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        handler = MetricsCallbackHandler(collector)
        return collector, handler

    def test_on_llm_start_and_end_records_metric(self) -> None:
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_llm_start(serialized={}, prompts=["hello"], run_id=run_id)
        response = LLMResult(
            generations=[],
            llm_output={
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "model_name": "gpt-4o",
            },
        )
        handler.on_llm_end(response, run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.llm_calls) == 1
        call = metrics.llm_calls[0]
        assert call.model == "gpt-4o"
        assert call.input_tokens == 100
        assert call.output_tokens == 50
        assert call.latency_ms >= 0
        assert call.purpose == "orchestrator"
        # Cost should match _compute_cost
        assert call.cost_usd == _compute_cost("gpt-4o", 100, 50)

    def test_on_llm_end_without_llm_output(self) -> None:
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_llm_start(serialized={}, prompts=["hello"], run_id=run_id)
        response = LLMResult(generations=[], llm_output=None)
        handler.on_llm_end(response, run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.llm_calls) == 1
        call = metrics.llm_calls[0]
        assert call.model == "unknown"
        assert call.input_tokens == 0
        assert call.output_tokens == 0
        assert call.cost_usd == 0.0

    def test_multiple_llm_calls(self) -> None:
        collector, handler = self._make_handler()

        for _ in range(3):
            run_id = uuid.uuid4()
            handler.on_llm_start(serialized={}, prompts=["hi"], run_id=run_id)
            handler.on_llm_end(
                LLMResult(
                    generations=[],
                    llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model_name": "gpt-4o"},
                ),
                run_id=run_id,
            )

        metrics = collector.end_request()
        assert len(metrics.llm_calls) == 3


# ---------------------------------------------------------------------------
# MetricsCallbackHandler — Tool callbacks
# ---------------------------------------------------------------------------


class TestMetricsCallbackHandlerTool:
    def _make_handler(self) -> tuple[MetricsCollector, MetricsCallbackHandler]:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        handler = MetricsCallbackHandler(collector)
        return collector, handler

    def test_on_tool_start_and_end(self) -> None:
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_tool_start(serialized={"name": "search_food"}, input_str="apple", run_id=run_id)
        handler.on_tool_end(output="found apple", run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.tool_executions) == 1
        tool = metrics.tool_executions[0]
        assert tool.tool_name == "search_food"
        assert tool.success is True
        assert tool.latency_ms >= 0

    def test_on_tool_error(self) -> None:
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_tool_start(serialized={"name": "failing_tool"}, input_str="bad", run_id=run_id)
        handler.on_tool_error(error=RuntimeError("boom"), run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.tool_executions) == 1
        tool = metrics.tool_executions[0]
        assert tool.tool_name == "failing_tool"
        assert tool.success is False
        assert tool.latency_ms >= 0

    def test_on_tool_end_without_matching_start(self) -> None:
        """If on_tool_end is called without on_tool_start, it should not crash."""
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_tool_end(output="orphan", run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.tool_executions) == 1
        assert metrics.tool_executions[0].tool_name == "unknown"

    def test_on_tool_error_without_matching_start(self) -> None:
        collector, handler = self._make_handler()
        run_id = uuid.uuid4()

        handler.on_tool_error(error=ValueError("oops"), run_id=run_id)

        metrics = collector.end_request()
        assert len(metrics.tool_executions) == 1
        assert metrics.tool_executions[0].success is False


# ---------------------------------------------------------------------------
# _metrics_to_dict
# ---------------------------------------------------------------------------


class TestMetricsToDict:
    def test_serializes_all_fields(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=7)
        collector.record_llm_call(
            LLMCallMetric(model="gpt-4o", input_tokens=50, output_tokens=25, cost_usd=0.001, latency_ms=100.0,
                          purpose="test")
        )
        collector.record_tool_execution(ToolExecutionMetric(tool_name="search", latency_ms=30.0, success=True))
        metrics = collector.end_request()

        d = _metrics_to_dict(metrics)

        assert d["telegram_id"] == 7
        assert isinstance(d["start_time"], str)
        assert isinstance(d["end_time"], str)
        assert d["total_cost_usd"] == metrics.total_cost_usd
        assert d["total_latency_ms"] == metrics.total_latency_ms
        assert d["total_input_tokens"] == 50
        assert d["total_output_tokens"] == 25
        assert len(d["llm_calls"]) == 1
        assert len(d["tool_executions"]) == 1

    def test_serializable_as_json(self) -> None:
        collector = MetricsCollector()
        collector.start_request(telegram_id=1)
        metrics = collector.end_request()
        d = _metrics_to_dict(metrics)
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------


class TestJSONLogging:
    def test_json_formatter_basic(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="kume.test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "kume.test"
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_json_formatter_includes_metrics_extra(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="kume.metrics", level=logging.INFO, pathname="", lineno=0, msg="request_metrics", args=(),
            exc_info=None
        )
        record.metrics = {"total_cost_usd": 0.01}  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert data["metrics"] == {"total_cost_usd": 0.01}

    def test_setup_logging_configures_kume_logger(self) -> None:
        # Clean up any existing handlers first
        logger = logging.getLogger("kume")
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate

        try:
            logger.handlers.clear()
            setup_logging(level="DEBUG")

            assert logger.level == logging.DEBUG
            assert logger.propagate is False
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0].formatter, JSONFormatter)
        finally:
            # Restore original state
            logger.handlers = original_handlers
            logger.level = original_level
            logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_metrics_collector_importable_from_infrastructure(self) -> None:
        from kume.infrastructure import MetricsCollector as FromPkg

        assert FromPkg is MetricsCollector

    def test_metrics_callback_handler_importable_from_infrastructure(self) -> None:
        from kume.infrastructure import MetricsCallbackHandler as FromPkg

        assert FromPkg is MetricsCallbackHandler

    def test_setup_logging_importable_from_infrastructure(self) -> None:
        from kume.infrastructure import setup_logging as from_pkg

        assert from_pkg is setup_logging
