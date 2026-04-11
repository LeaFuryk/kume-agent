from datetime import UTC, datetime

import pytest

from kume.domain.metrics import LLMCallMetric, RequestMetrics, ToolExecutionMetric


def _make_llm_call(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.01,
    latency_ms: float = 200.0,
) -> LLMCallMetric:
    return LLMCallMetric(
        model="gpt-4o",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        purpose="orchestrator",
    )


def _make_tool_exec(*, latency_ms: float = 50.0, success: bool = True) -> ToolExecutionMetric:
    return ToolExecutionMetric(tool_name="ask_recommendation", latency_ms=latency_ms, success=success)


def _make_request_metrics(
    *,
    llm_calls: list[LLMCallMetric] | None = None,
    tool_executions: list[ToolExecutionMetric] | None = None,
) -> RequestMetrics:
    return RequestMetrics(
        request_id="req-1",
        telegram_id=123,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        llm_calls=llm_calls or [],
        tool_executions=tool_executions or [],
    )


class TestLLMCallMetric:
    def test_creation(self) -> None:
        metric = _make_llm_call()
        assert metric.model == "gpt-4o"
        assert metric.purpose == "orchestrator"

    def test_immutability(self) -> None:
        metric = _make_llm_call()
        with pytest.raises(AttributeError):
            metric.model = "other"  # type: ignore[misc]


class TestToolExecutionMetric:
    def test_creation(self) -> None:
        metric = _make_tool_exec()
        assert metric.tool_name == "ask_recommendation"
        assert metric.success is True

    def test_immutability(self) -> None:
        metric = _make_tool_exec()
        with pytest.raises(AttributeError):
            metric.success = False  # type: ignore[misc]


class TestRequestMetrics:
    def test_empty_lists(self) -> None:
        metrics = _make_request_metrics()
        assert metrics.total_cost_usd == 0.0
        assert metrics.total_latency_ms == 0.0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0

    def test_single_llm_call(self) -> None:
        call = _make_llm_call(input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=200.0)
        metrics = _make_request_metrics(llm_calls=[call])
        assert metrics.total_cost_usd == pytest.approx(0.01)
        assert metrics.total_latency_ms == pytest.approx(200.0)
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 50

    def test_multiple_llm_calls(self) -> None:
        calls = [
            _make_llm_call(input_tokens=100, output_tokens=50, cost_usd=0.01, latency_ms=200.0),
            _make_llm_call(input_tokens=200, output_tokens=80, cost_usd=0.02, latency_ms=300.0),
        ]
        metrics = _make_request_metrics(llm_calls=calls)
        assert metrics.total_cost_usd == pytest.approx(0.03)
        assert metrics.total_latency_ms == pytest.approx(500.0)
        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 130

    def test_latency_includes_tool_executions(self) -> None:
        call = _make_llm_call(latency_ms=200.0)
        tool = _make_tool_exec(latency_ms=50.0)
        metrics = _make_request_metrics(llm_calls=[call], tool_executions=[tool])
        assert metrics.total_latency_ms == pytest.approx(250.0)

    def test_end_time_defaults_to_none(self) -> None:
        metrics = _make_request_metrics()
        assert metrics.end_time is None

    def test_end_time_can_be_set(self) -> None:
        metrics = _make_request_metrics()
        end = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        metrics.end_time = end
        assert metrics.end_time == end

    def test_mutable_lists(self) -> None:
        metrics = _make_request_metrics()
        metrics.llm_calls.append(_make_llm_call(cost_usd=0.05))
        assert metrics.total_cost_usd == pytest.approx(0.05)
