from datetime import UTC, datetime

import pytest

from kume.domain.metrics import (
    EmbeddingMetric,
    IngestionMetric,
    LLMCallMetric,
    RequestMetrics,
    ToolExecutionMetric,
)


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


def _make_embedding(
    *,
    cost_usd: float = 0.001,
    latency_ms: float = 50.0,
) -> EmbeddingMetric:
    return EmbeddingMetric(
        model="text-embedding-3-small",
        token_count=500,
        chunk_count=5,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def _make_ingestion(
    *,
    document_type: str = "pdf",
    chunks_created: int = 10,
    lab_markers_extracted: int = 3,
    total_latency_ms: float = 1000.0,
) -> IngestionMetric:
    return IngestionMetric(
        document_type=document_type,
        chunks_created=chunks_created,
        lab_markers_extracted=lab_markers_extracted,
        total_latency_ms=total_latency_ms,
    )


def _make_request_metrics(
    *,
    llm_calls: list[LLMCallMetric] | None = None,
    tool_executions: list[ToolExecutionMetric] | None = None,
    embeddings: list[EmbeddingMetric] | None = None,
    ingestions: list[IngestionMetric] | None = None,
) -> RequestMetrics:
    return RequestMetrics(
        request_id="req-1",
        telegram_id=123,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        llm_calls=llm_calls or [],
        tool_executions=tool_executions or [],
        embeddings=embeddings or [],
        ingestions=ingestions or [],
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

    def test_total_cost_includes_embeddings(self) -> None:
        call = _make_llm_call(cost_usd=0.01)
        emb = _make_embedding(cost_usd=0.002)
        metrics = _make_request_metrics(llm_calls=[call], embeddings=[emb])
        assert metrics.total_cost_usd == pytest.approx(0.012)

    def test_total_latency_includes_embeddings(self) -> None:
        call = _make_llm_call(latency_ms=200.0)
        tool = _make_tool_exec(latency_ms=50.0)
        emb = _make_embedding(latency_ms=30.0)
        metrics = _make_request_metrics(llm_calls=[call], tool_executions=[tool], embeddings=[emb])
        assert metrics.total_latency_ms == pytest.approx(280.0)

    def test_default_embeddings_and_ingestions_empty(self) -> None:
        metrics = _make_request_metrics()
        assert metrics.embeddings == []
        assert metrics.ingestions == []


class TestEmbeddingMetric:
    def test_creation(self) -> None:
        metric = _make_embedding()
        assert metric.model == "text-embedding-3-small"
        assert metric.token_count == 500
        assert metric.chunk_count == 5

    def test_immutability(self) -> None:
        metric = _make_embedding()
        with pytest.raises(AttributeError):
            metric.model = "other"  # type: ignore[misc]


class TestIngestionMetric:
    def test_creation(self) -> None:
        metric = _make_ingestion()
        assert metric.document_type == "pdf"
        assert metric.chunks_created == 10
        assert metric.lab_markers_extracted == 3
        assert metric.total_latency_ms == 1000.0

    def test_immutability(self) -> None:
        metric = _make_ingestion()
        with pytest.raises(AttributeError):
            metric.document_type = "audio"  # type: ignore[misc]
