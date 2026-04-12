from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class LLMCallMetric:
    """Tracks a single LLM invocation."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    purpose: str  # "orchestrator", "tool:ask_recommendation", etc.


@dataclass(frozen=True)
class ToolExecutionMetric:
    """Tracks a single tool execution."""

    tool_name: str
    latency_ms: float
    success: bool


@dataclass(frozen=True)
class EmbeddingMetric:
    """Tracks a single embedding invocation."""

    model: str
    token_count: int
    chunk_count: int
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class IngestionMetric:
    """Tracks a single document ingestion."""

    document_type: str  # "pdf", "audio", "text"
    chunks_created: int
    lab_markers_extracted: int
    total_latency_ms: float


@dataclass
class RequestMetrics:
    """Aggregates all metrics for a single user request."""

    request_id: str
    telegram_id: int
    start_time: datetime
    end_time: datetime | None = None
    llm_calls: list[LLMCallMetric] = field(default_factory=list)
    tool_executions: list[ToolExecutionMetric] = field(default_factory=list)
    embeddings: list[EmbeddingMetric] = field(default_factory=list)
    ingestions: list[IngestionMetric] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.llm_calls) + sum(e.cost_usd for e in self.embeddings)

    @property
    def total_latency_ms(self) -> float:
        return (
            sum(call.latency_ms for call in self.llm_calls)
            + sum(execution.latency_ms for execution in self.tool_executions)
            + sum(e.latency_ms for e in self.embeddings)
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.llm_calls)
