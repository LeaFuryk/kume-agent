from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        if record.exc_info and record.exc_info[1] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_data["stack_info"] = record.stack_info
        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development that renders request metrics clearly."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "metrics") and record.metrics:
            return self._format_metrics(record.metrics)
        # For non-metrics messages, use a simple readable format
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        return f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"

    def _format_metrics(self, metrics: dict) -> str:  # type: ignore[type-arg]
        rid = metrics.get("request_id", "????")[:8]
        tid = metrics.get("telegram_id", "?")
        lines = [
            f"\u2500\u2500 Request {rid} \u2500\u2500 telegram_id={tid} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        ]

        for call in metrics.get("llm_calls", []):
            model = call.get("model", "?")
            inp = call.get("input_tokens", 0)
            out = call.get("output_tokens", 0)
            cost = call.get("cost_usd", 0)
            lat = call.get("latency_ms", 0)
            lines.append(f"\u2502 \U0001f916 LLM: {model} | {inp} in \u2192 {out} out | ${cost:.4f} | {lat:.0f}ms")

        for tool in metrics.get("tool_executions", []):
            name = tool.get("tool_name", "?")
            lat = tool.get("latency_ms", 0)
            status = "\u2705" if tool.get("success") else "\u274c"
            lines.append(f"\u2502   \u2514\u2500 {status} Tool: {name} | {lat:.0f}ms")

        for emb in metrics.get("embeddings", []):
            model = emb.get("model", "?")
            chunks = emb.get("chunk_count", 0)
            cost = emb.get("cost_usd", 0)
            lat = emb.get("latency_ms", 0)
            lines.append(f"\u2502 \U0001f4e6 Embed: {model} | {chunks} chunks | ${cost:.4f} | {lat:.0f}ms")

        total_cost = metrics.get("total_cost_usd", 0)
        total_lat = metrics.get("total_latency_ms", 0)
        total_in = metrics.get("total_input_tokens", 0)
        total_out = metrics.get("total_output_tokens", 0)
        lines.append(f"\u2502 \u2705 Total: ${total_cost:.4f} | {total_lat:.0f}ms | {total_in} in | {total_out} out")
        lines.append(
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        )

        return "\n".join(lines)


def setup_logging(level: str = "INFO", log_format: str = "pretty") -> None:
    """Configure the kume logger with the chosen format to stdout.

    Safe to call multiple times — clears existing handlers before adding a new one.
    """
    root = logging.getLogger("kume")
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PrettyFormatter())
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False
