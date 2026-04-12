import logging

from kume.infrastructure.logging import JSONFormatter, PrettyFormatter, setup_logging


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    metrics: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="kume.test",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if metrics is not None:
        record.metrics = metrics  # type: ignore[attr-defined]
    return record


SAMPLE_METRICS: dict = {
    "request_id": "abcd1234-5678",
    "telegram_id": 42,
    "llm_calls": [
        {
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.0025,
            "latency_ms": 320.5,
        },
    ],
    "tool_executions": [
        {"tool_name": "search_context", "latency_ms": 45.0, "success": True},
        {"tool_name": "failing_tool", "latency_ms": 12.0, "success": False},
    ],
    "embeddings": [
        {
            "model": "text-embedding-3-small",
            "chunk_count": 3,
            "cost_usd": 0.0001,
            "latency_ms": 80.0,
        },
    ],
    "total_cost_usd": 0.0026,
    "total_latency_ms": 457.5,
    "total_input_tokens": 100,
    "total_output_tokens": 50,
}


def test_pretty_formatter_formats_metrics() -> None:
    fmt = PrettyFormatter()
    record = _make_record(metrics=SAMPLE_METRICS)
    output = fmt.format(record)

    # Header with truncated request id and telegram_id
    assert "Request abcd1234" in output
    assert "telegram_id=42" in output

    # LLM call
    assert "LLM: gpt-4o" in output
    assert "100 in" in output
    assert "50 out" in output
    assert "$0.0025" in output
    assert "320ms" in output  # 320.5 rounds to 320 (banker's rounding)

    # Tool executions
    assert "Tool: search_context" in output
    assert "Tool: failing_tool" in output

    # Embedding
    assert "Embed: text-embedding-3-small" in output
    assert "3 chunks" in output

    # Totals
    assert "$0.0026" in output
    assert "458ms" in output  # 457.5 rounded
    assert "100 in" in output
    assert "50 out" in output

    # Verify multiline structure
    lines = output.split("\n")
    assert len(lines) >= 4  # header + llm + tools + embed + total + footer


def test_pretty_formatter_handles_non_metrics_message() -> None:
    fmt = PrettyFormatter()
    record = _make_record(msg="Starting up")
    output = fmt.format(record)

    assert "[INFO]" in output
    assert "kume.test" in output
    assert "Starting up" in output
    # Should not contain metrics markers
    assert "Request" not in output


def test_setup_logging_selects_json_formatter() -> None:
    setup_logging("INFO", "json")
    logger = logging.getLogger("kume")

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, JSONFormatter)


def test_setup_logging_selects_pretty_formatter() -> None:
    setup_logging("INFO", "pretty")
    logger = logging.getLogger("kume")

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, PrettyFormatter)


def test_setup_logging_defaults_to_pretty_formatter() -> None:
    setup_logging("INFO")
    logger = logging.getLogger("kume")

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, PrettyFormatter)


def test_pretty_formatter_tool_success_and_failure_icons() -> None:
    fmt = PrettyFormatter()
    record = _make_record(metrics=SAMPLE_METRICS)
    output = fmt.format(record)

    # Success tool gets checkmark, failure gets cross
    lines = output.split("\n")
    tool_lines = [line for line in lines if "Tool:" in line]
    assert len(tool_lines) == 2
    assert "\u2705" in tool_lines[0]  # search_context succeeded
    assert "\u274c" in tool_lines[1]  # failing_tool failed
