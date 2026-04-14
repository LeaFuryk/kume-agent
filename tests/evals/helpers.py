from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import AsyncCallbackHandler


@dataclass
class EvalCase:
    id: str
    description: str
    input: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    has_image: bool = False
    user_prefix: str = ""


@dataclass
class QualityCase:
    id: str
    description: str
    input: str
    user_prefix: str = ""
    criteria: list[str] = field(default_factory=list)
    expected_language: str = "en"


@dataclass
class EvalResult:
    """Result from running a single eval case through the orchestrator."""

    tool_calls: list[str]
    response_text: str


def load_cases(yaml_path: str | Path) -> list[EvalCase]:
    """Load eval cases from a YAML file."""
    path = Path(yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    cases = []
    for item in data.get("cases", []):
        cases.append(
            EvalCase(
                id=item["id"],
                description=item.get("description", ""),
                input=item["input"],
                expected_tools=item.get("expected_tools", []),
                forbidden_tools=item.get("forbidden_tools", []),
                has_image=item.get("has_image", False),
                user_prefix=item.get("user_prefix", ""),
            )
        )
    return cases


def load_quality_cases(yaml_path: str | Path) -> list[QualityCase]:
    """Load response quality eval cases."""
    path = Path(yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return [
        QualityCase(
            id=item["id"],
            description=item.get("description", ""),
            input=item["input"],
            user_prefix=item.get("user_prefix", ""),
            criteria=item.get("criteria", []),
            expected_language=item.get("expected_language", "en"),
        )
        for item in data.get("cases", [])
    ]


# ---------------------------------------------------------------------------
# Eval runner — sends messages through a real orchestrator and captures results
# ---------------------------------------------------------------------------


class _ToolCapture(AsyncCallbackHandler):
    """Callback handler that records which tools the agent called."""

    def __init__(self) -> None:
        self.tool_calls: list[str] = []

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.tool_calls.append(serialized.get("name", "unknown"))


async def run_eval(
    orchestrator: Any,
    user_message: str,
    user_prefix: str = "",
    resources: list[Any] | None = None,
) -> EvalResult:
    """Run a single eval case through the orchestrator and capture tool calls.

    Patches the orchestrator's agent to inject a tool-capture callback,
    then calls process() and returns the tools called + response text.
    """
    capture = _ToolCapture()

    # Store the original ainvoke so we can wrap it
    original_ainvoke = orchestrator._agent.ainvoke

    async def _capturing_ainvoke(input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        # Inject our capture callback into the config
        if config and "callbacks" in config:
            config["callbacks"].append(capture)
        elif config:
            config["callbacks"] = [capture]
        return await original_ainvoke(input_data, config=config, **kwargs)

    orchestrator._agent.ainvoke = _capturing_ainvoke

    try:
        # Build the full message with user prefix if provided
        full_message = user_message
        if user_prefix:
            full_message = f"{user_prefix}\n{user_message}"

        result = await orchestrator.process(
            telegram_id=99999,
            user_message=full_message,
            resources=resources,
        )
        response_text = result.text if hasattr(result, "text") else str(result)
    finally:
        # Restore original ainvoke
        orchestrator._agent.ainvoke = original_ainvoke

    return EvalResult(tool_calls=capture.tool_calls, response_text=response_text)


async def judge_response(
    llm: Any,
    user_message: str,
    response_text: str,
    criteria: list[str],
    expected_language: str = "en",
) -> dict[str, int]:
    """Use an LLM to score a response on given criteria (1-5 each).

    Returns a dict of {criterion: score}.
    Cost: ~$0.002 per call at gpt-4o-mini pricing.
    """
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt = f"""\
Score this chatbot response on each criterion (1-5, where 5 is best).

User message: {user_message}
Expected response language: {expected_language}
Bot response: {response_text}

Criteria to score:
{criteria_text}

Return ONLY a JSON object with each criterion as key and integer score as value.
Example: {{"language_match": 5, "conciseness": 4}}
"""
    from langchain_core.messages import HumanMessage

    result = await llm.ainvoke([HumanMessage(content=prompt)])
    content = result.content if hasattr(result, "content") else str(result)

    try:
        scores = json.loads(content)
        return {k: int(v) for k, v in scores.items()}
    except (json.JSONDecodeError, ValueError):
        # Try to extract JSON from markdown code blocks
        import re

        match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
        if match:
            scores = json.loads(match.group(1).strip())
            return {k: int(v) for k, v in scores.items()}
        return {c: 0 for c in criteria}
