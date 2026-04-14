from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


def load_cases(yaml_path: str | Path) -> list[EvalCase]:
    """Load eval cases from a YAML file."""
    path = Path(yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return [
        EvalCase(
            id=item["id"],
            description=item.get("description", ""),
            input=item["input"],
            expected_tools=item.get("expected_tools", []),
            forbidden_tools=item.get("forbidden_tools", []),
            has_image=item.get("has_image", False),
            user_prefix=item.get("user_prefix", ""),
        )
        for item in data.get("cases", [])
    ]


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
# Cost tracking
# ---------------------------------------------------------------------------

# Fallback pricing per 1M tokens (USD) when LangChain doesn't know the model
_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single LLM call."""
    try:
        from langchain_community.callbacks.openai_info import TokenType, get_openai_token_cost_for_model

        input_cost = get_openai_token_cost_for_model(model, input_tokens, token_type=TokenType.PROMPT)
        output_cost = get_openai_token_cost_for_model(model, output_tokens, token_type=TokenType.COMPLETION)
        return input_cost + output_cost
    except (ValueError, ImportError):
        # Fallback to manual pricing
        for prefix, (inp_per_m, out_per_m) in _FALLBACK_PRICING.items():
            if prefix in model:
                return (input_tokens * inp_per_m + output_tokens * out_per_m) / 1_000_000
        return 0.0


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


class _ToolAndCostCapture(AsyncCallbackHandler):
    """Callback that records tool calls + token usage/cost per LLM call."""

    def __init__(self) -> None:
        self.tool_calls: list[str] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.model: str = ""

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.tool_calls.append(serialized.get("name", "unknown"))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        model = response.llm_output.get("model_name", "") if response.llm_output else ""
        if model:
            self.model = model
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cost_usd += _estimate_cost(model or self.model, inp, out)


async def run_eval(
    orchestrator: Any,
    user_message: str,
    user_prefix: str = "",
    resources: list[Any] | None = None,
) -> EvalResult:
    """Run a single eval case through the orchestrator and capture tool calls + cost."""
    capture = _ToolAndCostCapture()

    original_ainvoke = orchestrator._agent.ainvoke

    async def _capturing_ainvoke(input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        if config and "callbacks" in config:
            config["callbacks"].append(capture)
        elif config:
            config["callbacks"] = [capture]
        return await original_ainvoke(input_data, config=config, **kwargs)

    orchestrator._agent.ainvoke = _capturing_ainvoke

    try:
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
        orchestrator._agent.ainvoke = original_ainvoke

    return EvalResult(
        tool_calls=capture.tool_calls,
        response_text=response_text,
        input_tokens=capture.total_input_tokens,
        output_tokens=capture.total_output_tokens,
        cost_usd=capture.total_cost_usd,
        model=capture.model,
    )


async def judge_response(
    llm: Any,
    user_message: str,
    response_text: str,
    criteria: list[str],
    expected_language: str = "en",
) -> dict[str, int]:
    """Use an LLM to score a response on given criteria (1-5 each).

    Returns a dict of {criterion: score}.
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
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
        if match:
            scores = json.loads(match.group(1).strip())
            return {k: int(v) for k, v in scores.items()}
        return {c: 0 for c in criteria}
