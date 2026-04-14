from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EvalCase:
    id: str
    description: str
    input: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    has_image: bool = False
    user_prefix: str = ""


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
