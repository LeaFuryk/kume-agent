import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    orchestrator_model: str
    tool_model: str
    max_agent_iterations: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_token = os.environ.get("TELEGRAM_TOKEN", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not telegram_token:
            raise ValueError("TELEGRAM_TOKEN environment variable is required")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return cls(
            telegram_token=telegram_token,
            openai_api_key=openai_api_key,
            orchestrator_model=os.environ.get("ORCHESTRATOR_MODEL", "gpt-4o"),
            tool_model=os.environ.get("TOOL_MODEL", "gpt-4o-mini"),
            max_agent_iterations=int(os.environ.get("MAX_AGENT_ITERATIONS", "5")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
