import pytest

from kume.infrastructure.config import Settings


def test_from_env_loads_all_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "gpt-4-turbo")
    monkeypatch.setenv("TOOL_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("MAX_AGENT_ITERATIONS", "10")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@host:5433/mydb")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    monkeypatch.setenv("LOG_FORMAT", "json")

    s = Settings.from_env()

    assert s.telegram_token == "tok-123"
    assert s.openai_api_key == "sk-abc"
    assert s.orchestrator_model == "gpt-4-turbo"
    assert s.tool_model == "gpt-3.5-turbo"
    assert s.max_agent_iterations == 10
    assert s.log_level == "DEBUG"
    assert s.database_url == "postgresql+asyncpg://u:p@host:5433/mydb"
    assert s.openai_embedding_model == "text-embedding-ada-002"
    assert s.log_format == "json"


def test_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    # Remove optional vars to ensure defaults kick in
    monkeypatch.delenv("ORCHESTRATOR_MODEL", raising=False)
    monkeypatch.delenv("TOOL_MODEL", raising=False)
    monkeypatch.delenv("MAX_AGENT_ITERATIONS", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    s = Settings.from_env()

    assert s.orchestrator_model == "gpt-4o"
    assert s.tool_model == "gpt-4o-mini"
    assert s.max_agent_iterations == 5
    assert s.log_level == "INFO"
    assert s.database_url == "postgresql+asyncpg://kume:kume@localhost:5432/kume"
    assert s.openai_embedding_model == "text-embedding-3-small"
    assert s.log_format == "pretty"


def test_from_env_raises_when_telegram_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")

    with pytest.raises(ValueError, match="TELEGRAM_TOKEN"):
        Settings.from_env()


def test_from_env_raises_when_openai_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok-123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings.from_env()


def test_settings_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")

    s = Settings.from_env()

    with pytest.raises(AttributeError):
        s.telegram_token = "new-value"  # type: ignore[misc]


def test_settings_importable_from_infrastructure_package() -> None:
    from kume.infrastructure import Settings as FromInfra

    assert FromInfra is Settings
