"""Tests for the set_request_context bootstrap graph node."""

from __future__ import annotations

import pytest

from kume.infrastructure.request_context import get_context, set_context
from kume.services.nodes.set_context import set_request_context


class TestSetRequestContext:
    """set_request_context node populates RequestContext from graph state."""

    @pytest.fixture(autouse=True)
    def _clear_context(self) -> None:  # type: ignore[misc]
        set_context(None)  # type: ignore[arg-type]
        yield  # type: ignore[misc]
        set_context(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_sets_context_from_state(self) -> None:
        """When user_id is present, the RequestContext is populated."""
        state = {
            "user_id": "user-42",
            "user_language": "es",
        }
        result = await set_request_context(state)

        ctx = get_context()
        assert ctx is not None
        assert ctx.user_id == "user-42"
        assert ctx.language == "es"
        assert ctx.telegram_id == 0
        assert result == {}

    @pytest.mark.asyncio
    async def test_defaults_language_to_english(self) -> None:
        """When user_language is missing, defaults to 'en'."""
        state = {"user_id": "user-7"}
        await set_request_context(state)

        ctx = get_context()
        assert ctx is not None
        assert ctx.language == "en"

    @pytest.mark.asyncio
    async def test_skips_when_no_user_id(self) -> None:
        """When user_id is empty or missing, no context is set."""
        state: dict[str, str] = {}
        result = await set_request_context(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict(self) -> None:
        """The node never modifies graph state."""
        state = {"user_id": "user-1", "user_language": "pt"}
        result = await set_request_context(state)

        assert result == {}
