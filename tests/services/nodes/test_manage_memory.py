"""Tests for the manage_memory graph node."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

from kume.services.nodes.manage_memory import manage_memory


class TestManageMemory:
    """manage_memory node compresses conversation history when it grows too long."""

    async def test_passthrough_for_short_history(self) -> None:
        """When messages count <= threshold, no summarization occurs."""
        messages = [HumanMessage(content=f"Message {i}", id=str(i)) for i in range(10)]
        state = {"messages": messages}

        result = await manage_memory(state)

        assert result["memory_summarized"] is False
        # Should NOT include "messages" key — no mutation needed
        assert "messages" not in result

    async def test_summarizes_long_history(self) -> None:
        """When messages exceed threshold, older messages are summarized."""
        messages: list[BaseMessage] = []
        for i in range(25):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}", id=str(i)))
            else:
                messages.append(AIMessage(content=f"AI response {i}", id=str(i)))
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            new_callable=AsyncMock,
            return_value="Summary of older conversation about nutrition topics.",
        ):
            result = await manage_memory(state)

        assert result["memory_summarized"] is True
        assert "messages" in result

        # 15 RemoveMessages (for older msgs) + 1 summary + 10 recent = 26
        older_count = 25 - 10  # 15 messages to summarize
        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(removals) == older_count

        # Summary message follows the removals
        summary_msg = result["messages"][older_count]
        assert isinstance(summary_msg, HumanMessage)
        assert "[Previous conversation summary" in summary_msg.content
        assert "not instructions" in summary_msg.content

        # Last 10 original messages preserved after the summary
        kept = result["messages"][older_count + 1 :]
        assert len(kept) == 10

    async def test_preserves_last_10(self) -> None:
        """The last 10 messages are kept verbatim after summarization."""
        messages = []
        for i in range(25):
            messages.append(HumanMessage(content=f"Msg-{i}", id=str(i)))
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            new_callable=AsyncMock,
            return_value="Conversation summary.",
        ):
            result = await manage_memory(state)

        # Last 10 messages should be preserved after removals + summary
        older_count = 25 - 10  # 15
        last_10 = messages[-10:]
        assert result["messages"][older_count + 1 :] == last_10

    async def test_custom_threshold(self) -> None:
        """A custom threshold can be passed to control when summarization kicks in."""
        messages = [HumanMessage(content=f"Msg {i}", id=str(i)) for i in range(15)]
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            new_callable=AsyncMock,
            return_value="Short summary.",
        ):
            result = await manage_memory(state, threshold=5)

        assert result["memory_summarized"] is True
        assert "messages" in result
        # 5 older messages get RemoveMessage, then 1 summary + 10 recent
        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert len(removals) == 5
        # Find the summary message (first non-RemoveMessage)
        non_removals = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
        assert isinstance(non_removals[0], HumanMessage)

    async def test_llm_failure_falls_back(self) -> None:
        """LLM failure during summarization falls back to original messages."""
        messages = [HumanMessage(content=f"Msg {i}", id=str(i)) for i in range(30)]
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await manage_memory(state, threshold=20)

        assert result["memory_summarized"] is False
        assert "messages" not in result  # original messages unchanged
