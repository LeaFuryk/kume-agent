"""Tests for the manage_memory graph node."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kume.services.nodes.manage_memory import manage_memory


class TestManageMemory:
    """manage_memory node compresses conversation history when it grows too long."""

    def test_passthrough_for_short_history(self) -> None:
        """When messages count <= threshold, no summarization occurs."""
        messages = [HumanMessage(content=f"Message {i}") for i in range(10)]
        state = {"messages": messages}

        result = manage_memory(state)

        assert result["memory_summarized"] is False
        # Should NOT include "messages" key — no mutation needed
        assert "messages" not in result

    def test_summarizes_long_history(self) -> None:
        """When messages exceed threshold, older messages are summarized."""
        messages = []
        for i in range(25):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}"))
            else:
                messages.append(AIMessage(content=f"AI response {i}"))
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            return_value="Summary of older conversation about nutrition topics.",
        ):
            result = manage_memory(state)

        assert result["memory_summarized"] is True
        assert "messages" in result
        # SystemMessage (summary) + last 10 messages = 11 total
        assert len(result["messages"]) == 11
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Summary" in result["messages"][0].content

    def test_preserves_last_10(self) -> None:
        """The last 10 messages are kept verbatim after summarization."""
        messages = []
        for i in range(25):
            messages.append(HumanMessage(content=f"Msg-{i}"))
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            return_value="Conversation summary.",
        ):
            result = manage_memory(state)

        # Last 10 messages should be preserved exactly
        last_10 = messages[-10:]
        assert result["messages"][1:] == last_10

    def test_custom_threshold(self) -> None:
        """A custom threshold can be passed to control when summarization kicks in."""
        messages = [HumanMessage(content=f"Msg {i}") for i in range(8)]
        state = {"messages": messages}

        with patch(
            "kume.services.nodes.manage_memory._call_summarize_llm",
            return_value="Short summary.",
        ):
            result = manage_memory(state, threshold=5)

        assert result["memory_summarized"] is True
        assert "messages" in result
        assert isinstance(result["messages"][0], SystemMessage)
