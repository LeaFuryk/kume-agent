"""LangGraph StateGraph definition for the Kume agent pipeline.

Wires all nodes into a graph with guardrail routing:

    manage_memory -> input_guardrail -> agent -> format_response -> output_guardrail -> END
                          |                                              |
                          v                                              v
                    block_response                                 block_response

The output guardrail runs AFTER format_response so it validates the final
text the user will see, preventing the formatter LLM from introducing
unsafe content that bypasses the guardrail.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from kume.services.nodes import (
    block_response,
    format_response,
    input_guardrail,
    manage_memory,
    output_guardrail,
)
from kume.services.state import KumeGraphState


def _route_after_input_guardrail(state: dict[str, Any]) -> str:
    """Route to agent if input is safe, otherwise block."""
    if state.get("input_safe", True):
        return "agent"
    return "block_response"


def _route_after_output_guardrail(state: dict[str, Any]) -> str:
    """Route to END if output is safe, otherwise block."""
    if state.get("output_safe", True):
        return END
    return "block_response"


def build_graph(
    agent_runnable: Any = None,
    tools: list[Any] | None = None,
    memory_threshold: int = 20,
) -> Any:
    """Build and compile the Kume agent graph.

    Parameters
    ----------
    agent_runnable:
        The agent node callable (e.g. a ``create_react_agent`` graph or an
        async function that takes state and returns a state update dict).
    tools:
        Unused directly by the graph but accepted for forward-compatibility.
    memory_threshold:
        Number of messages before the memory management node triggers
        summarization.

    Returns
    -------
    A compiled LangGraph ``CompiledGraph`` ready for ``ainvoke``.
    """
    graph = StateGraph(KumeGraphState)

    # Wrap manage_memory to inject the threshold parameter
    async def memory_node(state: dict[str, Any]) -> dict[str, Any]:
        return await manage_memory(state, threshold=memory_threshold)

    graph.add_node("manage_memory", memory_node)  # type: ignore[type-var]
    graph.add_node("input_guardrail", input_guardrail)  # type: ignore[type-var]
    graph.add_node("agent", agent_runnable)  # type: ignore[type-var]
    graph.add_node("output_guardrail", output_guardrail)  # type: ignore[type-var]
    graph.add_node("format_response", format_response)  # type: ignore[type-var]
    graph.add_node("block_response", block_response)  # type: ignore[type-var]

    graph.set_entry_point("manage_memory")
    graph.add_edge("manage_memory", "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {"agent": "agent", "block_response": "block_response"},
    )
    graph.add_edge("agent", "format_response")
    graph.add_edge("format_response", "output_guardrail")
    graph.add_conditional_edges(
        "output_guardrail",
        _route_after_output_guardrail,
        {END: END, "block_response": "block_response"},
    )
    graph.add_edge("block_response", END)

    return graph.compile()
