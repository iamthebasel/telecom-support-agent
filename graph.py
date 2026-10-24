"""The multi-agent LangGraph workflow.

    START -> supervisor --knowledge--> knowledge_agent -> END
                        --account----> account_agent <-> tools (loop) -> END
                        --other------> other -> END
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .prompts import ACCOUNT_SYSTEM, KNOWLEDGE_SYSTEM, MISSING_CUSTOMER_ID, OTHER_REPLY
from .retriever import KnowledgeBase, RetrievedChunk
from .router import CUSTOMER_ID


class SupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str | None
    route: str
    route_reason: str
    sources: list[str]


def last_user_text(state: SupportState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def conversation_history(state: SupportState, limit: int = 6) -> list[AnyMessage]:
    """Recent human/assistant turns without tool-call plumbing (for the knowledge agent)."""
    turns = [m for m in state["messages"]
             if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]
    return turns[-limit:]


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.source}] {c.section}\n{c.text}" for c in chunks)


def build_graph(llm: BaseChatModel, kb: KnowledgeBase, tools: list[BaseTool], router,
                top_k: int = 4, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)

    def supervisor(state: SupportState) -> dict:
        text = last_user_text(state)
        decision = router.route(text)
        update = {"route": decision.route, "route_reason": decision.reason, "sources": []}
        # pick up a customer ID mentioned in the message (it overrides any earlier one)
        match = CUSTOMER_ID.search(text)
        if match:
            update["customer_id"] = match.group(0).upper()
        return update

    def knowledge_agent(state: SupportState) -> dict:
        chunks = kb.search(last_user_text(state), k=top_k)
        system = SystemMessage(KNOWLEDGE_SYSTEM.format(context=format_context(chunks)))
        answer = llm.invoke([system, *conversation_history(state)])
        sources = list(dict.fromkeys(c.source for c in chunks))  # unique, in rank order
        return {"messages": [answer], "sources": sources}

    def account_agent(state: SupportState) -> dict:
        customer_id = state.get("customer_id")
        if not customer_id:
            return {"messages": [AIMessage(MISSING_CUSTOMER_ID)]}
        system = SystemMessage(ACCOUNT_SYSTEM.format(customer_id=customer_id))
        return {"messages": [llm_with_tools.invoke([system, *state["messages"]])]}

    def other(state: SupportState) -> dict:
        return {"messages": [AIMessage(OTHER_REPLY)]}

    graph = StateGraph(SupportState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("knowledge_agent", knowledge_agent)
    graph.add_node("account_agent", account_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("other", other)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["route"],
                                {"knowledge": "knowledge_agent", "account": "account_agent", "other": "other"})
    # ReAct loop: the account agent keeps calling tools until it answers without a tool call
    graph.add_conditional_edges("account_agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "account_agent")
    graph.add_edge("knowledge_agent", END)
    graph.add_edge("other", END)

    return graph.compile(checkpointer=checkpointer)
