"""A scripted chat model so the graph can be tested without any real LLM."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AIMessages in order and records every prompt it receives."""

    responses: list[AIMessage] = Field(default_factory=list)
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self  # the script already decides which tools get called

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if len(self.prompts) >= len(self.responses):
            raise RuntimeError("ScriptedChatModel ran out of responses")
        self.prompts.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses[len(self.prompts) - 1])])


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
