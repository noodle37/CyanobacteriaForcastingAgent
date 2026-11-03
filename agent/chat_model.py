"""OpenAI 兼容接口上的 LangChain chat model，供 ReAct bind_tools 使用。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


class OpenAICompatibleChat(BaseChatModel):
    base_url: str
    api_key: str = Field(repr=False)
    llm_model: str
    timeout: float = 60

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _openai(self):
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload: dict[str, Any] = {
            "model": self.llm_model,
            "temperature": 0,
            "messages": [_to_openai(message) for message in messages],
        }
        if stop:
            payload["stop"] = stop
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        response = self._openai().chat.completions.create(**payload)
        message = response.choices[0].message
        return ChatResult(generations=[ChatGeneration(message=_ai_message(message))])


def _to_openai(message: BaseMessage) -> dict:
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text(message.content),
        }
    role = {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, message.type)
    body: dict[str, Any] = {"role": role, "content": _text(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        body["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                },
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return body


def _ai_message(message) -> AIMessage:
    tool_calls = []
    for index, call in enumerate(_tool_calls(message)):
        call_id, name, arguments = call
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool_calls.append({"id": call_id or f"call_{index}", "name": name, "args": args, "type": "tool_call"})
    return AIMessage(content=_text(getattr(message, "content", "")), tool_calls=tool_calls)


def _tool_calls(message) -> list[tuple[str, str, str]]:
    found = []
    for call in getattr(message, "tool_calls", None) or []:
        if isinstance(call, dict):
            function = call.get("function") or {}
            found.append((str(call.get("id") or ""), str(function.get("name") or ""), str(function.get("arguments") or "{}")))
        else:
            function = call.function
            found.append((str(call.id or ""), str(function.name or ""), str(function.arguments or "{}")))
    return found


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
