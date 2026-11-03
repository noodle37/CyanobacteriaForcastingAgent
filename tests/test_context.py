"""官方 middleware：旧工具结果先清掉，更早的对话再摘要。"""

from __future__ import annotations

import unittest

from langchain.agents.middleware import ClearToolUsesEdit
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.hooks import Policy
from agent.react_graph import _CLEARED_TOOL, context_middleware


class SummaryChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "summary"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="摘要：保留预测结论"))])


def _tool_pair(name: str, call_id: str, content: str) -> list:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
        ),
        ToolMessage(content=content, tool_call_id=call_id, name=name),
    ]


class ContextMiddlewareTest(unittest.TestCase):
    def test_old_tool_results_are_cleared_and_the_latest_stays(self):
        messages = []
        for index in range(4):
            messages.extend(_tool_pair("python", f"call_{index}", "X" * 400))
        ClearToolUsesEdit(trigger=1, keep=3, placeholder=_CLEARED_TOOL).apply(
            messages,
            count_tokens=count_tokens_approximately,
        )
        cleared = [message.content for message in messages if isinstance(message, ToolMessage)]
        self.assertEqual(cleared[0], _CLEARED_TOOL)
        self.assertTrue(all(item != _CLEARED_TOOL for item in cleared[1:]))

    def test_summarize_replaces_older_messages_when_over_the_window(self):
        model = SummaryChat()
        policy = Policy(
            allowed_tools=(),
            allowed_models=(),
            context_window_tokens=400,
        )
        summary = context_middleware(model, policy)[0]
        messages = [
            HumanMessage(content="A" * 3000, id="old"),
            HumanMessage(content="最近这一步", id="new"),
        ]
        update = summary.before_model({"messages": messages}, None)
        self.assertIsNotNone(update)
        contents = [getattr(message, "content", "") for message in update["messages"]]
        self.assertTrue(any("摘要" in str(content) for content in contents))
        self.assertTrue(any("最近这一步" in str(content) for content in contents))


if __name__ == "__main__":
    unittest.main()
