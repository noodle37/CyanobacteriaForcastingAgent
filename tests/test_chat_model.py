"""一轮补全可以带多个 tool call。"""

from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agent.chat_model import OpenAICompatibleChat


@tool
def ping(text: str) -> str:
    """回声。"""
    return text


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, call_id, name, arguments="{}"):
        self.id = call_id
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls


class _Response:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class _Completions:
    def __init__(self, box):
        self.box = box

    def create(self, **kwargs):
        self.box["kwargs"] = kwargs
        message = _Message([
            _Call("call_a", "list_models", "{}"),
            _Call("call_b", "python", '{"code": "result = 1"}'),
        ])
        return _Response(message)


class _Client:
    def __init__(self, box):
        self.chat = type("Chat", (), {"completions": _Completions(box)})()


class TwoCallChat(OpenAICompatibleChat):
    def _openai(self):
        return _Client(self._box)


class ChatModelTest(unittest.TestCase):
    def test_one_response_carries_two_tool_calls(self):
        box = {}
        model = TwoCallChat(base_url="http://example", api_key="test", llm_model="fake", timeout=1)
        model._box = box
        message = model.bind_tools([ping]).invoke([HumanMessage(content="hi")])
        self.assertEqual([call["name"] for call in message.tool_calls], ["list_models", "python"])
        self.assertEqual(len(box["kwargs"]["tools"]), 1)
        self.assertEqual(box["kwargs"]["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
