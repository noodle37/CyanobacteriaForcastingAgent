"""ReAct：一轮多个工具，下一轮能看到上一轮写回的序列。"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command
from pydantic import Field

from agent.graph import build_graph, prepare_input, task_message
from agent.hooks import Policy
from agent.llm import LLMClient
from agent.present import present_state


class ScriptedChat(BaseChatModel):
    steps: list[Any] = Field(default_factory=list)
    cursor: list[int] = Field(default_factory=lambda: [0])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = self.cursor[0]
        self.cursor[0] = index + 1
        step = self.steps[index] if index < len(self.steps) else "停止"
        if isinstance(step, str):
            message = AIMessage(content=step)
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{index}_{pos}", "type": "tool_call"}
                    for pos, (name, args) in enumerate(step)
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class ScriptedTools:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "list_models":
            return {
                "models": [
                    {"id": "chronos2", "enabled": True, "context_length": 32, "description": "chronos"},
                    {"id": "seasonal_naive", "enabled": True, "context_length": 32, "description": "naive"},
                ]
            }
        if name == "forecast":
            horizon = int(args["horizon"])
            last = float(args["values"][-1])
            return {"model": args["model"], "point": [last] * horizon, "q10": [0.0] * horizon, "q90": [1.0] * horizon, "freq": args["freq"]}
        if name == "evaluate":
            return {"mae": 0.5, "mase": 0.8, "coverage": 1.0, "crps_proxy": 0.2, "season": args["season"], "horizon": len(args["point"])}
        if name == "reload_models":
            return {"reloaded": True, "models": ["seasonal_naive"]}
        raise AssertionError(name)


def _policy(**overrides) -> Policy:
    raw = dict(
        allowed_tools=("list_models", "forecast", "evaluate", "python", "reload_models"),
        allowed_models=("chronos2", "timesfm3", "seasonal_naive"),
        max_series_length=32,
        context_tail=8,
        horizon_min=1,
        horizon_max=8,
        max_abs_value=1e6,
        require_approval_models=(),
        python_timeout_s=10,
    )
    raw.update(overrides)
    return Policy(**raw)


def _payload(**overrides) -> dict:
    base = {
        "user_request": "先分析再预测",
        "series": [1.0, 2, 3, 4],
        "horizon": 2,
        "freq": "D",
        "actual": [5.0, 6],
        "model_name": "",
        "tool_trace": [],
        "errors": [],
    }
    base.update(overrides)
    return base


def _run(steps, payload, policy=None, root=None):
    policy = policy or _policy()
    tools = ScriptedTools()
    graph = build_graph(
        tools,
        LLMClient("", "", "", 1),
        policy,
        {"react": "测试"},
        chat_model=ScriptedChat(steps=steps),
        root=root,
    )
    config = {"configurable": {"thread_id": "react"}, "recursion_limit": 20}
    result = asyncio.run(graph.ainvoke(prepare_input(payload, react=True), config))
    return result, tools, graph, config


class ReactTest(unittest.TestCase):
    def test_task_message_keeps_only_the_edges(self):
        text = task_message({"series": list(range(100)), "horizon": 2, "user_request": "看尾部", "freq": "D"})
        self.assertIn("99", text)
        self.assertNotIn("50", text)

    def test_one_turn_can_call_two_tools_then_continue(self):
        result, tools, _graph, _config = _run(
            [
                [("list_models", {}), ("python", {"code": "result = len(series)"})],
                [("forecast", {"model": "chronos2"})],
                [("evaluate", {})],
                "主模型已评分",
            ],
            _payload(),
        )
        messages = [message for message in result["messages"] if isinstance(message, AIMessage)]
        self.assertEqual({call["name"] for call in messages[0].tool_calls}, {"list_models", "python"})
        self.assertEqual([call["name"] for call in messages[1].tool_calls], ["forecast"])
        self.assertEqual([call["name"] for call in messages[2].tool_calls], ["evaluate"])
        self.assertFalse(messages[3].tool_calls)
        self.assertEqual([call[0] for call in tools.calls], ["list_models", "forecast", "evaluate"])
        self.assertEqual(result["metrics"]["mae"], 0.5)
        self.assertIn("主模型已评分", present_state(result, "react")["report"])

    def test_python_series_is_used_by_the_next_forecast(self):
        code = "series[:] = [float(i) for i in range(1, 9)]\nresult = series[-1]"
        result, tools, _graph, _config = _run(
            [
                [("python", {"code": code})],
                [("forecast", {"model": "seasonal_naive"})],
                "点预测已完成",
            ],
            _payload(series=[0.0, 0.0], actual=None),
        )
        self.assertEqual(result["series"], [1.0, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(tools.calls[0][0], "forecast")
        self.assertEqual(tools.calls[0][1]["values"][-1], 8.0)
        self.assertEqual(result["baseline"]["point"], [8.0, 8.0])

    def test_python_write_reloads_before_returning(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "write_file('tsfm_server/adapters/probe_marker.py', 'MARKER = 1\\n')\nresult = 'wrote'"
            _result, tools, _graph, _config = _run(
                [
                    [("python", {"code": code})],
                    "源码已重载",
                ],
                _payload(actual=None),
                root=root,
            )
            self.assertEqual((root / "tsfm_server/adapters/probe_marker.py").read_text(encoding="utf-8"), "MARKER = 1\n")
            self.assertEqual(tools.calls, [("reload_models", {})])

    def test_python_error_is_fed_back_to_the_model(self):
        result, tools, _graph, _config = _run(
            [
                [("python", {"code": "import os\nresult = 1"})],
                "分析失败后结束",
            ],
            _payload(actual=None),
        )
        self.assertEqual(tools.calls, [])
        self.assertTrue(any("不允许导入" in item for item in result["errors"]))
        self.assertIn("分析失败后结束", present_state(result, "react")["report"])

    def test_forecast_interrupt_can_deny_or_approve(self):
        policy = _policy(require_approval_models=("chronos2",))
        denied_tools = ScriptedTools()
        denied = build_graph(
            denied_tools,
            LLMClient("", "", "", 1),
            policy,
            {"react": "测试"},
            chat_model=ScriptedChat(steps=[[("forecast", {"model": "chronos2"})], "已拒绝"]),
        )
        config = {"configurable": {"thread_id": "deny"}, "recursion_limit": 20}
        payload = prepare_input(_payload(), react=True)
        asyncio.run(denied.ainvoke(payload, config))
        paused = asyncio.run(denied.aget_state(config))
        self.assertTrue(paused.next)
        self.assertEqual(paused.tasks[0].interrupts[0].value["tool"], "forecast")
        asyncio.run(denied.ainvoke(Command(resume={"approved": False}), config))
        finished = asyncio.run(denied.aget_state(config))
        self.assertEqual(denied_tools.calls, [])
        self.assertIn("未获批准", "".join(finished.values["errors"]))
        self.assertFalse(finished.next)

        approved_tools = ScriptedTools()
        approved = build_graph(
            approved_tools,
            LLMClient("", "", "", 1),
            policy,
            {"react": "测试"},
            chat_model=ScriptedChat(steps=[[("forecast", {"model": "chronos2"})], "已批准"]),
        )
        ok_config = {"configurable": {"thread_id": "allow"}, "recursion_limit": 20}
        asyncio.run(approved.ainvoke(prepare_input(_payload(), react=True), ok_config))
        asyncio.run(approved.ainvoke(Command(resume={"approved": True}), ok_config))
        done = asyncio.run(approved.aget_state(ok_config))
        self.assertEqual(approved_tools.calls[0][1]["model"], "chronos2")
        self.assertEqual(done.values["forecast"]["model"], "chronos2")


if __name__ == "__main__":
    unittest.main()
