"""图的路由测试。工具是假的，不加载模型。"""

from __future__ import annotations

import asyncio
import unittest

from langgraph.types import Command

from agent.graph import build_graph
from agent.hooks import Policy
from agent.llm import LLMClient


class FakeLLM(LLMClient):
    def __init__(self):
        super().__init__("", "", "", 1)


class ScriptedTools:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        if name == "list_models":
            return {
                "models": [
                    {"id": "chronos2", "enabled": True, "context_length": 8192, "quantiles": True, "device": "cuda", "task": "forecast", "description": "chronos"},
                    {"id": "timesfm3", "enabled": True, "context_length": 1024, "quantiles": True, "device": "cuda", "task": "forecast", "description": "timesfm"},
                    {"id": "seasonal_naive", "enabled": True, "context_length": 2048, "quantiles": True, "device": "cpu", "task": "forecast", "description": "naive"},
                ]
            }
        if name == "forecast":
            horizon = int(args["horizon"])
            return {
                "model": args["model"],
                "point": [float(args["values"][-1])] * horizon,
                "q10": [0.0] * horizon,
                "q90": [1.0] * horizon,
                "elapsed_ms": 1.0,
                "freq": args["freq"],
            }
        if name == "evaluate":
            return {"mae": 0.5, "mase": 0.8, "coverage": 1.0, "crps_proxy": 0.2, "season": args["season"], "horizon": len(args["point"])}
        raise AssertionError(name)


def _policy(**overrides) -> Policy:
    raw = dict(
        allowed_tools=("list_models", "forecast", "evaluate"),
        allowed_models=("chronos2", "timesfm3", "seasonal_naive"),
        max_series_length=32,
        context_tail=8,
        horizon_min=1,
        horizon_max=8,
        max_abs_value=1e6,
        require_approval_models=(),
    )
    raw.update(overrides)
    return Policy(**raw)


def _run(policy: Policy, payload: dict, tools: ScriptedTools | None = None):
    tools = tools or ScriptedTools()
    graph = build_graph(tools, FakeLLM(), policy, {"select_model": "选模", "report": "报告"})
    thread = "test"
    result = asyncio.run(graph.ainvoke({**payload, "tool_trace": [], "errors": []}, {"configurable": {"thread_id": thread}}))
    return result, tools, graph


class PipelineTest(unittest.TestCase):
    def test_default_model_calls_chronos_then_baseline(self):
        result, tools, _graph = _run(
            _policy(),
            {
                "user_request": "预测未来 3 天",
                "series": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "horizon": 3,
                "freq": "D",
                "actual": [11, 12, 13],
            },
        )
        names = [call[0] for call in tools.calls]
        self.assertEqual(names, ["list_models", "forecast", "forecast", "evaluate", "evaluate"])
        self.assertEqual(tools.calls[1][1]["model"], "chronos2")
        self.assertEqual(tools.calls[2][1]["model"], "seasonal_naive")
        self.assertEqual(len(tools.calls[1][1]["values"]), 8)
        self.assertIn("chronos2", result["report"])
        self.assertIsNotNone(result["metrics"])

    def test_bad_series_skips_models(self):
        result, tools, _graph = _run(
            _policy(),
            {"user_request": "坏数据", "series": [1.0], "horizon": 3, "freq": "D"},
        )
        self.assertEqual(tools.calls, [])
        self.assertIn("至少需要 2", result["report"])

    def test_baseline_only_skips_foundation_model(self):
        result, tools, _graph = _run(
            _policy(),
            {"user_request": "只要基线", "series": [1.0, 2, 3, 4], "horizon": 2, "freq": "D"},
        )
        models = [call[1].get("model") for call in tools.calls if call[0] == "forecast"]
        self.assertEqual(models, ["seasonal_naive"])
        self.assertIsNone(result.get("forecast"))
        self.assertEqual(result["baseline"]["model"], "seasonal_naive")

    def test_hint_selects_timesfm(self):
        _result, tools, _graph = _run(
            _policy(),
            {
                "user_request": "用指定模型",
                "series": [1.0, 2, 3, 4],
                "horizon": 2,
                "freq": "D",
                "model_name": "timesfm3",
            },
        )
        self.assertEqual(tools.calls[1][1]["model"], "timesfm3")

    def test_interrupt_before_forecast(self):
        tools = ScriptedTools()
        policy = _policy(require_approval_models=("chronos2",))
        graph = build_graph(tools, FakeLLM(), policy, {"select_model": "选模", "report": "报告"})
        config = {"configurable": {"thread_id": "approve"}}
        payload = {
            "user_request": "需要确认",
            "series": [1.0, 2, 3, 4, 5],
            "horizon": 2,
            "freq": "D",
            "model_name": "chronos2",
            "tool_trace": [],
            "errors": [],
        }
        asyncio.run(graph.ainvoke(payload, config))
        paused = asyncio.run(graph.aget_state(config))
        self.assertTrue(paused.next)
        self.assertEqual([call[0] for call in tools.calls], ["list_models"])
        asyncio.run(graph.ainvoke(Command(resume={"approved": False}), config))
        finished = asyncio.run(graph.aget_state(config))
        self.assertEqual([call[0] for call in tools.calls], ["list_models"])
        self.assertIn("未获批准", finished.values["report"])
        self.assertFalse(finished.next)


if __name__ == "__main__":
    unittest.main()
