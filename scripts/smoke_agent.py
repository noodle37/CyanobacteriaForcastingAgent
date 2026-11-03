"""跑完整条图：选模、MCP 调 Chronos-2、基线、报告。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph, prepare_input, run_config
from agent.mcp_client import MCPToolClient
from agent.present import present_state
from agent.settings import LLM, POLICY, TSFM_MCP_URL, read_prompt


async def main() -> None:
    sample = json.loads((ROOT / "samples" / "meichengmatou_tail128.json").read_text(encoding="utf-8"))
    history = sample["values"][:-3]
    actual = sample["values"][-3:]
    tools = MCPToolClient(TSFM_MCP_URL)
    graph = build_graph(
        tools.call,
        LLM,
        POLICY,
        {
            "select_model": read_prompt("select_model.md"),
            "report": read_prompt("report.md"),
            "react": read_prompt("react.md"),
        },
    )
    config = run_config("smoke-meicheng", POLICY)
    await graph.ainvoke(
        prepare_input(
            {
                "run_id": "smoke-meicheng",
                "user_request": "预测梅城码头叶绿素未来 3 天，并和基线比较",
                "series": history,
                "horizon": 3,
                "freq": "D",
                "actual": actual,
                "model_name": "",
                "tool_trace": [],
                "errors": [],
            },
            react=LLM.enabled,
        ),
        config,
    )
    state = await graph.aget_state(config)
    values = state.values
    shown = present_state(values, "smoke-meicheng")
    print("model:", shown.get("model_name"), shown.get("model_reason"))
    print("tools:", [item.get("tool") for item in shown.get("tool_trace") or []])
    print("forecast:", shown.get("forecast"))
    print("baseline:", shown.get("baseline"))
    print("metrics:", shown.get("metrics"))
    print("baseline_metrics:", shown.get("baseline_metrics"))
    print("report:", shown.get("report"))
    print("errors:", shown.get("errors"))
    if LLM.enabled:
        if not (shown.get("forecast") or shown.get("baseline") or shown.get("report")):
            raise SystemExit("ReAct 没有产出")
    elif not shown.get("forecast") or shown["forecast"].get("model") != "chronos2":
        raise SystemExit("主预测没有走到 chronos2")


if __name__ == "__main__":
    asyncio.run(main())
