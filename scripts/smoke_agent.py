"""跑完整条图：选模、MCP 调 Chronos-2、基线、报告。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph
from agent.mcp_client import MCPToolClient
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
        {"select_model": read_prompt("select_model.md"), "report": read_prompt("report.md")},
    )
    config = {"configurable": {"thread_id": "smoke-meicheng"}}
    await graph.ainvoke(
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
        config,
    )
    state = await graph.aget_state(config)
    values = state.values
    print("model:", values.get("model_name"), values.get("model_reason"))
    print("forecast:", values.get("forecast"))
    print("baseline:", values.get("baseline"))
    print("metrics:", values.get("metrics"))
    print("baseline_metrics:", values.get("baseline_metrics"))
    print("report:", values.get("report"))
    print("errors:", values.get("errors"))
    if not values.get("forecast") or values["forecast"].get("model") != "chronos2":
        raise SystemExit("主预测没有走到 chronos2")


if __name__ == "__main__":
    asyncio.run(main())
