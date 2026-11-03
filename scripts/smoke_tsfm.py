"""经 MCP 调用本地模型。服务需已在 TSFM_MCP_URL 监听。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.mcp_client import MCPToolClient
from agent.settings import TSFM_MCP_URL


async def main() -> None:
    sample = json.loads((ROOT / "samples" / "meichengmatou_tail128.json").read_text(encoding="utf-8"))
    values = sample["values"][-96:]
    client = MCPToolClient(TSFM_MCP_URL)
    listed = await client.call("list_models", {})
    print("models:", [item["id"] for item in listed["models"] if item["enabled"]])
    for model in ("seasonal_naive", "chronos2"):
        result = await client.call(
            "forecast",
            {"model": model, "values": values, "horizon": 3, "freq": sample["freq"]},
        )
        print(
            f"{model}: point={['%.3f' % v for v in result['point']]} "
            f"q10={['%.3f' % v for v in result['q10']]} q90={['%.3f' % v for v in result['q90']]} "
            f"ms={result['elapsed_ms']:.0f} context={result.get('context_used')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
