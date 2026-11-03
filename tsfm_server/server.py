"""FastMCP 工具面。Agent 只通过这三个工具接触模型。"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from shared.metrics import evaluate_forecast
from tsfm_server.registry import load_registry
from tsfm_server.runtime import ModelRuntime
from tsfm_server.settings import HOST, MODELS_PATH, PORT

log = logging.getLogger("tsfm_server")

_runtime: ModelRuntime | None = None


def runtime() -> ModelRuntime:
    global _runtime
    if _runtime is None:
        _runtime = ModelRuntime(load_registry(MODELS_PATH))
    return _runtime


mcp = MCPServer(
    "tsfm",
    instructions="本地时序基础模型与统计基线。先 list_models，再 forecast。",
)


@mcp.tool()
def list_models() -> dict:
    """列出已注册模型的能力元数据。不会加载权重。"""
    return {"models": runtime().list_cards()}


@mcp.tool()
def forecast(model: str, values: list[float], horizon: int, freq: str = "D") -> dict:
    """用指定模型预测未来 horizon 步。返回 point、q10、q90。"""
    if horizon < 1 or horizon > 64:
        raise ValueError("horizon 必须在 1 到 64 之间")
    if len(values) < 2 or len(values) > 8192:
        raise ValueError("values 长度必须在 2 到 8192 之间")
    log.info("forecast model=%s n=%s horizon=%s freq=%s", model, len(values), horizon, freq)
    return runtime().forecast(model, values, horizon, freq)


@mcp.tool()
def evaluate(
    point: list[float],
    actual: list[float],
    history: list[float],
    q10: list[float] | None = None,
    q90: list[float] | None = None,
    season: int = 1,
) -> dict:
    """计算 MAE、MASE、区间覆盖率和 CRPS 近似。"""
    return evaluate_forecast(point, actual, history, q10, q90, season)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("TSFM MCP http://%s:%s/mcp", HOST, PORT)
    mcp.run(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
