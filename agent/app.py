from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError, InvalidUpdateError
from pydantic import BaseModel, Field
from langgraph.types import Command

from agent.graph import build_graph, prepare_input, run_config
from agent.mcp_client import MCPToolClient
from agent.present import present_state
from agent.settings import AGENT_HOST, AGENT_PORT, LLM, POLICY, TSFM_MCP_URL, read_prompt

app = FastAPI(title="tsfm-agent", version="0.1.0")
_tools = MCPToolClient(TSFM_MCP_URL)
_prompts = {
    "select_model": read_prompt("select_model.md"),
    "report": read_prompt("report.md"),
    "react": read_prompt("react.md"),
}
_REACT = LLM.enabled
_graph = build_graph(_tools.call, LLM, POLICY, _prompts)


class RunRequest(BaseModel):
    user_request: str = Field(description="自然语言任务说明")
    series: list[float]
    horizon: int = 3
    freq: str = "D"
    actual: list[float] | None = None
    model_name: str | None = None
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    approved: bool
    thread_id: str


def _payload(req: RunRequest, thread_id: str) -> dict:
    return {
        "run_id": thread_id,
        "user_request": req.user_request,
        "series": req.series,
        "horizon": req.horizon,
        "freq": req.freq,
        "actual": req.actual,
        "model_name": req.model_name or "",
        "tool_trace": [],
        "errors": [],
    }


def _config(thread_id: str) -> dict:
    return run_config(thread_id, POLICY)


def _public(state: dict, thread_id: str, *, finished: bool = True) -> dict:
    return present_state(state, thread_id, finished=finished)


async def _run(config: dict, payload: dict) -> str | None:
    """跑一次图。正常结束返回 None；降级结束返回原因字符串。

    降级不等于失败：步数用尽、并发写冲突这类情况，checkpoint 里往往已经有
    可用的预测和评分，这里只把原因记下来，交给调用方决定怎么用。
    """
    try:
        await _graph.ainvoke(prepare_input(payload, react=_REACT), config)
        return None
    except GraphRecursionError:
        return "模型调用步数已达上限，已保留当前结果"
    except InvalidUpdateError as exc:
        return f"状态更新冲突: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


async def _snapshot(config: dict) -> tuple[dict, list]:
    """读 checkpoint。读状态本身也可能失败，失败时返回空状态而不是再抛一次。"""
    try:
        snapshot = await _graph.aget_state(config)
    except Exception:  # noqa: BLE001
        return {}, []
    return dict(snapshot.values or {}), _interrupt_payloads(snapshot)


def _finish_body(values: dict, thread_id: str, interrupts: list, degraded: str | None) -> dict:
    body = _public(values, thread_id, finished=not interrupts)
    if interrupts:
        body["status"] = "interrupt"
        body["interrupts"] = interrupts
    elif degraded:
        body["status"] = "partial"
        body["degraded_reason"] = degraded
        body["errors"] = [*(body.get("errors") or []), degraded]
    elif body.get("budget_exhausted"):
        # 限额中间件跳到 END，图正常收尾，但这次是被预算掐断的，标成 partial
        body["status"] = "partial"
        body["degraded_reason"] = body.get("budget_notice") or "预算用尽，返回部分结果"
    else:
        body["status"] = "completed"
        body["interrupts"] = []
    return body


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": LLM.enabled, "tsfm": TSFM_MCP_URL}


@app.post("/v1/runs")
async def create_run(req: RunRequest) -> dict:
    thread_id = req.thread_id or uuid.uuid4().hex
    config = _config(thread_id)
    degraded = await _run(config, _payload(req, thread_id))
    values, interrupts = await _snapshot(config)
    return _finish_body(values, thread_id, interrupts, degraded)


@app.post("/v1/runs/stream")
async def stream_run(req: RunRequest) -> StreamingResponse:
    thread_id = req.thread_id or uuid.uuid4().hex

    async def events() -> AsyncIterator[str]:
        config = _config(thread_id)
        degraded: str | None = None
        try:
            async for update in _graph.astream(
                prepare_input(_payload(req, thread_id), react=_REACT), config, stream_mode="updates"
            ):
                yield _sse({"thread_id": thread_id, "update": update})
        except GraphRecursionError:
            degraded = "模型调用步数已达上限，已保留当前结果"
        except InvalidUpdateError as exc:
            degraded = f"状态更新冲突: {exc}"
        except Exception as exc:  # noqa: BLE001
            degraded = f"{type(exc).__name__}: {exc}"
        values, interrupts = await _snapshot(config)
        yield _sse(_finish_body(values, thread_id, interrupts, degraded))
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest) -> dict:
    config = _config(thread_id)
    degraded: str | None = None
    try:
        await _graph.ainvoke(Command(resume={"approved": req.approved}), config)
    except GraphRecursionError:
        degraded = "模型调用步数已达上限，已保留当前结果"
    except InvalidUpdateError as exc:
        degraded = f"状态更新冲突: {exc}"
    except Exception as exc:  # noqa: BLE001
        degraded = f"{type(exc).__name__}: {exc}"
    values, interrupts = await _snapshot(config)
    return _finish_body(values, thread_id, interrupts, degraded)


def _interrupt_payloads(snapshot) -> list:
    found = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            found.append(getattr(item, "value", item))
    return found


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def main() -> None:
    import uvicorn

    uvicorn.run("agent.app:app", host=AGENT_HOST, port=AGENT_PORT, reload=False)


if __name__ == "__main__":
    main()
