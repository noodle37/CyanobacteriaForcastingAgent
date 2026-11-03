from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": LLM.enabled, "tsfm": TSFM_MCP_URL}


@app.post("/v1/runs")
async def create_run(req: RunRequest) -> dict:
    thread_id = req.thread_id or uuid.uuid4().hex
    config = _config(thread_id)
    await _graph.ainvoke(prepare_input(_payload(req, thread_id), react=_REACT), config)
    snapshot = await _graph.aget_state(config)
    body = _public(snapshot.values, thread_id, finished=not _interrupt_payloads(snapshot))
    interrupts = _interrupt_payloads(snapshot)
    if interrupts:
        body["status"] = "interrupt"
        body["interrupts"] = interrupts
    else:
        body["status"] = "completed"
    return body


@app.post("/v1/runs/stream")
async def stream_run(req: RunRequest) -> StreamingResponse:
    thread_id = req.thread_id or uuid.uuid4().hex

    async def events() -> AsyncIterator[str]:
        config = _config(thread_id)
        async for update in _graph.astream(prepare_input(_payload(req, thread_id), react=_REACT), config, stream_mode="updates"):
            yield _sse({"thread_id": thread_id, "update": update})
        snapshot = await _graph.aget_state(config)
        interrupts = _interrupt_payloads(snapshot)
        final = _public(snapshot.values, thread_id, finished=not interrupts)
        final["status"] = "interrupt" if interrupts else "completed"
        final["interrupts"] = interrupts
        yield _sse(final)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest) -> dict:
    config = _config(thread_id)
    await _graph.ainvoke(Command(resume={"approved": req.approved}), config)
    snapshot = await _graph.aget_state(config)
    interrupts = _interrupt_payloads(snapshot)
    body = _public(snapshot.values, thread_id, finished=not interrupts)
    body["status"] = "interrupt" if interrupts else "completed"
    body["interrupts"] = interrupts
    return body


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
