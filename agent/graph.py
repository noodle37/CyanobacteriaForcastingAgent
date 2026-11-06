"""业务图。

配了大模型时走 LangChain create_agent：
同一轮可以发出多个 tool call，全部返回后模型再决定下一步。
没有大模型时退回固定流水线。
"""

from __future__ import annotations

import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.hooks import Policy
from agent.llm import LLMClient
from agent.react_graph import SUPERSTEP_PER_MODEL_CALL
from agent.state import TaskLedger
from agent.workers import ToolClient, make_workers


def build_graph(
    call_tool: ToolClient,
    llm: LLMClient,
    policy: Policy,
    prompts: dict[str, str],
    checkpointer=None,
    chat_model=None,
    root: Path | None = None,
):
    model = chat_model
    if model is None and getattr(llm, "enabled", False):
        model = llm.as_chat_model()
    if model is not None:
        from agent.react_graph import compile_react

        return compile_react(
            call_tool,
            model,
            policy,
            (prompts.get("react") or "").strip(),
            checkpointer=checkpointer,
            root=root,
        )
    return _build_legacy(call_tool, llm, policy, prompts, checkpointer)


def prepare_input(payload: dict, *, react: bool) -> dict:
    if not react or payload.get("messages"):
        return payload
    from langchain_core.messages import HumanMessage

    return {**payload, "messages": [HumanMessage(content=task_message(payload))]}


def run_config(thread_id: str, policy: Policy) -> dict:
    """两层预算。

    语义预算（模型调用次数、工具调用次数）由 react_graph.limit_middleware
    执行，超限时图正常走到 END，已有结果留在 checkpoint 里。

    recursion_limit 只是更外层的保险丝。LangGraph 把每个中间件钩子各算作
    一个 superstep，所以一次模型调用约 6 个 superstep（见
    react_graph.SUPERSTEP_PER_MODEL_CALL），保险丝必须按这个成本留足余量，
    让语义预算先触发；万一中间件没生效，它兜底抛 GraphRecursionError，
    app 层会捕获并把已有的部分结果照样返回，而不是 500。
    """
    rounds = max(int(getattr(policy, "max_react_steps", 12) or 12), 2)
    limit = rounds * SUPERSTEP_PER_MODEL_CALL + 8
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": limit}


def task_message(payload: dict) -> str:
    series: list[float] = []
    for value in payload.get("series") or []:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            break
    brief = {
        "request": payload.get("user_request") or "",
        "horizon": payload.get("horizon"),
        "freq": payload.get("freq") or "D",
        "n": len(series),
        "head": series[:8],
        "tail": series[-16:],
        "actual": payload.get("actual"),
        "model_hint": payload.get("model_name") or None,
    }
    return (
        "这是一次时序预测任务。完整序列在 python 的 series 里，forecast 省略 values 时用这份序列。\n"
        + json.dumps(brief, ensure_ascii=False)
    )


def _build_legacy(call_tool: ToolClient, llm: LLMClient, policy: Policy, prompts: dict[str, str], checkpointer=None):
    nodes = make_workers(call_tool, llm, policy, prompts)
    graph = StateGraph(TaskLedger)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", _after_ingest, {"select_model": "select_model", "report": "report"})
    graph.add_conditional_edges("select_model", _after_select, {"forecast": "forecast", "baseline": "baseline", "report": "report"})
    graph.add_conditional_edges("forecast", _after_forecast, {"baseline": "baseline", "report": "report"})
    graph.add_edge("baseline", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def _after_ingest(state: TaskLedger) -> str:
    return "report" if state.get("route") == "report" else "select_model"


def _after_select(state: TaskLedger) -> str:
    route = state.get("route")
    if route == "baseline":
        return "baseline"
    if route == "report":
        return "report"
    return "forecast"


def _after_forecast(state: TaskLedger) -> str:
    return "report" if state.get("route") == "report" else "baseline"
