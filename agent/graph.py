"""手写业务图。LangGraph 提供运行时、条件边、checkpoint 和 interrupt。"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.llm import LLMClient
from agent.hooks import Policy
from agent.state import TaskLedger
from agent.workers import ToolClient, make_workers


def build_graph(call_tool: ToolClient, llm: LLMClient, policy: Policy, prompts: dict[str, str], checkpointer=None):
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
