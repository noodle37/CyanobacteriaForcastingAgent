"""LangGraph 自带的 ReAct。一轮里的多个 tool call 一起执行，然后回到模型。"""

from __future__ import annotations

import warnings
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.warnings import LangGraphDeprecatedSinceV10

from agent.hooks import Policy
from agent.state import ReactLedger
from agent.tools import make_tools
from agent.workers import ToolClient

_DEFAULT_PROMPT = "你是时序预测 agent。多轮调用工具完成分析、预测和改进，最后用中文给出结论。"


def compile_react(call_tool: ToolClient, model, policy: Policy, prompt: str, checkpointer=None, root: Path | None = None):
    tools = make_tools(call_tool, policy, root)
    text = (prompt or "").strip() or _DEFAULT_PROMPT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangGraphDeprecatedSinceV10)
        return create_react_agent(
            model,
            ToolNode(tools),
            prompt=text,
            state_schema=ReactLedger,
            checkpointer=checkpointer or MemorySaver(),
            version="v1",
        )
