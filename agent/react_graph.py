"""LangChain create_agent。快到上下文窗口时，用官方 middleware 清旧工具结果并摘要更早的对话。"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver

from agent.hooks import Policy
from agent.state import ReactLedger
from agent.tools import make_tools
from agent.workers import ToolClient

_DEFAULT_PROMPT = "你是时序预测 agent。多轮调用工具完成分析、预测和改进，最后用中文给出结论。"
_CLEARED_TOOL = "[旧的工具结果已从本次请求中清掉，最近的结果仍保留]"


def compile_react(call_tool: ToolClient, model, policy: Policy, prompt: str, checkpointer=None, root: Path | None = None):
    tools = make_tools(call_tool, policy, root)
    text = (prompt or "").strip() or _DEFAULT_PROMPT
    return create_agent(
        model,
        tools,
        system_prompt=text,
        middleware=context_middleware(model, policy),
        state_schema=ReactLedger,
        checkpointer=checkpointer or MemorySaver(),
        name="react",
    )


def context_middleware(model, policy: Policy) -> list:
    """窗口的 60% 清掉旧工具结果，75% 再把更早的对话摘要掉，最近约 30% 保持原文。"""
    window = max(int(policy.context_window_tokens), 1)
    clear_at = max(1, int(window * 0.60))
    summary_at = max(clear_at + 1, int(window * 0.75))
    keep_tokens = max(1, int(window * 0.30))
    return [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", summary_at),
            keep=("tokens", keep_tokens),
        ),
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=clear_at, keep=3, placeholder=_CLEARED_TOOL)],
        ),
    ]
