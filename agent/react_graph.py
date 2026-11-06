"""LangChain create_agent。快到上下文窗口时，用官方 middleware 清旧工具结果并摘要更早的对话。

预算也走官方 middleware：模型调用或工具调用到上限时 jump_to end，
图正常收尾，已有的预测和评分留在状态里，不会被当成异常丢掉。
"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver

from agent.hooks import Policy
from agent.state import ReactLedger
from agent.tools import make_tools
from agent.workers import ToolClient

_DEFAULT_PROMPT = "你是时序预测 agent。多轮调用工具完成分析、预测和改进，最后用中文给出结论。"
_CLEARED_TOOL = "[旧的工具结果已从本次请求中清掉，最近的结果仍保留]"

# LangGraph 把每个中间件的 before_model / after_model 钩子各算作一个 superstep。
# 当前这套中间件的组合，一次模型调用大约要经过：
#   ModelCallLimit.before_model + Summarization.before_model + model
#   + ToolCallLimit.after_model + ModelCallLimit.after_model + tools
# 实测约 6 个 superstep。recursion_limit 必须按这个成本留够余量，
# 否则语义预算还没到，保险丝先烧了。
SUPERSTEP_PER_MODEL_CALL = 6


def compile_react(call_tool: ToolClient, model, policy: Policy, prompt: str, checkpointer=None, root: Path | None = None):
    tools = make_tools(call_tool, policy, root)
    text = (prompt or "").strip() or _DEFAULT_PROMPT
    return create_agent(
        model,
        tools,
        system_prompt=text,
        middleware=[*limit_middleware(policy), *context_middleware(model, policy)],
        state_schema=ReactLedger,
        checkpointer=checkpointer or MemorySaver(),
        name="react",
    )


def limit_middleware(policy: Policy) -> list:
    """把预算做成框架级中间件。

    超限时中间件返回 jump_to="end"，图走 END 正常结束，run 里已经算出来的
    预测、评分、报告都留在 checkpoint 里。recursion_limit 只是更外层的
    保险丝，正常情况下轮不到它兜底。

    只设 run_limit，不设 thread_limit：thread_limit 会跨请求累计，
    同一 thread 的第二次追问会被上一轮的计数直接掐死。
    """
    steps = max(int(getattr(policy, "max_react_steps", 12) or 12), 1)
    tool_calls = max(int(getattr(policy, "max_tool_calls", steps * 4) or steps * 4), 1)
    return [
        ModelCallLimitMiddleware(run_limit=steps, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=tool_calls, exit_behavior="end"),
    ]


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
