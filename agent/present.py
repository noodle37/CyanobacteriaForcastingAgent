"""把图状态收成 API 能直接返回的字典。"""

from __future__ import annotations

from agent.workers import _template_report

# LangChain 限额中间件在跳到 END 时注入的提示文案前缀。
# 它是一条 AI 消息，但不是模型的业务结论：选报告文本时要跳过它，
# 同时把它当作「本次被预算掐断」的信号。
_LIMIT_NOTICE_PREFIXES = (
    "model call limits exceeded",
    "tool call limit reached",
)


def present_state(state: dict, thread_id: str, *, finished: bool = True) -> dict:
    notices = _limit_notices(state.get("messages"))
    report = state.get("report") or _assistant_text(state.get("messages"), skip=notices)
    if not report and finished:
        report = _template_report(state, state.get("metrics"), state.get("baseline_metrics"), [])
    return {
        "thread_id": thread_id,
        "model_name": state.get("model_name"),
        "model_reason": state.get("model_reason"),
        "series": state.get("series"),
        "forecast": state.get("forecast"),
        "baseline": state.get("baseline"),
        "metrics": state.get("metrics"),
        "baseline_metrics": state.get("baseline_metrics"),
        "forecasts": state.get("forecasts") or [],
        "evaluations": state.get("evaluations") or [],
        "budget_exhausted": bool(notices),
        "budget_notice": notices[0] if notices else None,
        "report": report or "",
        "prepared": state.get("prepared"),
        "tool_trace": state.get("tool_trace") or [],
        "errors": state.get("errors") or [],
    }


def _limit_notices(messages) -> list[str]:
    """挑出中间件注入的限流提示。"""
    found = []
    for message in list(messages or []):
        if getattr(message, "type", None) not in (None, "ai"):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = _content_text(getattr(message, "content", ""))
        if text and _is_limit_notice(text):
            found.append(text)
    return found


def _is_limit_notice(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _LIMIT_NOTICE_PREFIXES)


def _assistant_text(messages, skip: list[str] | None = None) -> str:
    skipped = {item for item in (skip or [])}
    for message in reversed(list(messages or [])):
        if getattr(message, "tool_calls", None):
            continue
        if getattr(message, "type", None) not in (None, "ai"):
            continue
        text = _content_text(getattr(message, "content", ""))
        if not text or text in skipped:
            continue
        return text
    return ""


def _content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return ""
