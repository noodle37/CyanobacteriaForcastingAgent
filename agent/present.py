"""把图状态收成 API 能直接返回的字典。"""

from __future__ import annotations

from agent.workers import _template_report


def present_state(state: dict, thread_id: str, *, finished: bool = True) -> dict:
    report = state.get("report") or _assistant_text(state.get("messages"))
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
        "report": report or "",
        "prepared": state.get("prepared"),
        "tool_trace": state.get("tool_trace") or [],
        "errors": state.get("errors") or [],
    }


def _assistant_text(messages) -> str:
    for message in reversed(list(messages or [])):
        if getattr(message, "tool_calls", None):
            continue
        if getattr(message, "type", None) not in (None, "ai"):
            continue
        text = _content_text(getattr(message, "content", ""))
        if text:
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
