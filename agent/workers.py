"""固定流水线的五个 worker。没有大模型时使用。配了大模型则走 ReAct，不经过这里。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from langgraph.types import interrupt

from agent.hooks import Policy, PolicyError, authorize_forecast, check_request, finite_series
from agent.llm import LLMClient, parse_json_object
from agent.state import TaskLedger

ToolClient = Callable[[str, dict], Awaitable[dict]]

_BASELINE_MARKS = ("只看基线", "只要基线", "仅基线", "baseline only")


def season_of(freq: str) -> int:
    return {"D": 7, "W": 1, "H": 24, "M": 12}.get((freq or "D").upper(), 1)


def make_workers(call_tool: ToolClient, llm: LLMClient, policy: Policy, prompts: dict[str, str]):
    async def ingest(state: TaskLedger) -> dict:
        freq = state.get("freq") or "D"
        request = state.get("user_request") or ""
        try:
            series = finite_series(list(state.get("series") or []))
            horizon = int(state.get("horizon"))
            check_request(series, horizon, policy)
        except (PolicyError, TypeError, ValueError) as exc:
            return {"route": "report", "freq": freq, "user_request": request, "errors": [str(exc)]}
        return {
            "route": "select_model",
            "series": series,
            "horizon": horizon,
            "freq": freq,
            "user_request": request,
        }

    async def select_model(state: TaskLedger) -> dict:
        try:
            listed = await call_tool("list_models", {})
        except Exception as exc:
            return {"route": "report", "errors": [f"list_models 失败: {exc}"]}
        cards = [item for item in listed.get("models", []) if item.get("enabled")]
        by_id = {item["id"]: item for item in cards}
        allowed = [item_id for item_id in by_id if item_id in policy.allowed_models]
        hint = (state.get("model_name") or "").strip()
        request = state.get("user_request") or ""
        trace = [{"tool": "list_models", "allowed": True, "models": allowed}]

        if _wants_baseline(request):
            if "seasonal_naive" not in allowed:
                return {"route": "report", "catalog": cards, "tool_trace": trace, "errors": ["基线模型未启用"]}
            return {
                "route": "baseline",
                "model_name": "seasonal_naive",
                "model_reason": "用户要求只看基线",
                "model_card": by_id["seasonal_naive"],
                "catalog": cards,
                "tool_trace": trace,
            }

        choice = ""
        reason = ""
        if llm.enabled:
            try:
                raw = llm.complete(
                    prompts["select_model"],
                    json.dumps(
                        {
                            "request": request,
                            "hint": hint or None,
                            "horizon": state.get("horizon"),
                            "freq": state.get("freq"),
                            "n": len(state.get("series") or []),
                            "tail": (state.get("series") or [])[-8:],
                            "models": cards,
                        },
                        ensure_ascii=False,
                    ),
                    json_mode=True,
                )
                parsed = parse_json_object(raw)
                choice = str(parsed.get("model") or "")
                reason = str(parsed.get("reason") or "模型选择")
            except Exception as exc:
                reason = f"选模调用失败，改用默认模型（{exc}）"
        if choice not in allowed:
            choice, default_reason = _default_model(allowed, hint)
            reason = reason or default_reason
            if not choice:
                return {
                    "route": "report",
                    "catalog": cards,
                    "tool_trace": trace,
                    "errors": ["没有可调用的模型"],
                }
            if not reason:
                reason = default_reason
        return {
            "route": "forecast",
            "model_name": choice,
            "model_reason": reason,
            "model_card": by_id[choice],
            "catalog": cards,
            "tool_trace": trace,
        }

    async def forecast(state: TaskLedger) -> dict:
        model = state.get("model_name") or ""
        card = state.get("model_card") or {}
        try:
            args, trace = authorize_forecast(
                model,
                list(state.get("series") or []),
                int(state.get("horizon") or 0),
                state.get("freq") or "D",
                policy,
                int(card.get("context_length") or policy.context_tail),
            )
        except PolicyError as exc:
            return {"route": "report", "errors": [str(exc)]}
        if model in policy.require_approval_models:
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "tool": "forecast",
                    "model": model,
                    "horizon": args["horizon"],
                    "n_used": trace["n_used"],
                }
            )
            approved = isinstance(decision, dict) and decision.get("approved") is True
            if not approved:
                return {
                    "route": "report",
                    "tool_trace": [{**trace, "allowed": False, "reason": "denied"}],
                    "errors": ["预测调用未获批准"],
                }
        try:
            result = await call_tool("forecast", args)
        except Exception as exc:
            return {
                "route": "report",
                "prepared": trace,
                "tool_trace": [{**trace, "error": str(exc)}],
                "errors": [f"forecast 失败: {exc}"],
            }
        return {"route": "baseline", "forecast": result, "prepared": trace, "tool_trace": [trace]}

    async def baseline(state: TaskLedger) -> dict:
        cards = {item["id"]: item for item in state.get("catalog") or []}
        card = cards.get("seasonal_naive") or {"context_length": policy.context_tail}
        try:
            args, trace = authorize_forecast(
                "seasonal_naive",
                list(state.get("series") or []),
                int(state.get("horizon") or 0),
                state.get("freq") or "D",
                policy,
                int(card.get("context_length") or policy.context_tail),
            )
        except PolicyError as exc:
            return {"errors": [str(exc)]}
        try:
            result = await call_tool("forecast", args)
        except Exception as exc:
            return {"tool_trace": [{**trace, "error": str(exc)}], "errors": [f"基线失败: {exc}"]}
        return {"baseline": result, "tool_trace": [trace]}

    async def report(state: TaskLedger) -> dict:
        extra_errors: list[str] = []
        traces: list[dict] = []
        metrics = None
        baseline_metrics = None
        actual = state.get("actual")
        horizon = int(state.get("horizon") or 0)
        if actual:
            try:
                actual_values = finite_series(list(actual))
            except PolicyError as exc:
                actual_values = []
                extra_errors.append(str(exc))
            if actual_values and len(actual_values) != horizon:
                extra_errors.append("actual 长度与 horizon 不一致，跳过评分")
            elif actual_values and "evaluate" not in policy.allowed_tools:
                extra_errors.append("evaluate 不在工具白名单")
            elif actual_values:
                history = list(state.get("series") or [])
                season = season_of(state.get("freq") or "D")
                for key, target in (("forecast", "metrics"), ("baseline", "baseline_metrics")):
                    payload = state.get(key)
                    if not payload:
                        continue
                    tool_args = {
                        "point": payload["point"],
                        "actual": actual_values,
                        "history": history,
                        "q10": payload.get("q10"),
                        "q90": payload.get("q90"),
                        "season": season,
                    }
                    try:
                        scored = await call_tool("evaluate", tool_args)
                    except Exception as exc:
                        extra_errors.append(f"{key} 评分失败: {exc}")
                        continue
                    traces.append({"tool": "evaluate", "target": key, "allowed": True})
                    if target == "metrics":
                        metrics = scored
                    else:
                        baseline_metrics = scored
        text = _template_report(state, metrics, baseline_metrics, extra_errors)
        if llm.enabled and state.get("forecast"):
            try:
                text = llm.complete(
                    prompts["report"],
                    json.dumps(
                        {
                            "request": state.get("user_request"),
                            "model": state.get("model_name"),
                            "reason": state.get("model_reason"),
                            "forecast": state.get("forecast"),
                            "baseline": state.get("baseline"),
                            "metrics": metrics,
                            "baseline_metrics": baseline_metrics,
                            "errors": list(state.get("errors") or []) + extra_errors,
                        },
                        ensure_ascii=False,
                    ),
                ).strip() or text
            except Exception:
                pass
        update: dict = {"report": text, "tool_trace": traces}
        if metrics is not None:
            update["metrics"] = metrics
        if baseline_metrics is not None:
            update["baseline_metrics"] = baseline_metrics
        if extra_errors:
            update["errors"] = extra_errors
        return update

    return {
        "ingest": ingest,
        "select_model": select_model,
        "forecast": forecast,
        "baseline": baseline,
        "report": report,
    }


def _wants_baseline(request: str) -> bool:
    lowered = request.lower()
    return any(mark in request or mark in lowered for mark in _BASELINE_MARKS)


def _default_model(allowed: list[str], hint: str) -> tuple[str, str]:
    if hint in allowed:
        return hint, "使用请求指定的模型"
    if "chronos2" in allowed:
        return "chronos2", "使用默认模型 Chronos-2"
    if allowed:
        return allowed[0], f"使用可用模型 {allowed[0]}"
    return "", "没有可用模型"


def _fmt(values: list[float] | None, digits: int = 3) -> str:
    if not values:
        return "无"
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def _template_report(state: TaskLedger, metrics: dict | None, baseline_metrics: dict | None, extra_errors: list[str]) -> str:
    errors = list(state.get("errors") or []) + extra_errors
    forecast = state.get("forecast")
    baseline = state.get("baseline")
    if forecast is None and baseline is None:
        detail = "；".join(errors) if errors else "没有生成预测"
        return f"这次没有产出预测。{detail}"
    lines: list[str] = []
    if forecast:
        lines.append(
            f"{forecast.get('model')} 未来 {len(forecast.get('point') or [])} 步点预测为 {_fmt(forecast.get('point'))}。"
        )
        if forecast.get("q10") and forecast.get("q90"):
            lines.append(f"对应区间 q10–q90 为 {_fmt(forecast.get('q10'))} 到 {_fmt(forecast.get('q90'))}。")
        if state.get("model_reason"):
            lines.append(f"选模理由：{state['model_reason']}。")
    if baseline:
        lines.append(f"季节朴素基线为 {_fmt(baseline.get('point'))}。")
    if metrics:
        line = _metric_line("主模型", metrics)
        if line:
            lines.append(line)
    if baseline_metrics:
        line = _metric_line("基线", baseline_metrics)
        if line:
            lines.append(line)
    if errors:
        lines.append("注意：" + "；".join(errors))
    return "".join(lines)


def _metric_line(label: str, metrics: dict) -> str:
    """指标字典可能缺字段，缺哪个就少写哪个，不能因此把整个响应打挂。"""
    parts: list[str] = []
    for key, name in (("mae", "MAE"), ("mase", "MASE"), ("coverage", "coverage"), ("crps_proxy", "CRPS 近似")):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            parts.append(f"{name}={float(value):.4f}")
        except (TypeError, ValueError):
            continue
    if not parts:
        return ""
    return f"{label} " + "，".join(parts) + "。"
