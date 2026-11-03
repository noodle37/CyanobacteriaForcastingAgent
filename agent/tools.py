"""ReAct 工具。TSFM 走 MCP，python 在本进程外的子进程里改数据和源码。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from agent.hooks import Policy, PolicyError, authorize_forecast, check_request, finite_series
from agent.python_exec import execute_python
from agent.settings import ROOT
from agent.workers import ToolClient, season_of


def make_tools(call_tool: ToolClient, policy: Policy, root: Path | None = None):
    root = Path(root) if root is not None else ROOT

    @tool
    async def list_models(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """列出已注册模型的 id、上下文长度和是否启用。不会加载权重。"""
        if "list_models" not in policy.allowed_tools:
            return _finish(tool_call_id, {"ok": False, "error": "list_models 不在工具白名单"}, {"tool": "list_models", "allowed": False})
        try:
            listed = await call_tool("list_models", {})
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return _finish(
                tool_call_id,
                {"ok": False, "error": str(exc)},
                {"tool": "list_models", "allowed": True, "error": str(exc)},
                errors=[f"list_models 失败: {exc}"],
            )
        cards = list(listed.get("models") or [])
        return _finish(
            tool_call_id,
            {"ok": True, "models": cards},
            {"tool": "list_models", "allowed": True, "models": [item.get("id") for item in cards]},
            catalog=cards,
        )

    @tool
    async def forecast(
        model: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        horizon: int | None = None,
        freq: str | None = None,
        values: list[float] | None = None,
    ) -> Command:
        """用白名单中的模型预测未来 horizon 步。省略 values 时使用当前序列，包括 python 写回的序列。基线模型 id 是 seasonal_naive。返回 point、q10、q90。"""
        if "forecast" not in policy.allowed_tools:
            return _finish(tool_call_id, {"ok": False, "error": "forecast 不在工具白名单"}, {"tool": "forecast", "allowed": False})
        model_id = (model or "").strip()
        chosen_horizon = int(state.get("horizon") or 0) if horizon is None else int(horizon)
        chosen_freq = freq or state.get("freq") or "D"
        chosen_values = list(state.get("series") or []) if values is None else list(values)
        cards = {item.get("id"): item for item in state.get("catalog") or []}
        card = cards.get(model_id) or {}
        try:
            args, trace = authorize_forecast(
                model_id,
                chosen_values,
                chosen_horizon,
                chosen_freq,
                policy,
                int(card.get("context_length") or policy.context_tail),
            )
        except (PolicyError, TypeError, ValueError) as exc:
            return _finish(tool_call_id, {"ok": False, "error": str(exc)}, {"tool": "forecast", "model": model_id, "allowed": False, "error": str(exc)})
        if model_id in policy.require_approval_models:
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "tool": "forecast",
                    "model": model_id,
                    "horizon": args["horizon"],
                    "n_used": trace["n_used"],
                }
            )
            approved = isinstance(decision, dict) and decision.get("approved") is True
            if not approved:
                return _finish(
                    tool_call_id,
                    {"ok": False, "error": "预测调用未获批准"},
                    {**trace, "allowed": False, "reason": "denied"},
                    errors=["预测调用未获批准"],
                )
        try:
            result = await call_tool("forecast", args)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return _finish(
                tool_call_id,
                {"ok": False, "error": str(exc)},
                {**trace, "error": str(exc)},
                errors=[f"forecast 失败: {exc}"],
            )
        update = {"prepared": trace} if model_id != "seasonal_naive" else {}
        if model_id == "seasonal_naive":
            update["baseline"] = result
        else:
            update["forecast"] = result
            update["model_name"] = model_id
            update["model_reason"] = "ReAct 调用"
        return _finish(tool_call_id, {"ok": True, "forecast": result}, trace, **update)

    @tool
    async def evaluate(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        target: str = "forecast",
    ) -> Command:
        """用 actual 给已有预测打分。target 取 forecast 或 baseline。返回 MAE、MASE、coverage、crps_proxy。"""
        if "evaluate" not in policy.allowed_tools:
            return _finish(tool_call_id, {"ok": False, "error": "evaluate 不在工具白名单"}, {"tool": "evaluate", "allowed": False})
        which = target if target in ("forecast", "baseline") else "forecast"
        payload = state.get("forecast" if which == "forecast" else "baseline")
        actual = state.get("actual")
        if not payload:
            return _finish(tool_call_id, {"ok": False, "error": f"还没有 {which} 预测"}, {"tool": "evaluate", "target": which, "allowed": False})
        if not actual:
            return _finish(tool_call_id, {"ok": False, "error": "任务没有 actual，不能评分"}, {"tool": "evaluate", "target": which, "allowed": False})
        try:
            actual_values = finite_series(list(actual))
            if len(actual_values) != len(payload.get("point") or []):
                raise PolicyError("actual 长度与预测不一致")
            scored = await call_tool(
                "evaluate",
                {
                    "point": payload["point"],
                    "actual": actual_values,
                    "history": list(state.get("series") or []),
                    "q10": payload.get("q10"),
                    "q90": payload.get("q90"),
                    "season": season_of(state.get("freq") or "D"),
                },
            )
        except GraphBubbleUp:
            raise
        except (PolicyError, TypeError, ValueError, KeyError) as exc:
            return _finish(tool_call_id, {"ok": False, "error": str(exc)}, {"tool": "evaluate", "target": which, "allowed": False, "error": str(exc)})
        except Exception as exc:
            return _finish(
                tool_call_id,
                {"ok": False, "error": str(exc)},
                {"tool": "evaluate", "target": which, "allowed": True, "error": str(exc)},
                errors=[f"{which} 评分失败: {exc}"],
            )
        field = "metrics" if which == "forecast" else "baseline_metrics"
        return _finish(
            tool_call_id,
            {"ok": True, "target": which, "metrics": scored},
            {"tool": "evaluate", "target": which, "allowed": True},
            **{field: scored},
        )

    @tool
    async def reload_models(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """重新加载 TSFM 进程里的适配器并卸下已缓存权重。python 写过源码后会自动调用，也可以再调一次。"""
        if "reload_models" not in policy.allowed_tools:
            return _finish(tool_call_id, {"ok": False, "error": "reload_models 不在工具白名单"}, {"tool": "reload_models", "allowed": False})
        try:
            result = await call_tool("reload_models", {})
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return _finish(
                tool_call_id,
                {"ok": False, "error": str(exc)},
                {"tool": "reload_models", "allowed": True, "error": str(exc)},
                errors=[f"reload_models 失败: {exc}"],
            )
        ok = bool(result.get("reloaded"))
        extra = {} if ok else {"errors": [f"reload_models 失败: {result.get('error')}"]}
        return _finish(tool_call_id, result, {"tool": "reload_models", "allowed": True, "reloaded": ok}, **extra)

    @tool
    async def python(code: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """执行 Python，用于分析、改写序列或修改 TSFM 源码。

        可用变量：series、horizon、freq、actual、forecast、baseline、metrics、result。
        原地修改 series 或重新赋值，增强后的序列会成为后续 forecast 的默认输入。
        read_file(path) 可读 tsfm_server、shared、prompts、configs。
        write_file(path, content) 只能写 tsfm_server 和 shared 下的源码，写完会自动 reload_models。
        同一轮里的其他工具看不到这次结果。改完序列或代码后，下一轮再 forecast。
        """
        if "python" not in policy.allowed_tools:
            return _finish(tool_call_id, {"ok": False, "error": "python 不在工具白名单"}, {"tool": "python", "allowed": False})
        result = await asyncio.to_thread(
            execute_python,
            code,
            root=root,
            series=list(state.get("series") or []),
            horizon=int(state.get("horizon") or 0),
            freq=state.get("freq") or "D",
            actual=state.get("actual"),
            forecast=state.get("forecast"),
            baseline=state.get("baseline"),
            metrics=state.get("metrics"),
            read_roots=policy.python_read_roots,
            write_roots=policy.python_write_roots,
            timeout_s=float(policy.python_timeout_s),
        )
        update: dict = {}
        problems: list[str] = []
        if result.get("ok") and result.get("series_changed"):
            try:
                new_series = finite_series(list(result.get("series") or []))
                new_horizon = int(result.get("horizon") or state.get("horizon") or 0)
                check_request(new_series, new_horizon, policy)
            except (PolicyError, TypeError, ValueError) as exc:
                problems.append(f"序列没有写回: {exc}")
            else:
                update["series"] = new_series
                update["horizon"] = new_horizon
        elif result.get("ok"):
            try:
                new_horizon = int(result.get("horizon"))
            except (TypeError, ValueError):
                new_horizon = int(state.get("horizon") or 0)
            if new_horizon != int(state.get("horizon") or 0):
                try:
                    check_request(list(state.get("series") or []), new_horizon, policy)
                except (PolicyError, TypeError, ValueError) as exc:
                    problems.append(f"horizon 没有写回: {exc}")
                else:
                    update["horizon"] = new_horizon
        if result.get("ok"):
            new_freq = str(result.get("freq") or "")
            if new_freq and new_freq != (state.get("freq") or "D"):
                update["freq"] = new_freq
        files = list(result.get("files_written") or [])
        reloaded = None
        if files and "reload_models" in policy.allowed_tools:
            try:
                reloaded = await call_tool("reload_models", {})
            except GraphBubbleUp:
                raise
            except Exception as exc:
                reloaded = {"reloaded": False, "error": str(exc)}
            if not reloaded.get("reloaded"):
                problems.append(f"reload_models 失败: {reloaded.get('error')}")
        elif files:
            problems.append("写了文件，但 reload_models 不在工具白名单")
        if not result.get("ok") and result.get("error"):
            problems.insert(0, str(result["error"]))
        series_now = update.get("series") or list(state.get("series") or [])
        view = {
            "ok": not problems,
            "result": result.get("result"),
            "stdout": result.get("stdout") or "",
            "series_changed": "series" in update,
            "n": len(series_now),
            "tail": series_now[-8:],
            "horizon": update.get("horizon", state.get("horizon")),
            "freq": update.get("freq", state.get("freq")),
            "files_written": files,
        }
        if problems:
            view["error"] = "；".join(problems)
        if reloaded is not None:
            view["reload"] = reloaded
        traces = [
            {
                "tool": "python",
                "allowed": True,
                "series_changed": "series" in update,
                "files_written": files,
            }
        ]
        if reloaded is not None:
            traces.append({"tool": "reload_models", "allowed": True, "reloaded": bool(reloaded.get("reloaded"))})
        if problems:
            update["errors"] = problems
        return _finish(tool_call_id, view, None, tool_trace=traces, **update)

    return [list_models, forecast, evaluate, reload_models, python]


def _finish(tool_call_id: str, payload: dict, trace: dict | None, **updates) -> Command:
    body = json.dumps(payload, ensure_ascii=False, default=str)
    if len(body) > 6000:
        body = body[:6000] + "…(截断)"
    update = {"messages": [ToolMessage(content=body, tool_call_id=tool_call_id)], **updates}
    if trace is not None:
        update["tool_trace"] = [trace]
    return Command(update=update)
