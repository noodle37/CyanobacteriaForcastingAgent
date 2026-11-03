"""工具白名单、数值检查、上下文尾部截断。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Policy:
    allowed_tools: tuple[str, ...]
    allowed_models: tuple[str, ...]
    max_series_length: int = 2048
    context_tail: int = 512
    horizon_min: int = 1
    horizon_max: int = 64
    max_abs_value: float = 1e12
    require_approval_models: tuple[str, ...] = ()


def load_policy(path: Path) -> Policy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Policy(
        allowed_tools=tuple(raw.get("allowed_tools") or ()),
        allowed_models=tuple(raw.get("allowed_models") or ()),
        max_series_length=int(raw.get("max_series_length", 2048)),
        context_tail=int(raw.get("context_tail", 512)),
        horizon_min=int(raw.get("horizon_min", 1)),
        horizon_max=int(raw.get("horizon_max", 64)),
        max_abs_value=float(raw.get("max_abs_value", 1e12)),
        require_approval_models=tuple(raw.get("require_approval_models") or ()),
    )


class PolicyError(PermissionError):
    pass


def finite_series(values: list) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise PolicyError("序列必须是数值数组")
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyError("序列里出现了非数值")
        number = float(value)
        if not math.isfinite(number):
            raise PolicyError("序列包含非有限数值")
        parsed.append(number)
    return parsed


def check_request(series: list[float], horizon: int, policy: Policy) -> None:
    if len(series) < 2:
        raise PolicyError("序列至少需要 2 个有限数值")
    if len(series) > policy.max_series_length:
        raise PolicyError(f"序列长度 {len(series)} 超过上限 {policy.max_series_length}")
    if any(abs(value) > policy.max_abs_value for value in series):
        raise PolicyError("序列数值超出允许范围")
    if horizon < policy.horizon_min or horizon > policy.horizon_max:
        raise PolicyError(f"horizon 必须在 {policy.horizon_min} 到 {policy.horizon_max} 之间")


def compress_series(values: list[float], context_length: int, policy: Policy) -> tuple[list[float], dict]:
    limit = max(2, min(int(context_length), int(policy.context_tail), len(values)))
    used = values[-limit:]
    return used, {
        "n_raw": len(values),
        "n_used": len(used),
        "truncated": len(used) != len(values),
    }


def authorize_forecast(model: str, values: list[float], horizon: int, freq: str, policy: Policy, context_length: int) -> tuple[dict, dict]:
    if "forecast" not in policy.allowed_tools:
        raise PolicyError("forecast 不在工具白名单")
    if model not in policy.allowed_models:
        raise PolicyError(f"模型不在白名单: {model}")
    check_request(values, horizon, policy)
    used, meta = compress_series(values, context_length, policy)
    args = {"model": model, "values": used, "horizon": int(horizon), "freq": freq or "D"}
    trace = {"tool": "forecast", "model": model, "allowed": True, **meta, "horizon": int(horizon)}
    return args, trace
