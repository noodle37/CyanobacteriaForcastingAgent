"""两边进程共用的评分函数。只依赖标准库。"""

from __future__ import annotations


def _pairs(point: list[float], actual: list[float]) -> list[tuple[float, float]]:
    if len(point) != len(actual) or not point:
        raise ValueError("point 与 actual 长度必须相同且非空")
    return list(zip(point, actual))


def mae(point: list[float], actual: list[float]) -> float:
    return sum(abs(yhat - y) for yhat, y in _pairs(point, actual)) / len(point)


def _pinball(y: float, yhat: float, tau: float) -> float:
    diff = y - yhat
    if diff >= 0:
        return tau * diff
    return (tau - 1.0) * diff


def mase(
    point: list[float],
    actual: list[float],
    history: list[float],
    season: int = 1,
) -> float | None:
    """MASE。分母是历史序列季节差分的平均绝对误差。历史太短时返回 None。"""
    season = max(int(season), 1)
    if len(history) <= season:
        return None
    scale_terms = [abs(history[i] - history[i - season]) for i in range(season, len(history))]
    scale = sum(scale_terms) / len(scale_terms)
    if scale == 0:
        return None
    return mae(point, actual) / scale


def coverage(
    actual: list[float],
    q10: list[float] | None,
    q90: list[float] | None,
) -> float | None:
    if not q10 or not q90 or len(q10) != len(actual) or len(q90) != len(actual):
        return None
    inside = 0
    for y, lo, hi in zip(actual, q10, q90):
        low, high = (lo, hi) if lo <= hi else (hi, lo)
        if low <= y <= high:
            inside += 1
    return inside / len(actual)


def crps_proxy(
    actual: list[float],
    point: list[float],
    q10: list[float] | None,
    q90: list[float] | None,
) -> float:
    """用 0.1 / 0.5 / 0.9 的平均 pinball 作为 CRPS 近似。没有区间时只用中位数。"""
    total = 0.0
    count = 0
    for i, (yhat, y) in enumerate(_pairs(point, actual)):
        total += _pinball(y, yhat, 0.5)
        count += 1
        if q10 and q90 and len(q10) == len(actual) and len(q90) == len(actual):
            total += _pinball(y, q10[i], 0.1)
            total += _pinball(y, q90[i], 0.9)
            count += 2
    return total / count


def evaluate_forecast(
    point: list[float],
    actual: list[float],
    history: list[float],
    q10: list[float] | None = None,
    q90: list[float] | None = None,
    season: int = 1,
) -> dict:
    return {
        "mae": mae(point, actual),
        "mase": mase(point, actual, history, season),
        "coverage": coverage(actual, q10, q90),
        "crps_proxy": crps_proxy(actual, point, q10, q90),
        "season": int(season),
        "horizon": len(point),
    }
