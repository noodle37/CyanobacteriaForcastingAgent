"""季节朴素基线。周期重复最后一个季节，区间用季节残差的经验分位。"""

from __future__ import annotations

import time

from tsfm_server.base import BaseTSFMModel, as_floats


class SeasonalNaiveAdapter(BaseTSFMModel):
    def forecast(self, values: list[float], horizon: int, freq: str) -> dict:
        started = time.perf_counter()
        series = as_floats(values)
        if len(series) < 2:
            raise ValueError("序列至少需要 2 个点")
        season = max(int(self.spec.season), 1)
        if len(series) >= season:
            pattern = series[-season:]
        else:
            pattern = [series[-1]]
        point = [pattern[i % len(pattern)] for i in range(horizon)]
        q10, q90 = _residual_band(series, pattern, point, season)
        elapsed = (time.perf_counter() - started) * 1000
        packed = self.pack(point, q10, q90, elapsed, freq)
        packed["context_used"] = len(series)
        return packed


def _residual_band(
    series: list[float],
    pattern: list[float],
    point: list[float],
    season: int,
) -> tuple[list[float] | None, list[float] | None]:
    if len(series) <= season:
        width = _std(series) or (abs(series[-1]) * 0.1 + 1e-6)
        return [p - width for p in point], [p + width for p in point]
    residuals: list[float] = []
    for idx in range(season, len(series)):
        residuals.append(series[idx] - series[idx - season])
    residuals.sort()
    lo = _quantile(residuals, 0.1)
    hi = _quantile(residuals, 0.9)
    return [p + lo for p in point], [p + hi for p in point]


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    weight = pos - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight
