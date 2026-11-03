"""统一预测接口。各模型适配器只实现这一层。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tsfm_server.registry import ModelSpec


def as_floats(values: Any) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError("序列包含非有限数值")
        out.append(number)
    return out


class BaseTSFMModel(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def forecast(self, values: list[float], horizon: int, freq: str) -> dict:
        """返回 point / q10 / q90 / model / elapsed_ms。"""

    def unload(self) -> None:
        return None

    def pack(
        self,
        point: list[float],
        q10: list[float] | None,
        q90: list[float] | None,
        elapsed_ms: float,
        freq: str,
    ) -> dict:
        if len(point) != int(self_horizon(point, q10, q90)):
            raise RuntimeError("预测长度不一致")
        return {
            "model": self.spec.id,
            "point": [float(x) for x in point],
            "q10": None if q10 is None else [float(x) for x in q10],
            "q90": None if q90 is None else [float(x) for x in q90],
            "elapsed_ms": float(elapsed_ms),
            "freq": freq,
            "context_used": None,
        }


def self_horizon(point: list[float], q10: list[float] | None, q90: list[float] | None) -> int:
    horizon = len(point)
    if q10 is not None and len(q10) != horizon:
        raise RuntimeError("q10 长度与点预测不一致")
    if q90 is not None and len(q90) != horizon:
        raise RuntimeError("q90 长度与点预测不一致")
    return horizon
