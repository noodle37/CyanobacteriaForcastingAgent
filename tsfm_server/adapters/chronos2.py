"""Chronos-2 适配器。权重按需加载，调用方负责换入换出。"""

from __future__ import annotations

import math
import time

from tsfm_server.base import BaseTSFMModel

# 与 GIFT-Eval 官方 Chronos-2 notebook 相同的九档。点预测用 0.5 分位。
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Chronos2Adapter(BaseTSFMModel):
    def __init__(self, spec):
        super().__init__(spec)
        self._pipe = None

    def forecast(self, values: list[float], horizon: int, freq: str) -> dict:
        return self.forecast_batch([values], horizon, freq, cross_learning=False, batch_size=1)[0]

    def forecast_batch(
        self,
        series: list[list],
        horizon: int,
        freq: str,
        cross_learning: bool = False,
        batch_size: int = 100,
    ) -> list[dict]:
        """一批序列一起预测。cross_learning 打开时，同一批内部共享信息，和官方评测一致。"""
        started = time.perf_counter()
        import numpy as np

        arrays = [_as_array(values)[-int(self.spec.context_length) :] for values in series]
        pipe = self._load()
        quantiles, mean = pipe.predict_quantiles(
            arrays,
            prediction_length=horizon,
            quantile_levels=QUANTILE_LEVELS,
            batch_size=max(int(batch_size), 1),
            cross_learning=bool(cross_learning),
        )
        if len(quantiles) != len(arrays) or len(mean) != len(arrays):
            raise RuntimeError(f"Chronos-2 返回条数异常: {len(quantiles)} / {len(arrays)}")
        elapsed = (time.perf_counter() - started) * 1000
        packed = []
        for quantile_tensor, mean_tensor, series_values in zip(quantiles, mean, arrays):
            grid = _to_2d(quantile_tensor)
            point = _to_1d(mean_tensor)
            if grid.shape[0] != horizon or grid.shape[1] != len(QUANTILE_LEVELS) or len(point) < horizon:
                raise RuntimeError(f"Chronos-2 分位数形状异常: {tuple(grid.shape)}")
            levels = {
                _level_key(level): grid[:horizon, index].tolist()
                for index, level in enumerate(QUANTILE_LEVELS)
            }
            item = self.pack(point[:horizon], levels["0.1"], levels["0.9"], elapsed, freq)
            item["quantiles"] = levels
            item["context_used"] = int(series_values.shape[0])
            packed.append(item)
        return packed

    def unload(self) -> None:
        self._pipe = None
        _release_cuda()

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from chronos import Chronos2Pipeline

        device = self.spec.device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        device_map = "cpu" if device == "cpu" else "cuda"
        self._pipe = Chronos2Pipeline.from_pretrained(self.spec.checkpoint, device_map=device_map)
        return self._pipe


def _level_key(level: float) -> str:
    return f"{level:g}"


def _as_array(values: list):
    import numpy as np

    parsed: list[float] = []
    for value in values:
        if value is None:
            parsed.append(np.nan)
            continue
        number = float(value)
        if math.isnan(number):
            parsed.append(np.nan)
            continue
        if math.isinf(number):
            raise ValueError("序列包含无穷值")
        parsed.append(number)
    if len(parsed) < 2:
        raise ValueError("序列至少需要 2 个点")
    return np.asarray(parsed, dtype=np.float32)


def _to_1d(tensor) -> list[float]:
    import numpy as np

    array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array = np.asarray(array, dtype="float64")
    if array.ndim == 2:
        array = array[0]
    return array.reshape(-1).tolist()


def _to_2d(tensor):
    import numpy as np

    array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array = np.asarray(array, dtype="float64")
    if array.ndim == 3:
        array = array[0]
    return array


def _release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
