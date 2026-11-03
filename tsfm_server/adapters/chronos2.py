"""Chronos-2 适配器。权重按需加载，调用方负责换入换出。"""

from __future__ import annotations

import time

from tsfm_server.base import BaseTSFMModel, as_floats


class Chronos2Adapter(BaseTSFMModel):
    def __init__(self, spec):
        super().__init__(spec)
        self._pipe = None

    def forecast(self, values: list[float], horizon: int, freq: str) -> dict:
        started = time.perf_counter()
        import numpy as np

        series = as_floats(values)
        series = series[-self.spec.context_length :]
        pipe = self._load()
        quantiles, mean = pipe.predict_quantiles(
            [np.asarray(series, dtype=np.float32)],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        point_tensor = mean[0]
        quantile_tensor = quantiles[0]
        point = _to_1d(point_tensor)
        grid = _to_2d(quantile_tensor)
        if grid.shape[0] != horizon or grid.shape[1] < 3:
            raise RuntimeError(f"Chronos-2 分位数形状异常: {tuple(grid.shape)}")
        elapsed = (time.perf_counter() - started) * 1000
        packed = self.pack(point[:horizon], grid[:, 0].tolist(), grid[:, 2].tolist(), elapsed, freq)
        packed["context_used"] = len(series)
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
