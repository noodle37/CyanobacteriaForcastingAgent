"""TimesFM-3 适配器。沿用本机已验证的 TimesFM3Evaluator 调用方式。"""

from __future__ import annotations

import time

from tsfm_server.base import BaseTSFMModel, as_floats


class TimesFM3Adapter(BaseTSFMModel):
    def __init__(self, spec):
        super().__init__(spec)
        self._pipeline = None

    def forecast(self, values: list[float], horizon: int, freq: str) -> dict:
        started = time.perf_counter()
        series = as_floats(values)
        series = series[-self.spec.context_length :]
        pipeline = self._load()
        import numpy as np

        outputs = list(
            pipeline.predict_batch(
                contexts=[np.asarray(series, dtype=np.float32)],
                horizon=horizon,
                return_quantiles=True,
                use_symmetric_averaging=True,
                make_positive=bool(self.spec.make_positive),
                univariate=True,
            )
        )
        if not outputs:
            raise RuntimeError("TimesFM-3 没有返回预测")
        out = outputs[0]
        forecast = np.asarray(out.forecast, dtype=np.float64)
        quantiles = np.asarray(out.quantiles, dtype=np.float64)
        if forecast.ndim == 2:
            forecast = forecast[0]
        if quantiles.ndim == 3:
            quantiles = quantiles[0]
        forecast = forecast.reshape(-1)[:horizon]
        if quantiles.ndim != 2 or quantiles.shape[1] < 9:
            raise RuntimeError(f"TimesFM-3 分位数形状异常: {quantiles.shape}")
        elapsed = (time.perf_counter() - started) * 1000
        packed = self.pack(
            forecast.tolist(),
            quantiles[:horizon, 0].tolist(),
            quantiles[:horizon, 8].tolist(),
            elapsed,
            freq,
        )
        packed["context_used"] = len(series)
        return packed

    def unload(self) -> None:
        self._pipeline = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        device = self.spec.device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        config = ModelConfig(
            checkpoint_path=self.spec.checkpoint,
            per_core_batch_size=max(int(self.spec.batch_size), 1),
            device=str(device),
            local_files_only=True,
        )
        self._pipeline = TimesFM3Evaluator(config)
        return self._pipeline
