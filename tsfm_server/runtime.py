"""一次只把一个 GPU 模型留在显存里。CPU 基线不挤占它。"""

from __future__ import annotations

import threading

from tsfm_server.adapters import build_adapter
from tsfm_server.base import BaseTSFMModel
from tsfm_server.registry import ModelSpec


class ModelRuntime:
    def __init__(self, specs: dict[str, ModelSpec]):
        self.specs = specs
        self._lock = threading.Lock()
        self._gpu_name: str | None = None
        self._gpu_model: BaseTSFMModel | None = None
        self._cpu_models: dict[str, BaseTSFMModel] = {}

    def list_cards(self) -> list[dict]:
        return [spec.card() for spec in self.specs.values()]

    def forecast(self, model: str, values: list[float], horizon: int, freq: str) -> dict:
        spec = self.specs.get(model)
        if spec is None or not spec.enabled:
            known = ", ".join(sorted(self.specs))
            raise ValueError(f"模型不可用: {model}。已注册: {known}")
        if horizon < 1:
            raise ValueError("horizon 必须为正")
        with self._lock:
            adapter = self._acquire(spec)
            return adapter.forecast(values, horizon, freq)

    def _acquire(self, spec: ModelSpec) -> BaseTSFMModel:
        if str(spec.device).startswith("cuda"):
            if self._gpu_name != spec.id:
                if self._gpu_model is not None:
                    self._gpu_model.unload()
                self._gpu_model = build_adapter(spec)
                self._gpu_name = spec.id
            return self._gpu_model
        cached = self._cpu_models.get(spec.id)
        if cached is None:
            cached = build_adapter(spec)
            self._cpu_models[spec.id] = cached
        return cached
