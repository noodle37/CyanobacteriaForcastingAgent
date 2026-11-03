"""把磁盘上的适配器重新导入当前进程。"""

from __future__ import annotations

import importlib
from pathlib import Path

from tsfm_server.runtime import ModelRuntime


def load_fresh_runtime(models_path: Path) -> ModelRuntime:
    import tsfm_server.adapters as adapters
    import tsfm_server.adapters.chronos2 as chronos2
    import tsfm_server.adapters.seasonal_naive as seasonal_naive
    import tsfm_server.adapters.timesfm3 as timesfm3
    import tsfm_server.base as base
    import tsfm_server.registry as registry
    import tsfm_server.runtime as runtime_mod

    importlib.reload(base)
    importlib.reload(registry)
    importlib.reload(chronos2)
    importlib.reload(timesfm3)
    importlib.reload(seasonal_naive)
    importlib.reload(adapters)
    importlib.reload(runtime_mod)
    return runtime_mod.ModelRuntime(registry.load_registry(models_path))
