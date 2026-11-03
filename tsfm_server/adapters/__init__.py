from tsfm_server.adapters.chronos2 import Chronos2Adapter
from tsfm_server.adapters.seasonal_naive import SeasonalNaiveAdapter
from tsfm_server.adapters.timesfm3 import TimesFM3Adapter
from tsfm_server.registry import ModelSpec

_BUILDERS = {
    "chronos2": Chronos2Adapter,
    "timesfm3": TimesFM3Adapter,
    "seasonal_naive": SeasonalNaiveAdapter,
}


def build_adapter(spec: ModelSpec):
    try:
        cls = _BUILDERS[spec.adapter]
    except KeyError as exc:
        raise KeyError(f"未知适配器: {spec.adapter}") from exc
    return cls(spec)
