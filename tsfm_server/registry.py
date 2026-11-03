"""模型注册表。YAML 里的 ${VAR} 在加载时展开。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    adapter: str
    enabled: bool
    device: str
    checkpoint: str
    context_length: int
    quantiles: bool
    task: str
    description: str
    season: int = 7
    make_positive: bool = False
    batch_size: int = 1

    def card(self) -> dict:
        return {
            "id": self.id,
            "enabled": self.enabled,
            "device": self.device,
            "context_length": self.context_length,
            "quantiles": self.quantiles,
            "task": self.task,
            "description": self.description,
        }


def _expand(value: object) -> str:
    return os.path.expandvars(str(value or ""))


def load_registry(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, ModelSpec] = {}
    for model_id, item in (raw.get("models") or {}).items():
        item = item or {}
        specs[model_id] = ModelSpec(
            id=model_id,
            adapter=str(item.get("adapter", model_id)),
            enabled=bool(item.get("enabled", True)),
            device=str(item.get("device", "cpu")),
            checkpoint=_expand(item.get("checkpoint", "")),
            context_length=int(item.get("context_length", 512)),
            quantiles=bool(item.get("quantiles", False)),
            task=str(item.get("task", "forecast")),
            description=str(item.get("description", "")),
            season=int(item.get("season", 7)),
            make_positive=bool(item.get("make_positive", False)),
            batch_size=int(item.get("batch_size", 1)),
        )
    return specs
