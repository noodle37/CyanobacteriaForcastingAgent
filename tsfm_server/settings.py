"""TSFM 侧配置。路径可用环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

os.environ.setdefault(
    "CHRONOS2_CHECKPOINT",
    "/home/jxy/.cache/huggingface/hub/models--autogluon--chronos-2/snapshots/60088152a34e242427b44c3100014473a0157d53",
)
os.environ.setdefault(
    "TIMESFM3_CHECKPOINT",
    "/home/jxy/chronos-cyano/models/timesfm-3.0-pytorch",
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HOST = os.getenv("TSFM_HOST", "127.0.0.1")
PORT = int(os.getenv("TSFM_PORT", "8765"))
MODELS_PATH = Path(os.getenv("TSFM_MODELS_CONFIG", ROOT / "configs" / "models.yaml"))
