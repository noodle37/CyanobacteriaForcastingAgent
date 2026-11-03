"""读仓库根目录的 .env。不提供方舟 Agent Plan 默认地址。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from agent.hooks import Policy, load_policy
from agent.llm import LLMClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


LLM = LLMClient(
    base_url=_text("LLM_BASE_URL"),
    api_key=_text("LLM_API_KEY"),
    model=_text("LLM_MODEL"),
    timeout=float(os.getenv("LLM_TIMEOUT_S", "60")),
)
POLICY = load_policy(Path(os.getenv("AGENT_POLICY", ROOT / "configs" / "permissions.yaml")))
TSFM_MCP_URL = _text("TSFM_MCP_URL", "http://127.0.0.1:8765/mcp")
AGENT_HOST = _text("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8000"))
PROMPTS = ROOT / "prompts"


def read_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


__all__ = ["LLM", "POLICY", "ROOT", "TSFM_MCP_URL", "AGENT_HOST", "AGENT_PORT", "Policy", "read_prompt"]
