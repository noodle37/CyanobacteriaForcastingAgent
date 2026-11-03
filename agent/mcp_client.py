"""通过 langchain-mcp-adapters 发现并调用 TSFM MCP 工具。"""

from __future__ import annotations

import json
from typing import Any


class MCPToolClient:
    def __init__(self, url: str):
        self.url = url
        self._client = None

    def _connection(self):
        if self._client is None:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(
                {"tsfm": {"transport": "streamable_http", "url": self.url}},
                handle_tool_errors=False,
            )
        return self._client

    async def call(self, name: str, args: dict) -> dict:
        tools = await self._connection().get_tools(server_name="tsfm")
        tool = next((item for item in tools if item.name == name), None)
        if tool is None:
            known = ", ".join(item.name for item in tools)
            raise RuntimeError(f"MCP 工具不存在: {name}。已发现: {known}")
        raw = await tool.ainvoke(args)
        data = _as_dict(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"MCP 工具 {name} 没有返回 JSON 对象")
        return data


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, tuple) and raw:
        artifact = raw[1] if len(raw) > 1 else None
        structured = getattr(artifact, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        raw = raw[0]
    if isinstance(raw, dict):
        structured = raw.get("structured_content")
        if isinstance(structured, dict):
            return structured
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        texts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return json.loads("".join(texts))
    raise TypeError(f"无法解析 MCP 返回值: {type(raw).__name__}")
