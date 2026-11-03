"""OpenAI 兼容聊天接口。三项环境变量缺任一则视为未启用。"""

from __future__ import annotations

import json


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float):
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.enabled:
            raise RuntimeError("LLM 未配置")
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"model": self.model, "temperature": 0, "messages": messages}
        if json_mode:
            try:
                response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            except Exception:
                response = client.chat.completions.create(**kwargs)
        else:
            response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""

    def as_chat_model(self):
        """给 ReAct 用的 LangChain chat model。未配置时不要调用。"""
        from agent.chat_model import OpenAICompatibleChat

        if not self.enabled:
            raise RuntimeError("LLM 未配置")
        return OpenAICompatibleChat(
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.model,
            timeout=self.timeout,
        )


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型没有返回 JSON 对象")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("模型返回的 JSON 不是对象")
    return data
