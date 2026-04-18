"""OpenAI-compatible chat client. One class, swap base_url per backend."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    tokens_per_sec: float
    raw: dict[str, Any]


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, str]], **opts: Any) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        body.update(opts)

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as c:
            r = c.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        latency = time.perf_counter() - t0

        msg = data["choices"][0]["message"]
        # Reasoning-model servers (Qwen3, DeepSeek-R1, etc. under llama.cpp)
        # put chain-of-thought in `reasoning_content` and the final answer in
        # `content`. Prefer `content`; fall back to reasoning if empty so the
        # benchmark never compares an empty string against a real response.
        choice = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tps = (ct / latency) if latency > 0 and ct > 0 else 0.0

        return ChatResult(
            text=choice,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=latency,
            tokens_per_sec=tps,
            raw=data,
        )
