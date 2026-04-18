"""OpenAI-compatible chat client. One class, swap base_url per backend.

Streams by default so time-to-first-token (TTFT) can be measured separately
from total latency. TTFT is the wall time between request-send and the first
non-empty delta (either `content` or `reasoning_content`).
"""
from __future__ import annotations

import json
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
    ttft_s: float | None
    tokens_per_sec: float
    raw: dict[str, Any]


def _parse_sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse one SSE `data:` line. Returns None for keep-alives / [DONE]."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_delta(chunk: dict[str, Any]) -> tuple[str, str]:
    """Return (content_delta, reasoning_delta) from a streamed chunk."""
    choices = chunk.get("choices") or []
    if not choices:
        return "", ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or "", delta.get("reasoning_content") or ""


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "none",
                 timeout_s: float = 120.0, stream: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.stream = stream

    def chat(self, messages: list[dict[str, str]], **opts: Any) -> ChatResult:
        if self.stream:
            return self._chat_stream(messages, **opts)
        return self._chat_blocking(messages, **opts)

    def _chat_blocking(self, messages: list[dict[str, str]], **opts: Any) -> ChatResult:
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
            ttft_s=None,
            tokens_per_sec=tps,
            raw=data,
        )

    def _chat_stream(self, messages: list[dict[str, str]], **opts: Any) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body.update(opts)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        last_chunk: dict[str, Any] = {}
        ttft: float | None = None

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as c:
            with c.stream("POST", url, json=body, headers=headers) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = _parse_sse_chunk(line)
                    if chunk is None:
                        continue
                    last_chunk = chunk
                    u = chunk.get("usage")
                    if u:
                        usage = u
                    c_delta, r_delta = _extract_delta(chunk)
                    if (c_delta or r_delta) and ttft is None:
                        ttft = time.perf_counter() - t0
                    if c_delta:
                        content_parts.append(c_delta)
                    if r_delta:
                        reasoning_parts.append(r_delta)
        latency = time.perf_counter() - t0

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        # Prefer content; fall back to reasoning so benchmark never scores "".
        text = content or reasoning
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tps = (ct / latency) if latency > 0 and ct > 0 else 0.0

        return ChatResult(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=latency,
            ttft_s=ttft,
            tokens_per_sec=tps,
            raw=last_chunk,
        )
