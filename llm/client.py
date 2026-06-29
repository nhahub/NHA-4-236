"""Ollama LLM client.

Talks to a local Ollama server (default http://localhost:11434) over its REST
API, so no paid API key is required. The interface is intentionally small —
``chat`` for grounded generation and ``generate`` for single-prompt calls — and
synchronous; the FastAPI layer offloads these blocking calls to a threadpool.
"""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import httpx

from config import PROMPTS_DIR, settings

# Reasoning models (e.g. qwen3) wrap chain-of-thought in <think>...</think>;
# strip it so only the final answer reaches the user.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _tail_prefix_len(segment: str, tag: str) -> int:
    """Length of the longest suffix of ``segment`` that is a partial prefix of
    ``tag`` — i.e. text we must hold back in case a tag straddles chunks."""
    for k in range(min(len(segment), len(tag) - 1), 0, -1):
        if tag.startswith(segment[-k:]):
            return k
    return 0


def _consume_think(buf: str, in_think: bool) -> tuple[str, str, bool]:
    """Strip ``<think>`` blocks from a streaming buffer.

    Returns ``(emit, remaining, in_think)``: ``emit`` is safe-to-show text,
    ``remaining`` is a held-back partial tag to prepend to the next chunk, and
    ``in_think`` tracks whether we are currently inside a think block.
    """
    out: list[str] = []
    i = 0
    while i < len(buf):
        if in_think:
            idx = buf.find(_THINK_CLOSE, i)
            if idx == -1:
                keep = _tail_prefix_len(buf[i:], _THINK_CLOSE)
                return "".join(out), buf[len(buf) - keep:] if keep else "", True
            i = idx + len(_THINK_CLOSE)
            in_think = False
        else:
            idx = buf.find(_THINK_OPEN, i)
            if idx == -1:
                keep = _tail_prefix_len(buf[i:], _THINK_OPEN)
                out.append(buf[i: len(buf) - keep] if keep else buf[i:])
                return "".join(out), buf[len(buf) - keep:] if keep else "", False
            out.append(buf[i:idx])
            i = idx + len(_THINK_OPEN)
            in_think = True
    return "".join(out), "", in_think


def load_prompt(name: str) -> str:
    """Load a prompt template from ``llm/prompts`` by filename stem.

    Reads from disk on every call — no caching — so prompt file edits take
    effect immediately without restarting the server.
    """
    path = Path(PROMPTS_DIR) / f"{name}.txt"
    return path.read_text(encoding="utf-8")


class OllamaClient:
    """Minimal synchronous client for the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    def _keep_alive(self):
        """Ollama wants a number (seconds; -1 = forever) or a duration string
        like "5m". A bare "-1" string is an invalid duration -> 400, so coerce
        numeric values to int and pass duration strings through unchanged."""
        ka = settings.ollama_keep_alive
        try:
            return int(ka)
        except (TypeError, ValueError):
            return ka

    def _options(self, temperature: float, extra: dict) -> dict:
        opts: dict = {"temperature": temperature, **extra}
        # Force CPU / set GPU offload when configured (avoids CUDA crashes).
        if settings.ollama_num_gpu is not None:
            opts.setdefault("num_gpu", settings.ollama_num_gpu)
        # Bound output length so CPU latency stays predictable.
        opts.setdefault("num_predict", settings.ollama_num_predict)
        return opts

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **options,
    ) -> str:
        """Send a chat request and return the assistant's text content."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive(),
            "options": self._options(temperature, options),
        }
        resp = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return _strip_thinking(resp.json()["message"]["content"])

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        **options,
    ):
        """Stream a chat response, yielding visible text deltas as they arrive.

        Chain-of-thought wrapped in ``<think>...</think>`` is suppressed: while
        inside a think block nothing is yielded, and everything up to and
        including the closing tag is dropped, so only the final answer streams.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive(),
            "options": self._options(temperature, options),
        }
        in_think = False
        pending = ""  # holds a partial tag that may span chunk boundaries
        with httpx.stream(
            "POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                delta = json.loads(line).get("message", {}).get("content", "")
                if not delta:
                    continue
                pending += delta
                emit, pending, in_think = _consume_think(pending, in_think)
                if emit:
                    yield emit
        # Flush any text held back as a possible-but-incomplete tag.
        if pending and not in_think:
            yield pending

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        **options,
    ) -> str:
        """Single-prompt completion (convenience wrapper over /api/generate)."""
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self._keep_alive(),
            "options": self._options(temperature, options),
        }
        if system:
            payload["system"] = system
        resp = httpx.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return _strip_thinking(resp.json()["response"])

    def health(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


@functools.lru_cache(maxsize=1)
def get_llm() -> OllamaClient:
    return OllamaClient()
