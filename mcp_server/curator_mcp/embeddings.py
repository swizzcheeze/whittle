"""
Embedding backends. Supports Ollama and any OpenAI-compatible /v1/embeddings endpoint
(LM Studio, vLLM, llama.cpp server, etc.). One small interface, no model lock-in.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx
import numpy as np


Backend = Literal["ollama", "openai"]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_URL = "http://localhost:1234/v1"   # LM Studio default; override per setup

_RETRY_DELAYS = (5.0, 20.0, 60.0)  # seconds between retries on 5xx / timeout


@dataclass(frozen=True)
class EmbeddingConfig:
    """How to reach the embedding service."""
    backend: Backend = "ollama"
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = "bge-m3"
    api_key: str | None = None      # optional; only needed for some OpenAI-compatible backends
    timeout_s: float = 120.0
    num_gpu: int | None = None      # Ollama only: 0 = CPU-only, None = Ollama default


class EmbeddingClient:
    """Synchronous embedding client. One call per text — simple and predictable."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.Client(timeout=config.timeout_s, headers=headers)

    def close(self) -> None:
        self._client.close()

    def embed_one(self, text: str) -> list[float]:
        last_exc: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                if self.config.backend == "ollama":
                    return self._embed_ollama(text)
                return self._embed_openai(text)
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                # Only retry on server-side errors (5xx) or timeouts; not 4xx (bad request).
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                if delay is not None:
                    time.sleep(delay)
        raise RuntimeError(f"Embedding failed after {len(_RETRY_DELAYS) + 1} attempts") from last_exc

    def embed_many(self, texts: list[str]) -> np.ndarray:
        # We loop instead of batching to keep the API surface uniform across backends
        # (Ollama's /api/embeddings is one-at-a-time; OpenAI accepts arrays).
        # Phase-1 simplicity wins; we can add batching for OpenAI later if needed.
        vecs = [self.embed_one(t) for t in texts]
        return np.asarray(vecs, dtype=np.float32)

    def _embed_ollama(self, text: str) -> list[float]:
        url = f"{self.config.base_url.rstrip('/')}/api/embeddings"
        payload: dict = {"model": self.config.model, "prompt": text}
        if self.config.num_gpu is not None:
            payload["options"] = {"num_gpu": self.config.num_gpu}
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Ollama response missing 'embedding': {data}")
        return data["embedding"]

    def _embed_openai(self, text: str) -> list[float]:
        url = f"{self.config.base_url.rstrip('/')}/embeddings"
        resp = self._client.post(url, json={"model": self.config.model, "input": text})
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected OpenAI-compatible response: {data}") from exc

    def probe(self) -> int:
        """Return the embedding dimensionality. Validates the backend is reachable."""
        vec = self.embed_one("probe")
        return len(vec)
