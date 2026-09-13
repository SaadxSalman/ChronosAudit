"""Pluggable embedding provider.

* ``openai`` — ``text-embedding-3-small`` (1536 dims),
* ``ollama`` — e.g. ``nomic-embed-text`` (768 dims) served by a local Ollama,
* ``hashing`` — deterministic count-based hashing bag-of-words (512 dims) that
  requires no network; used for sandboxed/offline demos.

All providers expose the same ``embed``/``embed_batch`` interface so the rest
of the stack never knows which one is active.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Optional

import numpy as np

from chronos.config import Settings, get_settings

log = logging.getLogger("chronos.embeddings")

_WORD = re.compile(r"[a-z0-9_']+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "with", "this", "that", "as",
}


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words embedding."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return [t for t in _WORD.findall(text.lower()) if t not in _STOP]

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        freqs = {}
        for tok in self._tokens(text):
            freqs[tok] = freqs.get(tok, 0) + 1
        for tok, count in freqs.items():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0 + np.log(count)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class EmbeddingProvider:
    """Facade that lazily builds the right real vectoriser and falls back to
    hashing on any failure."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._backing: object = None
        self._dim: Optional[int] = None
        self._kind: str = ""
        self._init()

    def _init(self) -> None:
        cfg = self.settings
        kind = cfg.embedding_provider
        try:
            if kind == "openai":
                if not cfg.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY missing")
                from langchain_openai import OpenAIEmbeddings

                self._backing = OpenAIEmbeddings(
                    model=cfg.embedding_model or "text-embedding-3-small",
                    api_key=cfg.openai_api_key,
                    base_url=cfg.openai_base_url or None,
                )
                self._dim = 1536
                self._kind = "openai"
            elif kind == "ollama":
                from langchain_ollama import OllamaEmbeddings

                self._backing = OllamaEmbeddings(
                    model=cfg.embedding_model or "nomic-embed-text",
                    base_url=cfg.ollama_base_url,
                )
                self._dim = cfg.embedding_dim or 768
                self._kind = "ollama"
            else:
                raise RuntimeError("hashing is the safe default")
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedding provider %r unavailable (%s); using hashing.", kind, exc)
            self._backing = HashingEmbedder(dim=cfg.embedding_dim or 512)
            self._dim = cfg.embedding_dim or 512
            self._kind = "hashing"

    @property
    def dim(self) -> int:
        assert self._dim is not None
        return self._dim

    @property
    def kind(self) -> str:
        return self._kind

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            if self._kind == "hashing":
                return self._backing.embed(text)  # type: ignore[attr-defined]
            if self._kind == "openai":
                vec = self._backing.embed_query(text)  # type: ignore[attr-defined]
                return np.asarray(vec, dtype=np.float32)
            vec = self._backing.embed_query(text)  # type: ignore[attr-defined]
            return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._kind == "hashing":
            return np.vstack([self.embed(t) for t in texts])
        if self._kind == "openai":
            vecs = self._backing.embed_documents(texts)  # type: ignore[attr-defined]
            return np.asarray(vecs, dtype=np.float32)
        # Ollama has no batch embedding in langchain-ollama < 0.2; loop.
        return np.vstack([self.embed(t) for t in texts])