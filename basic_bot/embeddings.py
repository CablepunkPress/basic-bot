"""Embedding provider abstraction.

Turns text into vectors. The rest of the system calls embed() and never
knows which provider produced the vectors. Swap providers with the
EMBEDDING_PROVIDER env var without touching any caller.

Providers translate a generic task ("document" | "query") into their own
mechanism: the Google GenAI SDK uses task_type in EmbedContentConfig;
a future local model (EmbeddingGemma in a confidential VM) would use
prompt prefixes. Provider-specific imports are deferred into each class
so an environment only installs the SDK for the provider it actually uses.
"""

import logging
from typing import Protocol

from basic_bot.config import (
    EMBEDDING_LOCATION,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        ...


class VertexEmbedder:
    """Google GenAI embeddings via Vertex AI. Used by oravec.io (cloud)."""

    def __init__(self, model_name: str, location: str):
        self._model_name = model_name
        self._location = location
        self._client = None  # lazy — no network or credentials at import time

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                location=self._location,
            )
            logger.info("GenAI embedder ready: %s (%s)", self._model_name, self._location)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        self._ensure_client()
        assert self._client is not None
        from google.genai.types import EmbedContentConfig

        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=EmbedContentConfig(task_type=task_type),
        )
        if not result.embeddings:
            return []
        return [e.values for e in result.embeddings if e.values is not None]


def _build_embedder() -> EmbeddingProvider:
    if EMBEDDING_PROVIDER == "vertex":
        return VertexEmbedder(EMBEDDING_MODEL, EMBEDDING_LOCATION)
    raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER!r}")


_embedder: EmbeddingProvider | None = None


def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    """Embed a list of texts into vectors.

    task is "document" (storing turns) or "query" (searching). Returns one
    vector per input text, in order.
    """
    global _embedder
    if _embedder is None:
        _embedder = _build_embedder()
    return _embedder.embed(texts, task)