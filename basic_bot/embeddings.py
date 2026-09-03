"""Embedding provider — local llama-server only.

Turns text into vectors via the llama-server /v1/embeddings endpoint.
The rest of the system calls embed() and never knows the details.
"""

import json
import logging
import urllib.request

from basic_bot.config import EMBEDDING_URL

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Qwen3-Embedding via llama-server on localhost.

    Qwen3-Embedding uses an instruction prefix on queries only;
    documents are embedded with no prefix.
    """

    QUERY_INSTRUCT = (
        "Instruct: Given a search query, retrieve relevant passages "
        "from past conversation that answer the query\nQuery: "
    )

    def __init__(self, base_url: str):
        self._endpoint = base_url.rstrip("/") + "/v1/embeddings"
        logger.info("Local embedder ready: %s", self._endpoint)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        if task == "query":
            inputs = [self.QUERY_INSTRUCT + t for t in texts]
        else:
            inputs = texts

        payload = json.dumps({"input": inputs}).encode("utf-8")

        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]


_embedder: LocalEmbedder | None = None


def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    """Embed a list of texts into vectors.

    task is "document" (storing turns) or "query" (searching).
    Returns one vector per input text, in order.
    """
    global _embedder
    if _embedder is None:
        _embedder = LocalEmbedder(EMBEDDING_URL)
    return _embedder.embed(texts, task)
