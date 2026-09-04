"""Local embedding provider.

Turns text into vectors via the llama-server /v1/embeddings endpoint.
Built by the factory, carried on the runtime. Engine-core code
receives it as a parameter — never imports it directly.
"""

import json
import logging
import urllib.request

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
        logger.info("Local embedder configured: %s", self._endpoint)

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
