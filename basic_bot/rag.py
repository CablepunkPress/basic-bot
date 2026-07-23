"""Long-term memory — RAG over past conversation turns.

Stores verbatim turn pairs with embedding vectors at fold time (when
messages leave the sliding window and enter the summary). Retrieves
them by semantic search when the agent invokes search_archive.

The embedding call is delegated to embeddings.py, which handles
provider selection (Vertex, local llama-server). This module pairs
text with vectors and hands them to the store. It never writes to
the database directly.
"""

import logging

from basic_bot.embeddings import embed
from basic_bot.store import MessageStore

logger = logging.getLogger(__name__)

RAG_RESULT_LIMIT = 5


def pair_turns(messages: list[dict]) -> list[dict]:
    """Group raw messages into user+assistant turn pairs.

    Shared by the normal fold path (store_turns) and the re-embedding
    script (reembed.py) so both chunk history identically.

    Returns:
        [{"content": str, "seq_start": int, "seq_end": int}, ...]
    """
    turns = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            content = f"User: {messages[i]['content']}\nAssistant: {messages[i + 1]['content']}"
            turns.append({
                "content": content,
                "seq_start": messages[i]["seq"],
                "seq_end": messages[i + 1]["seq"],
            })
            i += 2
        else:
            i += 1
    return turns


def store_turns(
    store: MessageStore,
    user_id: str,
    messages: list[dict],
) -> int:
    """Embed and store turn pairs from a folded batch.

    Takes the raw message dicts (with role, content, seq) from the fold batch,
    pairs them into turns, embeds in one batch call, and writes to the store.
    Returns the number of turns stored.
    """
    turns = pair_turns(messages)
    if not turns:
        return 0

    texts = [t["content"] for t in turns]
    vectors = embed(texts, task="document")

    store_ready = []
    for turn, vector in zip(turns, vectors):
        store_ready.append({
            "content": turn["content"],
            "embedding": vector,
            "seq_start": turn["seq_start"],
            "seq_end": turn["seq_end"],
        })

    stored = store.store_vectors(user_id, store_ready)

    logger.info(
        "Stored %d turn(s) in RAG for user %s (seq %d–%d)",
        stored, user_id, turns[0]["seq_start"], turns[-1]["seq_end"],
    )
    return stored


def search_memory(
    store: MessageStore,
    user_id: str,
    query: str,
    limit: int = RAG_RESULT_LIMIT,
) -> list[dict]:
    """Search past conversation turns by semantic similarity.

    Returns up to limit results, each with verbatim content and seq range.
    """
    query_vector = embed([query], task="query")[0]
    return store.search_vectors(user_id, query_vector, limit)
