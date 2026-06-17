"""Long-term memory — RAG over past conversation turns.

Stores verbatim turn pairs with embedding vectors at fold time (when
messages leave the sliding window and enter the summary). Retrieves
them by semantic search when the agent invokes search_memory.

The embedding call is delegated to embeddings.py, which handles
provider selection (Vertex, Gemma, local). This module owns storage
and retrieval only.
"""

import logging

from google.cloud import firestore as firestore_module
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

from basic_bot.config import CONVERSATION_COLLECTION
from basic_bot.embeddings import embed
from basic_bot.memory import db

logger = logging.getLogger(__name__)

RAG_RESULT_LIMIT = 5


def store_turns(
    user_id: str,
    messages: list[dict],
    collection: str = CONVERSATION_COLLECTION,
) -> int:
    """Embed and store turn pairs from a folded batch into the RAG subcollection.

    Takes the raw message dicts (with role, content, seq) from the fold batch,
    pairs them into turns, embeds in one batch call, and writes to Firestore.
    Returns the number of turns stored.
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

    if not turns:
        return 0

    texts = [t["content"] for t in turns]
    vectors = embed(texts, task="document")

    rag_ref = db.collection(collection).document(user_id).collection("rag")
    for turn, vector in zip(turns, vectors):
        rag_ref.add({
            "content": turn["content"],
            "embedding": Vector(vector),
            "seq_start": turn["seq_start"],
            "seq_end": turn["seq_end"],
            "created_at": firestore_module.SERVER_TIMESTAMP,
        })

    logger.info(
        "Stored %d turn(s) in RAG for user %s (seq %d–%d)",
        len(turns), user_id, turns[0]["seq_start"], turns[-1]["seq_end"],
    )
    return len(turns)


def search_memory(
    user_id: str,
    query: str,
    limit: int = RAG_RESULT_LIMIT,
    collection: str = CONVERSATION_COLLECTION,
) -> list[dict]:
    """Search past conversation turns by semantic similarity.

    Returns up to `limit` results, each with verbatim content and seq range.
    """
    query_vector = embed([query], task="query")[0]

    rag_ref = db.collection(collection).document(user_id).collection("rag")
    results = rag_ref.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=limit,
    )

    matches = []
    for doc in results.stream():
        data = doc.to_dict()
        if data:
            matches.append({
                "content": data["content"],
                "seq_start": data["seq_start"],
                "seq_end": data["seq_end"],
            })
    return matches