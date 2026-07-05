"""Firestore implementation of MessageStore.

Used by any Cloud Run deployment. All conversation data
lives in Firestore under the collection passed at construction time.

Document structure per user:
    {collection}/{user_id}              — summary, summarized_through, next_seq
    {collection}/{user_id}/messages/*   — individual messages with seq, role, content,
                                          and metadata (assistant messages only)
    {collection}/{user_id}/summaries/*  — append-only summary archive
    {collection}/{user_id}/rag/*        — embedded turn pairs with vectors
"""

import logging
from typing import cast

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

logger = logging.getLogger(__name__)

METADATA_FIELDS = ("model_used", "display_name", "effort", "thinking", "fallback")


def _extract_metadata(data: dict) -> dict | None:
    """Pull metadata fields from a Firestore document dict.

    Returns a metadata dict if any fields are present, None otherwise.
    User messages will have no metadata fields and return None.
    """
    meta = {k: data[k] for k in METADATA_FIELDS if k in data}
    return meta or None


class FirestoreMessageStore:
    """MessageStore backed by Google Cloud Firestore.

    One instance per bot. The collection name is baked in at construction
    so no method needs it as a parameter.
    """

    def __init__(self, collection: str):
        self._collection = collection
        self._db = firestore.Client()

    def _user_ref(self, user_id: str):
        """Reference to the user's parent document."""
        return self._db.collection(self._collection).document(user_id)

    def _messages_ref(self, user_id: str):
        """Reference to the user's messages subcollection."""
        return self._user_ref(user_id).collection("messages")

    # --- Conversation state ---

    def get_state(self, user_id: str) -> dict:
        doc = cast(DocumentSnapshot, self._user_ref(user_id).get())
        data = doc.to_dict() or {}
        return {
            "summary": data.get("summary", ""),
            "summarized_through": data.get("summarized_through", 0),
            "next_seq": data.get("next_seq", 1),
        }

    def get_messages(self, user_id: str, limit: int) -> list[dict]:
        query = self._messages_ref(user_id).order_by(
            "seq", direction=firestore.Query.DESCENDING,
        ).limit(limit)

        docs = list(query.stream())
        docs.reverse()

        messages = []
        for doc in docs:
            data = doc.to_dict()
            msg = {
                "role": data["role"],
                "content": data["content"],
                "seq": data["seq"],
                "metadata": _extract_metadata(data),
            }
            messages.append(msg)
        return messages

    def get_messages_after(self, user_id: str, after_seq: int) -> list[dict]:
        query = self._messages_ref(user_id).where(
            filter=FieldFilter("seq", ">", after_seq),
        ).order_by("seq")

        messages = []
        for doc in query.stream():
            data = doc.to_dict()
            msg = {
                "role": data["role"],
                "content": data["content"],
                "seq": data["seq"],
                "metadata": _extract_metadata(data),
            }
            messages.append(msg)
        return messages

    def save_turn(
        self, user_id: str, user_msg: str, agent_msg: str,
        metadata: dict | None = None,
    ) -> int:
        parent_ref = self._user_ref(user_id)
        messages_ref = self._messages_ref(user_id)
        transaction = self._db.transaction()

        @firestore.transactional
        def _commit(transaction) -> int:
            snapshot = cast(
                DocumentSnapshot, parent_ref.get(transaction=transaction),
            )
            data = snapshot.to_dict() or {}
            next_seq = data.get("next_seq", 1)

            transaction.set(messages_ref.document(), {
                "role": "user",
                "content": user_msg,
                "seq": next_seq,
                "created_at": firestore.SERVER_TIMESTAMP,
            })

            assistant_doc = {
                "role": "assistant",
                "content": agent_msg,
                "seq": next_seq + 1,
                "created_at": firestore.SERVER_TIMESTAMP,
            }
            if metadata:
                for key in METADATA_FIELDS:
                    if key in metadata:
                        assistant_doc[key] = metadata[key]

            transaction.set(messages_ref.document(), assistant_doc)
            transaction.set(parent_ref, {"next_seq": next_seq + 2}, merge=True)
            return next_seq

        return _commit(transaction)

    def save_summary(
        self, user_id: str, summary: str, summarized_through: int,
    ) -> None:
        parent_ref = self._user_ref(user_id)
        parent_ref.set({
            "summary": summary,
            "summarized_through": summarized_through,
        }, merge=True)
        parent_ref.collection("summaries").add({
            "summary": summary,
            "summarized_through": summarized_through,
            "char_count": len(summary),
            "created_at": firestore.SERVER_TIMESTAMP,
        })

    # --- Vector storage (RAG) ---

    def store_vectors(self, user_id: str, turns: list[dict]) -> int:
        rag_ref = self._user_ref(user_id).collection("rag")
        for turn in turns:
            rag_ref.add({
                "content": turn["content"],
                "embedding": Vector(turn["embedding"]),
                "seq_start": turn["seq_start"],
                "seq_end": turn["seq_end"],
                "created_at": firestore.SERVER_TIMESTAMP,
            })
        return len(turns)

    def search_vectors(
        self, user_id: str, query_vector: list[float], limit: int,
    ) -> list[dict]:
        rag_ref = self._user_ref(user_id).collection("rag")
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