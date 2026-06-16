from typing import cast

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot

from basic_bot.config import CONVERSATION_COLLECTION, MESSAGE_LIMIT

db = firestore.Client()


def save_message(user_id: str, role: str, content: str, collection: str = CONVERSATION_COLLECTION):
    """Save a message to the user's conversation in Firestore."""
    db.collection(collection).document(user_id)\
        .collection("messages").add({
            "role": role,
            "content": content,
            "created_at": firestore.SERVER_TIMESTAMP,
        })


def get_messages(user_id: str, limit: int = MESSAGE_LIMIT, collection: str = CONVERSATION_COLLECTION) -> list[dict[str, str]]:
    """Get recent messages for a user, in chronological order."""
    messages_ref = (
        db.collection(collection)
        .document(user_id)
        .collection("messages")
    )
    query = messages_ref.order_by(
        "seq", direction=firestore.Query.DESCENDING
    ).limit(limit)

    docs = list(query.stream())
    docs.reverse()

    messages = []
    for doc in docs:
        data = doc.to_dict()
        messages.append({"role": data["role"], "content": data["content"]})
    return messages


def get_summary(user_id: str, collection: str = CONVERSATION_COLLECTION) -> dict | None:
    """Get the rolling summary for a user, if one exists."""
    doc = cast(DocumentSnapshot, db.collection(collection).document(user_id).get())
    if doc.exists:
        data = doc.to_dict()
        if data and "summary" in data:
            return {"summary": data["summary"], "summarized_through": data["summarized_through"]}
    return None


def save_summary(user_id: str, summary: str, summarized_through, collection: str = CONVERSATION_COLLECTION):
    """Write the rolling summary to the user's parent document."""
    db.collection(collection).document(user_id).set({
        "summary": summary,
        "summarized_through": summarized_through,
    }, merge=True)


def save_turn(user_id: str, user_msg: str, agent_msg: str, collection: str = CONVERSATION_COLLECTION) -> int:
    """Atomically save a user+assistant turn with sequence numbers.

    Both messages and the counter increment commit together, or not at all.
    Returns the user message's sequence number.
    """
    parent_ref = db.collection(collection).document(user_id)
    messages_ref = parent_ref.collection("messages")
    transaction = db.transaction()

    @firestore.transactional
    def _commit(transaction) -> int:
        snapshot = cast(DocumentSnapshot, parent_ref.get(transaction=transaction))
        data = snapshot.to_dict() or {}
        next_seq = data.get("next_seq", 1)

        transaction.set(messages_ref.document(), {
            "role": "user",
            "content": user_msg,
            "seq": next_seq,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        transaction.set(messages_ref.document(), {
            "role": "assistant",
            "content": agent_msg,
            "seq": next_seq + 1,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        transaction.set(parent_ref, {"next_seq": next_seq + 2}, merge=True)
        return next_seq

    return _commit(transaction)