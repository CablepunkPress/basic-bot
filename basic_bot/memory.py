from typing import cast

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.base_query import FieldFilter

from basic_bot.config import CONVERSATION_COLLECTION, MESSAGE_LIMIT

db = firestore.Client()


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
    """Write the rolling summary to the parent doc, and append a versioned
    copy to the summaries archive subcollection (append-only history)."""
    parent_ref = db.collection(collection).document(user_id)
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


def get_state(user_id: str, collection: str = CONVERSATION_COLLECTION) -> dict:
    """Read the conversation's parent doc: summary, boundary, and counter.

    One read returns everything the loader and the trigger need.
    """
    doc = cast(DocumentSnapshot, db.collection(collection).document(user_id).get())
    data = doc.to_dict() or {}
    return {
        "summary": data.get("summary", ""),
        "summarized_through": data.get("summarized_through", 0),
        "next_seq": data.get("next_seq", 1),
    }


def get_messages_after(user_id: str, after_seq: int, collection: str = CONVERSATION_COLLECTION) -> list[dict]:
    """All messages with seq greater than after_seq, ordered by seq ascending."""
    messages_ref = (
        db.collection(collection)
        .document(user_id)
        .collection("messages")
    )
    query = messages_ref.where(
        filter=FieldFilter("seq", ">", after_seq)
    ).order_by("seq")

    messages = []
    for doc in query.stream():
        data = doc.to_dict()
        messages.append({"role": data["role"], "content": data["content"], "seq": data["seq"]})
    return messages


def load_window(
    user_id: str,
    current_message: str,
    collection: str = CONVERSATION_COLLECTION,
) -> tuple[list[dict[str, str]], str, dict]:
    """Assemble the model's context: verbatim window + summary + position.

    Returns (messages, summary, position) where messages is the API-ready
    list (everything after the summary boundary, plus the incoming turn).
    """
    state = get_state(user_id, collection)
    after = state["summarized_through"]
    window = get_messages_after(user_id, after, collection)

    messages = [{"role": m["role"], "content": m["content"]} for m in window]
    messages.append({"role": "user", "content": current_message})

    latest = state["next_seq"] - 1
    position = {
        "total": latest,
        "summarized_through": after,
        "window_start": window[0]["seq"] if window else None,
        "window_end": window[-1]["seq"] if window else None,
    }

    return messages, state["summary"], position