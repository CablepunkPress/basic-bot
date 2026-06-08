from google.cloud import firestore

from basic_bot.config import CONVERSATION_COLLECTION, MESSAGE_LIMIT

db = firestore.Client()


def save_message(user_id: str, role: str, content: str):
    """Save a message to the user's conversation in Firestore."""
    db.collection(CONVERSATION_COLLECTION).document(user_id)\
        .collection("messages").add({
            "role": role,
            "content": content,
            "created_at": firestore.SERVER_TIMESTAMP,
        })


def get_messages(user_id: str, limit: int = MESSAGE_LIMIT) -> list[dict[str, str]]:
    """Get recent messages for a user, in chronological order."""
    messages_ref = (
        db.collection(CONVERSATION_COLLECTION)
        .document(user_id)
        .collection("messages")
    )
    query = messages_ref.order_by(
        "created_at", direction=firestore.Query.DESCENDING
    ).limit(limit)

    docs = list(query.stream())
    docs.reverse()

    messages = []
    for doc in docs:
        data = doc.to_dict()
        messages.append({"role": data["role"], "content": data["content"]})
    return messages