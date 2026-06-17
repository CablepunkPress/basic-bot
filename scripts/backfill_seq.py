from google.cloud import firestore

db = firestore.Client()


def backfill(collection: str):
    # list_documents() returns virtual parents too (docs that exist only
    # because they have a messages subcollection) — which is exactly why
    # the console wasn't showing client-001.
    parents = db.collection(collection).list_documents()
    for parent in parents:
        msgs = list(
            parent.collection("messages").order_by("created_at").stream()
        )
        for i, doc in enumerate(msgs, start=1):
            doc.reference.update({"seq": i})
        parent.set({"next_seq": len(msgs) + 1}, merge=True)
        print(f"{collection}/{parent.id}: {len(msgs)} messages, next_seq={len(msgs) + 1}")


if __name__ == "__main__":
    for c in ["basic-bot-sessions", "cablepunk-sessions"]:
        print(f"--- {c} ---")
        backfill(c)
    print("Done.")