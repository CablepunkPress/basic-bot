"""One-time backfill: embed all existing messages into the RAG subcollection."""

from google.cloud import firestore

from basic_bot.store_firestore import FirestoreMessageStore
from basic_bot.rag import store_turns

COLLECTION = "basic-bot-sessions"
USER_ID = "client-001"

store = FirestoreMessageStore(COLLECTION)

# Load all messages ordered by seq (raw query — no protocol method for "all messages")
db = firestore.Client()
messages_ref = (
    db.collection(COLLECTION)
    .document(USER_ID)
    .collection("messages")
    .order_by("seq")
)

all_messages = []
for doc in messages_ref.stream():
    data = doc.to_dict()
    all_messages.append({
        "role": data["role"],
        "content": data["content"],
        "seq": data["seq"],
    })

print(f"Found {len(all_messages)} messages")

# Check what's already in RAG to avoid duplicates
rag_ref = db.collection(COLLECTION).document(USER_ID).collection("rag")
existing_seqs = set()
for doc in rag_ref.stream():
    data = doc.to_dict()
    if data:
        existing_seqs.add(data.get("seq_start"))

if existing_seqs:
    print(f"Already in RAG: {len(existing_seqs)} turns (seq_starts: {sorted(existing_seqs)})")

# Process in chunks of 20 (matching SUMMARY_INTERVAL) to keep embedding batches reasonable
BATCH_SIZE = 20
total_stored = 0

for i in range(0, len(all_messages), BATCH_SIZE):
    batch = all_messages[i:i + BATCH_SIZE]
    first_seq = batch[0]["seq"]
    last_seq = batch[-1]["seq"]

    # Skip if this batch's first turn is already embedded
    if first_seq in existing_seqs:
        print(f"  Skipping seq {first_seq}–{last_seq} (already in RAG)")
        continue

    stored = store_turns(store, USER_ID, batch)
    total_stored += stored
    print(f"  Embedded seq {first_seq}–{last_seq}: {stored} turns")

print(f"\nDone. Stored {total_stored} total turns in RAG.")