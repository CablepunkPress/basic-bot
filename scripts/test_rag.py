"""Smoke test for RAG storage and retrieval."""

from google.cloud import firestore

from basic_bot.store_firestore import FirestoreMessageStore
from basic_bot.rag import store_turns, search_memory

COLLECTION = "basic-bot-sessions"
TEST_USER = "rag-test"

store = FirestoreMessageStore(COLLECTION)

# Fake turn pairs with seq numbers
test_messages = [
    {"role": "user", "content": "Elephants like peanuts better than people in Paris.", "seq": 1},
    {"role": "assistant", "content": "That's an interesting claim about elephants and their peanut preferences!", "seq": 2},
    {"role": "user", "content": "I'm building a three-tier memory system for my AI agents.", "seq": 3},
    {"role": "assistant", "content": "That sounds like a solid architecture with short, mid, and long-term tiers.", "seq": 4},
    {"role": "user", "content": "My favorite retro game is The Legend of Zelda on NES.", "seq": 5},
    {"role": "assistant", "content": "A classic! The open-world design was groundbreaking for 1986.", "seq": 6},
]

print("Storing 3 turns...")
stored = store_turns(store, TEST_USER, test_messages)
print(f"Stored {stored} turns")

print("\nSearching for 'elephants peanuts'...")
results = search_memory(store, TEST_USER, "elephants peanuts")
for r in results:
    print(f"  seq {r['seq_start']}–{r['seq_end']}: {r['content'][:80]}...")

print("\nSearching for 'retro NES games'...")
results = search_memory(store, TEST_USER, "retro NES games")
for r in results:
    print(f"  seq {r['seq_start']}–{r['seq_end']}: {r['content'][:80]}...")

# Raw Firestore cleanup — no protocol method for deleting test data
print("\nCleaning up test data...")
db = firestore.Client()
rag_ref = db.collection(COLLECTION).document(TEST_USER).collection("rag")
for doc in rag_ref.stream():
    doc.reference.delete()
db.collection(COLLECTION).document(TEST_USER).delete()
print("Done.")