"""Manual smoke test for the rolling-summary fold path.

Side effects: writes to the throwaway `fold-test` Firestore collection and
makes one real Haiku call. Requires ANTHROPIC_API_KEY in the environment:

    export ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=basic-bot-api-key)

Re-runnable: resets the test doc at the start. Inspect results in Firestore
afterward or read the printed output. Delete the `fold-test` collection when
fully done.
"""

from google.cloud import firestore

from basic_bot.config import WINDOW_CEILING, WINDOW_FLOOR
from basic_bot.memory import load_window
from basic_bot.rag import store_turns
from basic_bot.store_firestore import FirestoreMessageStore
from basic_bot.summary import summarize_batch

COLLECTION = "fold-test"
USER = "folduser"

store = FirestoreMessageStore(COLLECTION)

# Raw Firestore cleanup — no protocol method for deleting test data
db = firestore.Client()


def reset():
    parent = db.collection(COLLECTION).document(USER)
    for msg in parent.collection("messages").stream():
        msg.reference.delete()
    parent.delete()


reset()

# Fill to ceiling so a fold triggers
for i in range(1, WINDOW_CEILING + 1):
    store.save_turn(
        USER,
        f"Message {i}: my lucky number is {i}.",
        f"Noted — lucky number {i} recorded.",
    )

state = store.get_state(USER)
print("Before fold:", state["next_seq"] - 1, "messages, boundary at", state["summarized_through"])

# Fold: RAG sync, then summary
fold_size = WINDOW_CEILING - WINDOW_FLOOR
tail = store.get_messages_after(USER, state["summarized_through"])
chunk = tail[:fold_size]

stored = store_turns(store, USER, chunk)
print(f"RAG: embedded {stored} turn(s)")

batch = [{"role": m["role"], "content": m["content"]} for m in chunk]
new_summary = summarize_batch(state["summary"], batch)
print("Summary fired:", bool(new_summary))

if new_summary:
    new_through = chunk[-1]["seq"]
    store.save_summary(USER, new_summary, new_through)

state = store.get_state(USER)
print("After fold: boundary at", state["summarized_through"])
print("Summary:\n", state["summary"][:400])

messages, summary, position = load_window(store, USER, "what was my first lucky number?")
print("\nWindow size (incl. current message):", len(messages))
print("Position:", position)