"""Manual smoke test for the rolling-summary fold path.

Side effects: writes to the throwaway `fold-test` Firestore collection and
makes one real Haiku call. Requires ANTHROPIC_API_KEY in the environment:

    export ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=basic-bot-api-key)

Re-runnable: resets the test doc at the start. Inspect results in Firestore
afterward or read the printed output. Delete the `fold-test` collection when
fully done.
"""

from google.cloud import firestore

from basic_bot.store_firestore import FirestoreMessageStore
from basic_bot.memory import load_window
from basic_bot.chat import maybe_summarize

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

for i in range(1, 21):
    store.save_turn(
        USER,
        f"Message {i}: my lucky number is {i}.",
        f"Noted — lucky number {i} recorded.",
    )

state = store.get_state(USER)
print("Before fold:", state["next_seq"] - 1, "messages, boundary at", state["summarized_through"])

fired = maybe_summarize(store, USER)
print("Fold fired:", fired)

state = store.get_state(USER)
print("After fold: boundary at", state["summarized_through"])
print("Summary:\n", state["summary"][:400])

messages, summary, position = load_window(store, USER, "what was my first lucky number?")
print("\nWindow size (incl. current message):", len(messages))
print("Position:", position)