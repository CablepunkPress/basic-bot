"""Manual smoke test for the rolling-summary fold path.

Side effects: writes to the throwaway `fold-test` Firestore collection and
makes one real Haiku call. Requires ANTHROPIC_API_KEY in the environment:

    export ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=basic-bot-api-key)

Re-runnable: resets the test doc at the start. Inspect results in Firestore
afterward or read the printed output. Delete the `fold-test` collection when
fully done.
"""
from basic_bot.memory import db, save_turn, get_state, load_window
from basic_bot.chat import maybe_summarize

COLLECTION = "fold-test"
USER = "folduser"


def reset():
    parent = db.collection(COLLECTION).document(USER)
    for msg in parent.collection("messages").stream():
        msg.reference.delete()
    parent.delete()


reset()

for i in range(1, 21):
    save_turn(
        USER,
        f"Message {i}: my lucky number is {i}.",
        f"Noted — lucky number {i} recorded.",
        collection=COLLECTION,
    )

state = get_state(USER, COLLECTION)
print("Before fold:", state["next_seq"] - 1, "messages, boundary at", state["summarized_through"])

fired = maybe_summarize(USER, COLLECTION)
print("Fold fired:", fired)

state = get_state(USER, COLLECTION)
print("After fold: boundary at", state["summarized_through"])
print("Summary:\n", state["summary"][:400])

messages, summary, position = load_window(USER, "what was my first lucky number?", COLLECTION)
print("\nWindow size (incl. current message):", len(messages))
print("Position:", position)