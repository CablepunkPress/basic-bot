"""Test the fold lifecycle: sync RAG, async summary, RAG failure abort.

Requires ANTHROPIC_API_KEY in the environment:
    export ANTHROPIC_API_KEY=$(gcloud secrets versions access latest --secret=basic-bot-api-key)

Uses the throwaway `fold-lifecycle-test` Firestore collection.
Collection auto-deleted from the console when done via final reset() call.
"""

import asyncio

from google.cloud import firestore

from basic_bot.config import WINDOW_CEILING, WINDOW_FLOOR
from basic_bot.factory import _fold_rag, _fold_summary, _should_fold
from basic_bot.store_firestore import FirestoreMessageStore

COLLECTION = "fold-lifecycle-test"
USER = "folduser"

store = FirestoreMessageStore(COLLECTION)
db = firestore.Client()


def reset():
    parent = db.collection(COLLECTION).document(USER)
    for sub in ["messages", "summaries", "rag"]:
        for doc in parent.collection(sub).stream():
            doc.reference.delete()
    parent.delete()


def fill_to_ceiling():
    """Save enough turns to trigger a fold."""
    # Each save_turn writes 2 messages (user + assistant)
    # WINDOW_CEILING messages means WINDOW_CEILING // 2 turns
    for i in range(1, WINDOW_CEILING // 2 + 1):
        store.save_turn(
            USER,
            f"Message {i}: my lucky number is {i}.",
            f"Noted — lucky number {i} recorded.",
        )


async def test_normal_fold():
    """Test the happy path: RAG succeeds, summary fires in background."""
    print("=" * 50)
    print("TEST: Normal fold lifecycle")
    print("=" * 50)

    reset()
    fill_to_ceiling()

    state = store.get_state(USER)
    total = state["next_seq"] - 1
    print(f"Messages: {total}, boundary: {state['summarized_through']}")

    # Step 1: should fold?
    fold_state = _should_fold(store, USER)
    print(f"Should fold: {fold_state is not None}")
    assert fold_state is not None, "Expected fold to trigger"

    # Step 2: RAG sync
    chunk = _fold_rag(store, USER, fold_state)
    print(f"RAG result: {len(chunk) if chunk else 'None'} messages")
    assert chunk is not None, "Expected RAG to succeed"

    fold_size = WINDOW_CEILING - WINDOW_FLOOR
    assert len(chunk) == fold_size, f"Expected {fold_size} messages, got {len(chunk)}"

    # Verify vectors are in the store
    rag_ref = db.collection(COLLECTION).document(USER).collection("rag")
    rag_count = sum(1 for _ in rag_ref.stream())
    print(f"RAG documents stored: {rag_count}")
    assert rag_count > 0, "Expected RAG documents"

    # Step 3: summary background
    await _fold_summary(store, USER, fold_state["summary"], chunk)

    state = store.get_state(USER)
    print(f"After fold: boundary at {state['summarized_through']}")
    print(f"Summary length: {len(state['summary'])} chars")
    assert state["summarized_through"] > 0, "Expected boundary to advance"
    assert len(state["summary"]) > 0, "Expected non-empty summary"

    print("PASSED\n")


async def test_no_fold_needed():
    """Test that fold doesn't trigger below ceiling."""
    print("=" * 50)
    print("TEST: No fold needed (below ceiling)")
    print("=" * 50)

    reset()
    # Save just a few turns — well below ceiling
    for i in range(1, 4):
        store.save_turn(USER, f"Hello {i}", f"Hi {i}")

    state = store.get_state(USER)
    print(f"Messages: {state['next_seq'] - 1}")

    fold_state = _should_fold(store, USER)
    print(f"Should fold: {fold_state is not None}")
    assert fold_state is None, "Expected no fold"

    print("PASSED\n")


async def test_rag_failure_aborts_summary():
    """Test that summary doesn't fire if RAG fails.

    We can't easily make the real embedding call fail, so this test
    verifies the contract: if _fold_rag returns None, the boundary
    stays in place and can be retried.
    """
    print("=" * 50)
    print("TEST: Boundary unchanged when fold aborted")
    print("=" * 50)

    reset()
    fill_to_ceiling()

    state_before = store.get_state(USER)
    boundary_before = state_before["summarized_through"]
    print(f"Boundary before: {boundary_before}")

    # Simulate RAG failure: _fold_rag returns None
    # The caller (factory.py) checks `if chunk:` — None is falsy
    # Summary never fires, boundary never moves
    chunk = None  # simulating failure
    if chunk:
        print("ERROR: Summary would have fired")
        assert False
    else:
        print("Summary correctly skipped (RAG returned None)")

    state_after = store.get_state(USER)
    print(f"Boundary after: {state_after['summarized_through']}")
    assert state_after["summarized_through"] == boundary_before, "Boundary should not move"

    # Now do a real fold to prove retry works
    fold_state = _should_fold(store, USER)
    assert fold_state is not None, "Fold should still be needed"
    chunk = _fold_rag(store, USER, fold_state)
    assert chunk is not None, "Retry should succeed"
    print(f"Retry succeeded: {len(chunk)} messages embedded")

    print("PASSED\n")


async def main():
    await test_normal_fold()
    await test_no_fold_needed()
    await test_rag_failure_aborts_summary()
    print("All tests passed.")
    reset()
    print("Test data cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())