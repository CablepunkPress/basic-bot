"""Fold lifecycle — manages the sliding window boundary.

The fold is the mechanism that keeps the sliding window bounded.
When unsummarized messages reach WINDOW_CEILING, the fold:
  1. Embeds the oldest batch into RAG (must succeed)
  2. Generates an updated rolling summary (can fail safely)

This module is engine-core — it has no knowledge of servers,
hardware, or deployment topology. The caller (infrastructure/
orchestration.py for local, or a cloud handler) decides how
to sequence the fold.
"""

import logging

from basic_bot.config import WINDOW_CEILING, WINDOW_FLOOR
from basic_bot.providers.protocol import InferenceProvider
from basic_bot.rag import store_turns
from basic_bot.store import MessageStore
from basic_bot.summary import summarize_batch

logger = logging.getLogger(__name__)


def should_fold(store: MessageStore, user_id: str) -> dict | None:
    """Check whether a fold is needed. Returns state dict if yes, None if no."""
    state = store.get_state(user_id)
    boundary = state["summarized_through"]
    latest = state["next_seq"] - 1

    if latest - boundary >= WINDOW_CEILING:
        return state
    return None


def fold_rag(store: MessageStore, user_id: str, state: dict, embedder) -> list[dict] | None:
    """Embed and store the fold batch into RAG.

    Returns the chunk of messages that were folded, or None if embedding
    failed. When None, the caller must not advance the summary boundary.
    """
    boundary = state["summarized_through"]
    fold_size = WINDOW_CEILING - WINDOW_FLOOR

    tail = store.get_messages_after(user_id, boundary)
    chunk = tail[:fold_size]

    try:
        stored = store_turns(store, user_id, chunk, embedder)
        logger.info("Embedded %d turn(s) in RAG for user %s", stored, user_id)
        return chunk
    except Exception:
        logger.exception(
            "RAG embedding failed for user %s — fold aborted, will retry next cycle",
            user_id,
        )
        return None


def fold_summary(
    provider: InferenceProvider,
    store: MessageStore,
    user_id: str,
    existing_summary: str,
    chunk: list[dict],
    sampling: dict,
) -> None:
    """Generate and save the updated rolling summary.

    If it fails, RAG already captured the data — the summary
    catches up on the next fold.
    """
    new_through = chunk[-1]["seq"]
    batch = [{"role": m["role"], "content": m["content"]} for m in chunk]

    try:
        new_summary = summarize_batch(provider, existing_summary, batch, sampling)

        if not new_summary:
            logger.warning(
                "Summary for user %s came back empty — leaving boundary at %d "
                "to retry next fold",
                user_id, new_through,
            )
            return

        store.save_summary(user_id, new_summary, new_through)
        logger.info(
            "Summarized user %s through seq %d (%d messages folded, %d chars)",
            user_id, new_through, len(batch), len(new_summary),
        )
    except Exception:
        logger.exception(
            "Summary failed for user %s — summary left unchanged, "
            "RAG has the data",
            user_id,
        )


def build_metadata(result: dict) -> dict:
    """Extract the metadata to persist from a chat result."""
    return {
        "model_used": result["model_used"],
        "display_name": result["display_name"],
        "effort": result["effort"],
        "thinking": result["thinking"],
        "fallback": result["fallback"],
    }
