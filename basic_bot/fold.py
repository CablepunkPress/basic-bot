"""Fold lifecycle — manages the sliding window boundary.

The fold is the mechanism that keeps the sliding window bounded.
When unsummarized messages reach WINDOW_CEILING, the fold:
  1. Embeds the oldest batch into RAG
  2. Generates an updated rolling summary, and moves the boundary

Only a saved summary moves the boundary. If either step fails, the
error is raised, so the caller can say that memory wasn't updated;
the boundary stays put, and the next fold sends the same batch again.
Storing a turn's vector again replaces it (see basic_bot.store), so a
retry never duplicates the archive. A failure must never be logged
and passed over: until October 28 both steps were, the front end was
told memory was up to date, and the same failing fold ran before
every message.

This module is engine-core — it has no knowledge of servers,
hardware, or deployment topology. The caller (infrastructure/
orchestration.py for local, or a cloud handler) decides how
to sequence the fold.
"""

import logging

import basic_bot.config as config
from basic_bot.providers.protocol import TokenizingProvider
from basic_bot.rag import store_turns
from basic_bot.store import MessageStore
from basic_bot.summary import SummaryError, summarize_batch

logger = logging.getLogger(__name__)


def should_fold(store: MessageStore, user_id: str) -> dict | None:
    """Check whether a fold is needed. Returns state dict if yes, None if no."""
    state = store.get_state(user_id)
    boundary = state["summarized_through"]
    latest = state["next_seq"] - 1

    if latest - boundary >= config.WINDOW_CEILING:
        return state
    return None


def fold_rag(store: MessageStore, user_id: str, state: dict, embedder) -> list[dict]:
    """Embed and store the fold batch into RAG.

    Returns the chunk of messages that were folded. Raises if
    embedding fails; the boundary then stays where it was.
    """
    boundary = state["summarized_through"]
    fold_size = config.WINDOW_CEILING - config.WINDOW_FLOOR

    tail = store.get_messages_after(user_id, boundary)
    chunk = tail[:fold_size]

    stored = store_turns(store, user_id, chunk, embedder)
    logger.info("Embedded %d turn(s) in RAG for user %s", stored, user_id)
    return chunk


def fold_summary(
    provider: TokenizingProvider,
    store: MessageStore,
    user_id: str,
    existing_summary: str,
    chunk: list[dict],
    sampling: dict,
    *,
    prompt_tokens: int,
    message_tokens: int,
) -> None:
    """Generate and save the updated rolling summary, moving the boundary.

    prompt_tokens and message_tokens are the summary model's limits,
    from the hardware profile (see basic_bot.summary). Raises if the
    summary can't be made or comes back empty; the boundary then stays
    where it was.
    """
    new_through = chunk[-1]["seq"]
    batch = [
        {"role": m["role"], "content": m["content"], "seq": m["seq"]}
        for m in chunk
    ]

    new_summary = summarize_batch(
        provider, existing_summary, batch, sampling,
        prompt_tokens=prompt_tokens, message_tokens=message_tokens,
    )
    if not new_summary:
        raise SummaryError(
            f"The summary through seq {new_through} came back empty."
        )

    store.save_summary(user_id, new_summary, new_through)
    logger.info(
        "Summarized user %s through seq %d (%d messages folded, %d chars)",
        user_id, new_through, len(batch), len(new_summary),
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
