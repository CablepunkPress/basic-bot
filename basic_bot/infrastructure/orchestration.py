"""Local fold orchestration.

Sequences the fold lifecycle with server management: stops chat,
runs embedding (handled by ManagedEmbedder), cycles summary server,
restarts chat. This is local-only infrastructure — engine-core fold
logic lives in basic_bot.fold.
"""

import logging

from basic_bot.fold import fold_rag, fold_summary
from basic_bot.store import MessageStore

logger = logging.getLogger(__name__)


def fold_sequential(
    runtime,
    store: MessageStore,
    user_id: str,
    state: dict,
) -> None:
    """Full fold with sequential server lifecycle."""
    from basic_bot.diagnostics import snapshot_memory
    from basic_bot.infrastructure.server import start, stop, is_running, CHAT, SUMMARY

    existing_summary = state["summary"]
    chat_was_running = is_running(CHAT)

    snapshot_memory("pre-fold")

    if chat_was_running:
        logger.info("Fold triggered — stopping chat server")
        stop(CHAT)
        snapshot_memory("chat-stopped")

    # --- Embedding phase ---
    chunk = fold_rag(store, user_id, state, runtime.embedder)
    snapshot_memory("embedding-done")

    if chunk is None:
        logger.warning("RAG failed — skipping summary")
        if chat_was_running:
            model_id = getattr(runtime.chat_provider, "active_local_model", None)
            start(CHAT, model_id)
            snapshot_memory("chat-resumed")
        return

    # --- Summary phase ---
    start(SUMMARY)
    fold_summary(
        runtime.summary_provider,
        store,
        user_id,
        existing_summary,
        chunk,
        runtime.summary_sampling,
    )
    stop(SUMMARY)
    snapshot_memory("summary-done")

    # --- Resume chat ---
    if chat_was_running:
        model_id = getattr(runtime.chat_provider, "active_local_model", None)
        start(CHAT, model_id)
        snapshot_memory("chat-resumed")
