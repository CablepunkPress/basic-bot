"""Local fold orchestration.

Sequences the fold lifecycle with server management: stops chat,
cycles embedding and summary servers, restarts chat. This is
local-only infrastructure — engine-core fold logic lives in
basic_bot.fold.
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
    """Full fold with sequential server lifecycle.

    Stops the chat server, cycles through embedding and summary
    servers one at a time, then restarts chat. Only one llama-server
    process runs at any given moment.
    """
    from basic_bot.diagnostics import snapshot_memory
    from basic_bot.infrastructure.server import start, stop, CHAT, EMBEDDING, SUMMARY

    existing_summary = state["summary"]

    snapshot_memory("pre-fold")
    logger.info("Fold triggered — stopping chat server")

    stop(CHAT)
    snapshot_memory("chat-stopped")

    # --- Embedding phase ---
    start(EMBEDDING)
    chunk = fold_rag(store, user_id, state)
    stop(EMBEDDING)
    snapshot_memory("embedding-done")

    if chunk is None:
        logger.warning("RAG failed — restarting chat, skipping summary")
        start(CHAT)
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
    start(CHAT)
    snapshot_memory("chat-resumed")
