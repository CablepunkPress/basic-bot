"""Local fold orchestration.

Sequences the fold with server lifecycle: suspends chat, runs the
embedding phase (the embedder manages its own server), cycles the
summary server, and resumes chat. This is local-only infrastructure;
engine-core fold logic lives in basic_bot.fold.

Chat is suspended through the router, which owns the chat server,
so the router always knows what is running.
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

    Chat always resumes, even if a phase fails. A failure still
    propagates once chat is back.
    """
    from basic_bot.diagnostics import snapshot_memory
    from basic_bot.infrastructure.server import start, stop, SUMMARY

    router = runtime.chat_provider
    existing_summary = state["summary"]

    snapshot_memory("pre-fold")
    logger.info("Fold triggered — suspending chat")
    router.suspend()
    snapshot_memory("chat-suspended")

    try:
        # --- Embedding phase ---
        chunk = fold_rag(store, user_id, state, runtime.embedder)
        snapshot_memory("embedding-done")

        if chunk is None:
            logger.warning("RAG failed — skipping summary")
            return

        # --- Summary phase ---
        # fold_summary handles its own failures. The try covers the
        # server: if it fails to start, the boundary stays put.
        try:
            start(SUMMARY)
            fold_summary(
                runtime.summary_provider,
                store,
                user_id,
                existing_summary,
                chunk,
                runtime.summary_sampling,
            )
        finally:
            stop(SUMMARY)
            snapshot_memory("summary-done")

    finally:
        router.resume()
        snapshot_memory("chat-resumed")
