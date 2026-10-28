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
from basic_bot.infrastructure.server import SUMMARY, start, stop
from basic_bot.store import MessageStore

logger = logging.getLogger(__name__)


def fold_sequential(
    runtime,
    store: MessageStore,
    user_id: str,
    state: dict,
) -> None:
    """Full fold with sequential server lifecycle.

    Chat always resumes, even if a phase fails. A failure then
    propagates, so the session can say memory wasn't updated.
    """
    router = runtime.chat_provider
    existing_summary = state["summary"]

    logger.info("Fold triggered — suspending chat")
    router.suspend()

    try:
        # --- Embedding phase ---
        chunk = fold_rag(store, user_id, state, runtime.embedder)

        # --- Summary phase ---
        # The summary server is stopped whether the summary succeeds
        # or not; a failure goes on to the caller once chat is back.
        try:
            start(SUMMARY)
            fold_summary(
                runtime.summary_provider,
                store,
                user_id,
                existing_summary,
                chunk,
                runtime.summary_sampling,
                prompt_tokens=runtime.summary_prompt_tokens,
                message_tokens=runtime.summary_message_tokens,
            )
        finally:
            stop(SUMMARY)

    finally:
        router.resume()
