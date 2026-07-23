"""Re-embed a message archive after an embedding provider change.

Vectors are a derived index over messages — they can always be rebuilt.
When the embedding provider changes (e.g. Vertex AI to a local
llama-server model), old vectors live in an incompatible space and
must be discarded and regenerated from the raw messages, which are
never touched by this script.

Only messages the fold system has already folded (seq <= the
summarized_through boundary) are re-embedded. Messages still in the
sliding window were never embedded — the next natural fold handles
them. This keeps reembed and the fold lifecycle from double-embedding
the same turns.

Usage:
    python -m basic_bot.scripts.reembed basic-bot-sessions
    python -m basic_bot.scripts.reembed basic-bot-sessions --user local
    python -m basic_bot.scripts.reembed basic-bot-sessions --yes
"""

import argparse
import logging
import os
import sqlite3
import sys

from basic_bot.config import EMBEDDING_PROVIDER, SQLITE_DIR
from basic_bot.embeddings import embed
from basic_bot.rag import pair_turns
from basic_bot.store_sqlite import SQLiteMessageStore

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def list_sessions(db_path: str) -> list[str]:
    """Distinct session_ids present in the messages table.

    Reads the file directly — this is outside the MessageStore protocol,
    which is scoped to one session at a time by design.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT session_id FROM messages").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def reembed_session(store: SQLiteMessageStore, user_id: str) -> int:
    cleared = store.clear_vectors(user_id)
    logger.info("Cleared %d old vector(s) for %s", cleared, user_id)

    # Only re-embed what the fold system has already folded. Messages
    # past the boundary are still in the sliding window — they were
    # never embedded, and the next natural fold will handle them.
    boundary = store.get_state(user_id)["summarized_through"]
    if boundary == 0:
        logger.info("No folded history for %s — nothing to embed", user_id)
        return 0

    messages = store.get_messages_in_range(user_id, 1, boundary)
    if not messages:
        logger.info("No messages for %s — nothing to embed", user_id)
        return 0

    turns = pair_turns(messages)
    if not turns:
        logger.info("No complete turn pairs for %s", user_id)
        return 0

    total = 0
    for i in range(0, len(turns), BATCH_SIZE):
        batch = turns[i:i + BATCH_SIZE]
        texts = [t["content"] for t in batch]
        vectors = embed(texts, task="document")
        store_ready = [{**t, "embedding": v} for t, v in zip(batch, vectors)]
        total += store.store_vectors(user_id, store_ready)
        logger.info("Embedded %d/%d turns for %s", total, len(turns), user_id)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collection",
        help="Collection name — matches the .db filename, e.g. basic-bot-sessions",
    )
    parser.add_argument(
        "--user",
        help="Only re-embed this session_id. Default: every session in the database.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    db_path = os.path.join(SQLITE_DIR, f"{args.collection}.db")
    if not os.path.exists(db_path):
        sys.exit(f"No database found at {db_path}")

    sessions = [args.user] if args.user else list_sessions(db_path)
    if not sessions:
        sys.exit("No sessions found in database")

    print(f"Database:  {db_path}")
    print(f"Provider:  {EMBEDDING_PROVIDER}")
    print(f"Sessions:  {', '.join(sessions)}")

    if not args.yes:
        confirm = input(
            "\nThis deletes existing vectors and rebuilds them from stored "
            "messages. Messages themselves are never touched. Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            sys.exit("Aborted — no changes made")

    store = SQLiteMessageStore(db_path)
    grand_total = 0
    for user_id in sessions:
        grand_total += reembed_session(store, user_id)

    print(f"\nDone — {grand_total} turn(s) re-embedded across {len(sessions)} session(s).")


if __name__ == "__main__":
    main()
