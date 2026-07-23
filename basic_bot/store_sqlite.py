"""SQLite implementation of MessageStore.

Used by local installs. All conversation data lives in a single
SQLite database file that creates itself on first run.

Table structure:
    messages    — individual messages with seq, role, content, and metadata
    state       — summary, summarized_through, next_seq per user
    summaries   — append-only summary archive
    vectors     — embedded turn pairs for RAG retrieval
"""

import logging
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

METADATA_FIELDS = ("model_used", "display_name", "effort", "thinking", "fallback")

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    session_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    summarized_through INTEGER NOT NULL DEFAULT 0,
    next_seq INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_used TEXT,
    display_name TEXT,
    effort TEXT,
    thinking INTEGER,
    fallback INTEGER,
    UNIQUE(session_id, seq)
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    summarized_through INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq_start INTEGER NOT NULL,
    seq_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_messages_session_timestamp
    ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_vectors_session
    ON vectors(session_id);
"""


def _extract_metadata(row: sqlite3.Row) -> dict | None:
    """Pull metadata fields from a database row.

    Returns a metadata dict if any fields are present, None otherwise.
    """
    meta = {k: row[k] for k in METADATA_FIELDS if row[k] is not None}
    return meta or None


def _embed_to_blob(embedding: list[float]) -> bytes:
    """Pack a float vector into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embed(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a float vector."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteMessageStore:
    """MessageStore backed by a local SQLite database.

    One instance per bot. The database file is created on first use.
    """

    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row factory enabled."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        logger.info("SQLite store ready: %s", self._db_path)

    # --- Conversation state ---

    def get_state(self, user_id: str) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT summary, summarized_through, next_seq "
                "FROM state WHERE session_id = ?",
                (user_id,),
            ).fetchone()
            if row:
                return {
                    "summary": row["summary"],
                    "summarized_through": row["summarized_through"],
                    "next_seq": row["next_seq"],
                }
            return {"summary": "", "summarized_through": 0, "next_seq": 1}
        finally:
            conn.close()

    def get_messages(self, user_id: str, limit: int) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY seq DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "seq": r["seq"],
                    "metadata": _extract_metadata(r),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_messages_after(self, user_id: str, after_seq: int) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND seq > ? "
                "ORDER BY seq",
                (user_id, after_seq),
            ).fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "seq": r["seq"],
                    "metadata": _extract_metadata(r),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def save_turn(
        self, user_id: str, user_msg: str, agent_msg: str,
        metadata: dict | None = None,
    ) -> int:
        conn = self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Get or initialize state
            row = conn.execute(
                "SELECT next_seq FROM state WHERE session_id = ?",
                (user_id,),
            ).fetchone()

            if row:
                next_seq = row["next_seq"]
            else:
                next_seq = 1
                conn.execute(
                    "INSERT INTO state (session_id, next_seq) VALUES (?, ?)",
                    (user_id, 1),
                )

            # Save user message
            conn.execute(
                "INSERT INTO messages "
                "(session_id, seq, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, next_seq, "user", user_msg, now),
            )

            # Save assistant message with metadata
            conn.execute(
                "INSERT INTO messages "
                "(session_id, seq, role, content, created_at, "
                "model_used, display_name, effort, thinking, fallback) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, next_seq + 1, "assistant", agent_msg, now,
                    metadata.get("model_used") if metadata else None,
                    metadata.get("display_name") if metadata else None,
                    metadata.get("effort") if metadata else None,
                    metadata.get("thinking") if metadata else None,
                    metadata.get("fallback") if metadata else None,
                ),
            )

            # Advance counter
            conn.execute(
                "UPDATE state SET next_seq = ? WHERE session_id = ?",
                (next_seq + 2, user_id),
            )

            conn.commit()
            return next_seq
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_summary(
        self, user_id: str, summary: str, summarized_through: int,
    ) -> None:
        conn = self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Upsert current state
            conn.execute(
                "INSERT INTO state (session_id, summary, summarized_through) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "summarized_through = excluded.summarized_through",
                (user_id, summary, summarized_through),
            )

            # Append to archive
            conn.execute(
                "INSERT INTO summaries "
                "(session_id, summary, summarized_through, char_count, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, summary, summarized_through, len(summary), now),
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_messages_in_range(
        self, user_id: str, start_seq: int, end_seq: int,
    ) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND seq >= ? AND seq <= ? "
                "ORDER BY seq",
                (user_id, start_seq, end_seq),
            ).fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "seq": r["seq"],
                    "metadata": _extract_metadata(r),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_messages_by_date(
        self, user_id: str, after: str, before: str, limit: int = 50,
    ) -> list[dict]:
        conn = self._connect()
        try:
            after_dt = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
            before_dt = datetime.fromisoformat(before).replace(tzinfo=timezone.utc)

            if "T" not in before:
                before_dt = before_dt + timedelta(days=1) - timedelta(seconds=1)

            rows = conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND created_at >= ? AND created_at <= ? "
                "ORDER BY created_at LIMIT ?",
                (user_id, after_dt.isoformat(), before_dt.isoformat(), limit),
            ).fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "seq": r["seq"],
                    "metadata": _extract_metadata(r),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # --- Vector storage (RAG) ---

    def store_vectors(self, user_id: str, turns: list[dict]) -> int:
        conn = self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            for turn in turns:
                conn.execute(
                    "INSERT INTO vectors "
                    "(session_id, seq_start, seq_end, content, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        turn["seq_start"],
                        turn["seq_end"],
                        turn["content"],
                        _embed_to_blob(turn["embedding"]),
                        now,
                    ),
                )
            conn.commit()
            return len(turns)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def search_vectors(
        self, user_id: str, query_vector: list[float], limit: int,
    ) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT content, seq_start, seq_end, embedding "
                "FROM vectors WHERE session_id = ?",
                (user_id,),
            ).fetchall()

            scored = []
            for row in rows:
                stored = _blob_to_embed(row["embedding"])
                sim = _cosine_similarity(query_vector, stored)
                scored.append((sim, row))

            scored.sort(key=lambda x: x[0], reverse=True)

            return [
                {
                    "content": row["content"],
                    "seq_start": row["seq_start"],
                    "seq_end": row["seq_end"],
                }
                for _, row in scored[:limit]
            ]
        finally:
            conn.close()

    def clear_vectors(self, user_id: str) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM vectors WHERE session_id = ?",
                (user_id,),
            )
            conn.commit()
            return cursor.rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
