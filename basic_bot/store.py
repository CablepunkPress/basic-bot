"""Storage protocol for conversation memory.

Defines the MessageStore interface that all backends (Firestore, SQLite)
implement. One instance per bot, scoped to a single collection or database.

The protocol covers two concerns:
  - Conversation state: messages, summaries, sequence counters
  - Vector storage: pre-embedded turn pairs for RAG retrieval

The store never calls the embedding provider. It receives and returns
raw vectors. This keeps the storage backend and embedding provider
independently swappable.
"""

from typing import Protocol


class MessageStore(Protocol):
    """Backend-agnostic conversation storage.

    Implementations:
      - FirestoreMessageStore (cloud deployments)
      - SQLiteMessageStore   (local installs)
    """

    # --- Conversation state ---

    def get_state(self, user_id: str) -> dict:
        """Read the conversation's summary, boundary, and sequence counter.

        Returns:
            {
                "summary": str,           # rolling summary text, "" if none
                "summarized_through": int, # last seq folded into summary, 0 if none
                "next_seq": int,           # next available sequence number, 1 if new
            }
        """
        ...

    def get_messages(self, user_id: str, limit: int) -> list[dict]:
        """Last N messages in chronological order.

        Returns:
            [
                {
                    "role": str,
                    "content": str,
                    "seq": int,
                    "metadata": dict | None,  # assistant messages only
                },
                ...
            ]
        """
        ...

    def get_messages_after(self, user_id: str, after_seq: int) -> list[dict]:
        """All messages with seq > after_seq, ordered ascending.

        Returns:
            [
                {
                    "role": str,
                    "content": str,
                    "seq": int,
                    "metadata": dict | None,  # assistant messages only
                },
                ...
            ]
        """
        ...

    def save_turn(
        self, user_id: str, user_msg: str, agent_msg: str,
        metadata: dict | None = None,
    ) -> int:
        """Atomically save a user+assistant turn pair with sequence numbers.

        Assigns the next two sequence numbers and increments the counter.
        Both messages commit together or not at all.

        Metadata (model_used, display_name, effort, thinking, fallback) is
        stored on the assistant message only. The values should reflect what
        the API actually returned, not what was requested. The API does not 
        return actual effort level used, so requested effort is stored instead.

        Returns:
            The user message's sequence number.
        """
        ...

    def save_summary(
        self, user_id: str, summary: str, summarized_through: int,
    ) -> None:
        """Write the rolling summary and append a copy to the version archive.

        Updates the current summary and boundary in place, and stores a
        timestamped copy in append-only history for auditability.
        """
        ...

    def get_messages_in_range(
        self, user_id: str, start_seq: int, end_seq: int,
    ) -> list[dict]:
        """Messages with seq between start_seq and end_seq inclusive, ordered ascending.

        Returns:
            [
                {
                    "role": str,
                    "content": str,
                    "seq": int,
                    "metadata": dict | None,
                },
                ...
            ]
        """
        ...

    def get_messages_by_date(
        self, user_id: str, after: str, before: str, limit: int = 50,
    ) -> list[dict]:
        """Messages with created_at between after and before, ordered ascending.

        Date strings in ISO format (e.g., "2026-06-15" or "2026-06-15T00:00:00Z").
        Limit caps results to prevent unbounded queries.

        Returns:
            [
                {
                    "role": str,
                    "content": str,
                    "seq": int,
                    "metadata": dict | None,
                    "created_at": str,
                },
                ...
            ]
        """
        ...
    
    # --- Vector storage (RAG) ---

    def store_vectors(
        self, user_id: str, turns: list[dict],
    ) -> int:
        """Store pre-embedded turn pairs for RAG retrieval.

        Each turn dict contains:
            content:   str          — verbatim turn text
            embedding: list[float]  — pre-computed vector
            seq_start: int          — first seq in the turn pair
            seq_end:   int          — last seq in the turn pair

        Returns:
            Number of turns stored.
        """
        ...

    def search_vectors(
        self, user_id: str, query_vector: list[float], limit: int,
    ) -> list[dict]:
        """Find nearest turn vectors by cosine similarity.

        Returns:
            [{"content": str, "seq_start": int, "seq_end": int}, ...]
            ordered by similarity, up to limit results.
        """
        ...

    def clear_vectors(self, user_id: str) -> int:
        """Delete all stored vectors for a session.

        Used when the embedding provider changes and old vectors need
        to be rebuilt from scratch — the vectors live in a different,
        incompatible space and must be discarded before re-embedding.
        Messages, summaries, and state are never touched.

        Returns:
            Number of vectors deleted.
        """
        ...
