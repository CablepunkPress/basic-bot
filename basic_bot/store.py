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

    def get_messages(self, user_id: str, limit: int) -> list[dict[str, str]]:
        """Last N messages in chronological order.

        Returns:
            [{"role": str, "content": str}, ...]
        """
        ...

    def get_messages_after(self, user_id: str, after_seq: int) -> list[dict]:
        """All messages with seq > after_seq, ordered ascending.

        Returns:
            [{"role": str, "content": str, "seq": int}, ...]
        """
        ...

    def save_turn(self, user_id: str, user_msg: str, agent_msg: str) -> int:
        """Atomically save a user+assistant turn pair with sequence numbers.

        Assigns the next two sequence numbers and increments the counter.
        Both messages commit together or not at all.

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