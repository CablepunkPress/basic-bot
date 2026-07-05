# Basic Bot

General-purpose AI assistant engine

### Memory System

Basic Bot uses a three-tier memory architecture: *short-term*, *intermediate-term*, and *long-term*. Each tier serves a different purpose on two axes: how long the data persists, and how quickly it updates.

**Persistence axis — what the bot remembers:**

- **Short-term (sliding window):** The most recent messages in verbatim form. The window holds between 20 messages (WINDOW_FLOOR) and 40 messages (WINDOW_CEILING) before a fold compresses the oldest batch. This is what the model sees directly in its context.
- **Intermediate-term (rolling summary):** A compressed narrative of everything that has left the sliding window. Preserves durable facts — names, preferences, decisions, ongoing topics — without consuming the full token cost of raw messages.
- **Long-term (RAG archive):** Verbatim turn pairs embedded as vectors and stored permanently. Searchable by the agent via the `search_archive` tool. The summary knows the facts; RAG has the receipts.

**Timing axis — when each tier updates during a fold:**

- **Immediate (sliding window):** Updates every turn. The model always sees the most recent messages.
- **Near-immediate (RAG):** Embeds synchronously at fold time. Adds a second or two to the fold turn. Ingestion completes before the response returns to the user.
- **Eventual (summary):** Generates asynchronously in the background at fold time. The Haiku summarization call takes 10–15 seconds but does not block the response. The bot picks up the new summary on the next turn after it saves.

**The boundary and failure recovery:**

The boundary is a sequence number that tracks which messages have been folded. Everything behind the boundary has been ingested into RAG and compressed into the summary. Everything ahead of it is still in the sliding window. When a fold succeeds, the boundary advances. If it fails, the boundary stays in place and the next fold retries the same batch.

RAG and summary are coupled by design: if RAG embedding fails, the summary does not fire. The ceiling is a trigger point, not a hard cap. If a fold fails, the window continues to grow until the next successful fold brings it back down. This prevents a gap where messages pass the boundary but are never indexed as vectors. Both summary and RAG succeed or neither does. Regardless, raw messages are never deleted; they remain in the database and can always be recovered.

The two axes together ensure there is no memory gap under normal operation. When messages leave the sliding window, RAG has already ingested them. If the summary is still processing in the background, the agent can still retrieve specific details through `search_archive`.