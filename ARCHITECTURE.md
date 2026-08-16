# Architecture

Basic Bot is a general-purpose AI assistant engine built on Anthropic's
Claude models. It provides chat, three-tier memory, and a tool system.
Agents are directories of configuration that the engine assembles at
startup — see [build-a-bot](https://github.com/CablepunkPress/build-a-bot)
for the agent template and
[basic-ui](https://github.com/CablepunkPress/basic-ui) for the reference
interface.

This document covers engine internals: how the pieces connect and why
they're built the way they are.

## Package Contents

```
basic_bot/
    __init__.py           # Package version (single source of truth)
    chat.py               # Chat loop — sends messages to Claude, executes tool calls
    config.py             # Configuration constants and env-var overrides
    embeddings.py         # EmbeddingProvider protocol, LocalEmbedder, VertexEmbedder
    factory.py            # create_runtime() — assembles a BotRuntime from an agent directory
    fold.py               # Fold lifecycle — should_fold, fold_rag, fold_summary
    memory.py             # Memory orchestration — context building, turn storage
    models.py             # Model definitions, capability flags, reply extraction
    rag.py                # RAG operations — turn pairing, vector storage, search
    runtime.py            # BotRuntime dataclass — the assembled agent
    store.py              # MessageStore protocol — the storage interface
    store_firestore.py    # Firestore implementation (cloud)
    store_sqlite.py       # SQLite implementation (local, default)
    summary.py            # Summarization — prompt construction, XML fencing
    tools.py              # Tool discovery and registry assembly
    instructions/
        capabilities.md   # Engine-owned system prompt section
    tool_belt/
        recall_message.py # Deterministic lookup by seq or date range
        search_archive.py # Semantic vector search over the RAG archive

scripts/                  # Maintenance utilities (reembed, rebuild_summary, backfills, tests)
```

## The Factory

`create_runtime(agent_path)` is the engine's entry point. It takes one
argument — the path to an agent directory — and derives everything else:

1. Reads `dashboard.json`; the `id` field becomes the agent's identity
2. Configures logging with the agent ID as prefix
3. Reads `persona.md`, substituting `{{ name }}` with the display name
4. Appends the engine's `capabilities.md` to form the system prompt
5. Builds the storage backend — SQLite at `~/.{id}/{id}.db` by default
6. Discovers tools — belt tools from the engine package, plugin tools
   from `agent_path/tools/`
7. Returns a `BotRuntime` dataclass holding all of it

The factory builds runtimes, not web applications. Flask, FastAPI, or
any other serving layer wraps the runtime in routes. This separation
keeps the engine framework-agnostic.

**Identity flows from `dashboard.json`.** The `id` names the database,
the dot-directory, the keyring service, and the log prefix. One
declaration, everything derived.

## Tool System

`tools.py` assembles the registry from two sources:

- **Belt tools** (`tool_belt/`) — ship with the engine, always loaded,
  discovered by package import. These are the agent's memory access:
  `search_archive` and `recall_message`.
- **Plugin tools** (`agent_path/tools/`) — discovered from the
  filesystem. Each subdirectory is a tool group. Files starting with
  `_` load as shared modules that sibling tools import directly
  (`from _auth import auth_headers`); every other `.py` file exposing
  a `TOOL` dict and a `handler` callable registers as a tool.

Plugin tools require no Python package, no `__init__.py`, no imports
in agent code. The directory is the installation — tool groups can be
copied between agents or installed from
[extend-a-bot](https://github.com/CablepunkPress/extend-a-bot).

Shared modules are removed from `sys.modules` after each group loads,
so same-named `_config.py` files in different groups never collide.

**Tool handler convention:** `handler(context, **tool_input)`. The
context dict carries the store and user ID. No base classes, no
decorators.

**Tool results** arrive as user-role messages in the Anthropic API.
This is the API's convention — it matters when parsing history.

The `TOOL_BOX_ENABLED` env var disables all plugin tools without
removing them.

## Memory System

Three tiers across two axes — persistence and timing — so no gap exists
between what the agent remembers and what it can retrieve.

**Persistence:**

- **Short-term (sliding window):** Recent messages verbatim. Holds
  between `WINDOW_FLOOR` (20) and `WINDOW_CEILING` (40) messages.
- **Intermediate-term (rolling summary):** Compressed narrative of
  everything past the window. Preserves durable facts cheaply.
- **Long-term (RAG archive):** Turn pairs embedded as vectors, stored
  permanently, searchable via `search_archive`.

**Timing during a fold:**

- Window updates every turn.
- RAG embeds synchronously at fold time — ingestion completes before
  the response returns.
- Summary generates asynchronously in the background; the agent picks
  it up next turn.

### Fold Lifecycle

The fold moves messages from the window into long-term storage when the
window hits `WINDOW_CEILING`. `fold.py` provides sync functions; the
caller picks the async strategy (`asyncio` for FastAPI, `threading` for
Flask).

**Ordering constraint:** RAG embeds first. If RAG fails, the summary
does not fire and the boundary does not advance — the next fold retries
the same batch. No message ever passes the boundary without a vector.
Raw messages are never deleted regardless.

The window oscillates between floor and ceiling in a sawtooth pattern.
Expected behavior, not a bug.

### Sequence Numbers

Every message gets a permanent sequence number, shown to the model as
`<!-- seq:N -->` HTML comments. HTML comments were chosen because Haiku
echoed bracket formats (`[#N]`) back into replies; comments read as
non-content metadata. The engine strips any echoed annotations at reply
extraction and at context build — one leaked echo stored in history
becomes in-context training signal, so stripping happens at both ends.

### Retrieval

- **`search_archive`** — semantic search by topic.
- **`recall_message`** — deterministic lookup: single seq (±2 context),
  seq range, or date range. Capped at 50 results, with the cap noted in
  responses so the agent knows more may exist.

Complementary modes: semantic search fails when you know the exact
turn; deterministic lookup fails when you only remember the gist.

### Summary Hardening

The summary is generated from raw transcripts, which can contain text
that looks like instructions. Three defenses: XML fencing around the
transcript, a summarizer system prompt that forbids following
transcript instructions, and a prefilled assistant turn
(`"Updated summary:"`) anchoring the output as summary continuation.

## Storage Abstraction

`MessageStore` is a typing protocol in `store.py` covering conversation
state, message CRUD, summaries, and vectors.

- **`SQLiteMessageStore`** — local default. WAL mode, four tables,
  vectors as `struct.pack` blobs, cosine similarity in Python. Stdlib
  only.
- **`FirestoreMessageStore`** — cloud deployments.

Selected by `STORAGE_BACKEND` env var (default `"sqlite"`). Imports are
deferred so local installs never pull in google.cloud.

**The database file is the agent's memory.** SQLite makes it portable —
migration is copying the file.

## Embedding Abstraction

`EmbeddingProvider` protocol in `embeddings.py`, one method:
`embed(texts, task)`.

- **`LocalEmbedder`** — HTTP to llama-server on localhost (default)
- **`VertexEmbedder`** — Vertex AI (cloud)

Selected by `EMBEDDING_PROVIDER` env var (default `"local"`).

**Qwen3 task prefixes:** queries get an instruct prefix, documents pass
through bare. Applied inside the embedder, not by callers.

**Embedding spaces are model-specific.** Different models produce
incompatible vector spaces even at identical dimensions. Swapping
models requires re-embedding everything — `scripts/reembed.py` is
permanent infrastructure for exactly this.

### Local Embedding Stack

Qwen3-Embedding-0.6B Q8_0 GGUF (639MB, 1024 dims, 32K context) served
by llama.cpp, CPU-only. Hosted at
`CablepunkPress/Qwen3-Embedding-0.6B-GGUF` on Hugging Face — a
self-conversion with the `tokenizer.ggml.add_eos_token` fix the
official release omits. Shared infrastructure lives at `~/.bountiful/`,
built once per machine, used by every agent.

## System Prompt Structure

Markdown headings that Haiku navigates by section name:

```
# PERSONA        (agent-authored, from persona.md)
# CAPABILITIES   (engine-owned, from instructions/capabilities.md)
# MODEL          (injected dynamically — model, effort, thinking)
# MEMORY         (injected dynamically — rolling summary, window position)
```

The heading structure is critical for smaller models: Haiku fails on
unstructured prompts but navigates named sections reliably.

## Model Support

Five models: Haiku 4.5 (default), Sonnet 4.6, Opus 4.6, 4.7, 4.8.
Thinking modes vary — extended thinking with budgets on Haiku, adaptive
thinking with effort levels on Sonnet/Opus.

Haiku without Deep Reasoning hallucinated tool calls in production.
Deep Reasoning is required for reliable autonomous agent use on Haiku.

**Thinking detection is truthful:** metadata records whether the
response actually contained a `ThinkingBlock`, not whether thinking was
requested.

## Configuration

All in `config.py` with env-var overrides:

| Constant | Default | Purpose |
|---|---|---|
| `WINDOW_FLOOR` | 20 | Minimum window size after a fold |
| `WINDOW_CEILING` | 40 | Message count that triggers a fold |
| `SUMMARY_MAX_TOKENS` | 2000 | Token budget for the rolling summary |
| `HISTORY_LIMIT` | 10 | Messages returned by `/history` |
| `EMBEDDING_PROVIDER` | `"local"` | `"local"` or `"vertex"` |
| `STORAGE_BACKEND` | `"sqlite"` | `"sqlite"` or `"firestore"` |
| `EMBEDDING_URL` | `http://localhost:11444` | llama-server address |
| `TOOL_BOX_ENABLED` | `true` | Kill switch for plugin tools |

## Scripts

Maintenance utilities in `scripts/`, run manually against an agent's
database:

- **`reembed.py`** — re-embed all vectors (required after model swap)
- **`rebuild_summary.py`** — regenerate the rolling summary
- **`backfill_rag.py`** / **`backfill_seq.py`** — legacy data migration
- **`test_fold.py`** / **`test_fold_lifecycle.py`** / **`test_rag.py`**
  — manual verification

## Design Rationale

Decisions that might look arbitrary but weren't:

- **Factory returns a runtime, not an app:** web frameworks are a
  deployment concern. The engine stays framework-agnostic; UIs wrap
  the runtime.
- **Filesystem tool loading:** plugin tools are user-owned files, like
  editor plugins. No package structure means drag-and-drop installation
  and portability between agents.
- **`~/.{id}/{id}.db` derived from dashboard.json:** one identity
  declaration drives everything; no per-agent storage configuration.
- **HTML comments for seq numbers:** Haiku echoed bracket formats;
  comments read as non-content metadata.
- **RAG blocks summary on failure:** prevents messages passing the
  boundary without vectors.
- **Haiku for summarization:** fast, cheap, sufficient for a
  constrained prompt. `summary.py` is extracted as its own module so a
  local model can replace it.
- **`search_archive` not `search_memory`:** Haiku pattern-matched
  "search your memory" to a literal tool name, causing unwanted calls.
- **`struct.pack` for vectors:** IEEE 754 binary beats JSON for
  high-dimensional floats in SQLite.
- **Brute-force cosine similarity:** at personal-assistant scale
  (thousands of vectors), Python brute force is fast enough and avoids
  an ANN dependency.
- **Qwen3-Embedding over EmbeddingGemma:** EmbeddingGemma's 2K context
  failed on real turn pairs; the cap applies per concatenated pair, so
  window tuning can't fix it. Qwen3 has 32K.
