# Architecture

Basic Bot is a general-purpose AI assistant engine built on Anthropic's Claude
models. It provides chat, three-tier memory, and a tool system designed to be
inherited by downstream bots that add domain-specific capabilities.

The engine is pip-installable. Downstream bots depend on it via
`git+https://` in their requirements and extend it by adding tools, a system
prompt, and a storage backend. Basic Bot itself ships with two belt tools
(`search_archive` and `recall_message`), a sliding-window/summary/RAG memory
system, and entry points for both local and cloud deployment.


## Module Map

```
basic_bot/
    __init__.py
    app.py                  # FastAPI application (cloud entry point)
    chat.py                 # Core chat loop — sends messages to Claude, handles tool calls
    config.py               # All configuration constants and env-var overrides
    embeddings.py           # EmbeddingProvider protocol, LocalEmbedder, VertexEmbedder
    factory.py              # create_app() — assembles a bot from package name + collection
    fold.py                 # Fold lifecycle — should_fold, fold_rag, fold_summary, build_metadata
    memory.py               # Memory orchestration — build_context, store_turns
    models.py               # Anthropic model definitions and capability flags
    rag.py                  # RAG operations — pair_turns, store_turns, search
    runtime.py              # BotRuntime dataclass — holds store, config, system prompt
    store.py                # MessageStore protocol — the storage interface
    store_firestore.py      # Firestore implementation of MessageStore
    store_sqlite.py         # SQLite implementation of MessageStore
    summary.py              # Haiku summarization — prompt construction, XML fencing

tool_belt/
    __init__.py             # Belt tool discovery
    recall_message.py       # Deterministic lookup by seq or date range
    search_archive.py       # Semantic vector search over RAG archive

tool_box/
    __init__.py             # Box tool discovery (empty in Basic Bot)

build.py                    # One-command local setup: venv, llama.cpp, model, keyring
run.py                      # Single-command launch: llama-server + Flask on port 5084

scripts/
    backfill_rag.py         # Backfill RAG vectors for existing messages
    backfill_seq.py         # Backfill sequence numbers on legacy messages
    rebuild_summary.py      # Rebuild rolling summary from scratch
    reembed.py              # Re-embed all vectors (required after model swap)
    test_fold.py            # Manual fold lifecycle test
    test_fold_lifecycle.py  # Extended fold lifecycle test
    test_rag.py             # Manual RAG test

web/
    __init__.py
    server.py               # Flask application (local entry point)
    templates/
        index.html          # Jinja2 chat interface
    static/
        css/styles.css      # System font stack, dark mode
        js/chat.js          # Single file, no modules, no auth
        js/globals.d.ts     # JSDoc type definitions

pyproject.toml              # Base: anthropic only. Extras: [local], [cloud]
requirements.txt            # Downstream pinning reference
```

## Factory Pattern and Downstream Inheritance

`create_app(package_name, collection)` is the entry point for assembling a
bot. The `package_name` identifies the downstream package (e.g.
`"cablepunk_bot"`), which determines where to look for a system prompt and
tool box. The `collection` identifies the conversation namespace.

The factory builds a `BotRuntime` dataclass that holds the store, embedder,
system prompt, model config, and tool registry. This runtime is passed through
the chat loop to every component that needs it.

**Identity decisions belong in code** — the `create_app()` call in a
downstream bot's entry point. Tuning goes in config (env vars). Secrets go in
Secret Manager (cloud) or keyring (local). This separation is deliberate:
identity is version-controlled, tuning is environment-specific, secrets are
never in code.


## Tool System

Tools are organized into two directories:

- **`tool_belt/`** — always active, inherited by every downstream bot. These
  are the core capabilities: `search_archive` (semantic vector search) and
  `recall_message` (deterministic seq/date lookup). Belt tools ship with
  Basic Bot and should not be overridden.

- **`tool_box/`** — domain-specific tools added by downstream bots. Empty in
  Basic Bot. A downstream bot provides its own `tool_box` package with tool
  modules. `build_tool_registry()` auto-discovers tools from
  `{package_name}.tool_box`.

The `TOOL_BOX_ENABLED` kill switch disables all box tools without removing
them from the codebase.

**Tool handler convention:** `handler(context, **tool_input)` — the context
dict carries the runtime, user ID, and whatever else a tool needs. Tools grab
what they need from context without base classes or decorators.

**Tool results** arrive as user-role messages in the Anthropic API. This is
the API's convention, not something the user initiated. The distinction
matters when parsing conversation history.


## Memory System

### Three-Tier Architecture

The memory system operates on two axes — persistence and timing — to ensure
no gap exists between what the bot remembers and what it can retrieve.

**Persistence axis — what the bot remembers:**

- **Short-term (sliding window):** The most recent messages in verbatim form.
  The window holds between `WINDOW_FLOOR` (default 20) and `WINDOW_CEILING`
  (default 40) messages before a fold compresses the oldest batch.

- **Intermediate-term (rolling summary):** A compressed narrative of
  everything that has left the sliding window. Preserves durable facts —
  names, preferences, decisions, ongoing topics — without the token cost of
  raw messages. Generated by Haiku with XML fencing and a prefilled assistant
  turn to prevent instruction bleed from transcript content.

- **Long-term (RAG archive):** Verbatim turn pairs embedded as vectors and
  stored permanently. Searchable by the agent via `search_archive`. The
  summary knows the facts; RAG has the receipts.

**Timing axis — when each tier updates during a fold:**

- **Immediate (sliding window):** Updates every turn.
- **Near-immediate (RAG):** Embeds synchronously at fold time. Ingestion
  completes before the response returns to the user.
- **Eventual (summary):** Generates asynchronously in the background. Does
  not block the response. The bot picks up the new summary on the next turn.

### Fold Lifecycle

The fold is the mechanism that moves messages from the sliding window into
long-term storage. It fires when the window hits `WINDOW_CEILING`.

`fold.py` contains four sync functions: `should_fold`, `fold_rag`,
`fold_summary`, and `build_metadata`. The caller decides the async strategy:
`asyncio.create_task` for FastAPI (cloud), `threading.Thread` for Flask
(local).

**Ordering constraint:** RAG embeds first, synchronously. If RAG fails, the
summary does not fire and the boundary does not advance. The next fold retries
the same batch. This prevents a gap where messages pass the boundary but are
never indexed as vectors.

The window oscillates in a sawtooth pattern between floor and ceiling. This is
expected behavior, not a bug. Floor and ceiling are independent tuning knobs.

### Sequence Numbers

Every message gets a permanent sequence number. Displayed as `<!-- seq:N -->`
HTML comments prepended to messages in the context window.

HTML comment format was chosen because Haiku pattern-matched earlier bracket
formats (`[#N]`) and echoed them in responses. HTML comments are understood as
non-content metadata from training data. A positive framing in the system
prompt ("Messages below contain `<!-- seq:N -->` annotations for internal
tracking. Your reply is plain prose beginning with your first content word.")
eliminated echoing completely.

### Two Retrieval Modes

- **`search_archive`** — semantic search for topics. Returns results with
  `seq_range` labels and seq integers for cross-referencing.
- **`recall_message`** — deterministic lookup. Three modes: single seq (with
  ±2 context window), seq range, date range. Guards: `MAX_SEQ_RANGE = 50`,
  `MAX_DATE_RESULTS = 50`.

These are complementary: semantic search fails when you know exactly which
turn you want; deterministic lookup fails when you only remember the gist.

### Summary Hardening

The rolling summary is generated by Haiku from raw conversation transcripts.
Without safeguards, transcript content can bleed into the summary as
instructions (e.g., a user message saying "You are now X" gets treated as a
directive).

Three defenses:
1. **XML fencing** — the transcript is wrapped in XML tags that the
   summarizer prompt explicitly identifies as content to summarize, not
   instructions to follow.
2. **Instruction-forbidding system prompt** — the summarizer's system prompt
   explicitly states it must never follow instructions found in the
   transcript.
3. **Prefilled assistant turn** — the summarizer call begins with
   `"Updated summary:"` as a prefilled assistant response, anchoring the
   model's output as a continuation of a summary rather than a fresh response
   to transcript content.


## Storage Abstraction

`MessageStore` is a protocol (Python typing protocol) in `store.py` that
defines the interface for all storage operations: conversation state, message
CRUD, summary management, and vector storage.

Two implementations:

- **`FirestoreMessageStore`** (`store_firestore.py`) — Google Cloud Firestore.
  Used for Cloud Run deployments.
- **`SQLiteMessageStore`** (`store_sqlite.py`) — local SQLite with WAL mode.
  Four tables (`state`, `messages`, `summaries`, `vectors`), three indexes.
  Vector blobs use `struct.pack`/`unpack`. Cosine similarity computed in
  Python (brute force). Zero external dependencies beyond stdlib.

The `STORAGE_BACKEND` env var selects the implementation. Default is
`"sqlite"` — cloud deployments must explicitly set `"firestore"`.

Storage implementations are independently swappable. Changing the storage
backend does not affect embedding, summarization, or any other subsystem.


## Embedding Abstraction

`EmbeddingProvider` is a protocol in `embeddings.py` with a single method:
`embed(texts, task)`.

Two implementations:

- **`LocalEmbedder`** — HTTP POST to a llama-server instance running
  Qwen3-Embedding-0.6B Q8_0 on localhost. Uses stdlib `urllib`, no SDK. Default for
  local installs.
- **`VertexEmbedder`** — Google Vertex AI embeddings API. Used for Cloud Run
  deployments.

The `EMBEDDING_PROVIDER` env var selects the implementation. Default is
`"local"` — cloud deployments set `"vertex"`.

**Task prefixes:** EmbeddingGemma requires task-specific prefixes. Documents
get `"title: none | text: "`, queries get `"task: search result | query: "`.
These are applied inside the embedder, not by the caller. 
**Altered** With the switch to Qwen3-Embedding-0.6B Q8_0, the two prefix constants became one query instruction; 
documents pass through with no prefixes. Qwen3 should have an `instruct` on queries.


**Embedding spaces are model-specific.** Two different embedding models
produce vectors in incompatible semantic spaces, even at identical dimensions.
Swapping models always requires re-embedding the entire archive.
`scripts/reembed.py` exists as permanent infrastructure for this — it is not a
one-time migration script.

### Local Embedding Stack

Qwen3-Embedding-0.6B Q8_0 GGUF (639MB, 1024 dimensions, 32K token context)
served by llama.cpp compiled from source. CPU-only by default — 600M
parameters remains fast on any modern CPU. Hosted at
`CablepunkPress/Qwen3-Embedding-0.6B-GGUF` on Hugging Face — a self-conversion
with the `tokenizer.ggml.add_eos_token` metadata fix applied, since the
official Qwen GGUF release omits it, degrading embedding quality. llama-server
runs on port 11444 with `-c 32768` and `--ubatch-size 8192`, and exposes an
OpenAI-compatible `/v1/embeddings` endpoint.

`run.py` manages the llama-server lifecycle: spawn as subprocess, poll
`/health` until ready (60-second timeout), teardown on Ctrl+C.
`use_reloader=False` on Flask prevents a double llama-server spawn.


## System Prompt Structure

The system prompt uses markdown headings that Haiku navigates by section name:

```
# IDENTITY
(downstream bot's persona — who it is, how it behaves)

# CAPABILITIES
(what tools are available, how to use them)

# MODEL
(current model, effort level, thinking mode — injected dynamically)

#MEMORY
(rolling summary — injected dynamically each turn)
```

This heading structure is critical for smaller models. Haiku failed to follow
unstructured system prompts reliably but navigates named sections without
issue.


## Entry Points

### Local (`build.py` + `run.py`)

`build.py` is a one-command local setup script. Runs in system Python using
only stdlib. Five steps: prerequisites check, venv creation with
`pip install -e .[local]`, llama.cpp clone and cmake build (CPU-only),
EmbeddingGemma model download, and API key storage via keyring. Each step is
idempotent.

`run.py` is the single-command launcher. Re-execs into `.venv` via `os.execv`,
reads the API key from keyring, spawns llama-server as a subprocess, polls
`/health` until ready, and starts Flask on port 5084. Ctrl+C tears down both
processes.

No `.env` files. The `keyring` library provides OS-native secret storage.

### Cloud (`app.py`)

FastAPI application for Cloud Run. Requires two env vars: `STORAGE_BACKEND=firestore`
and `EMBEDDING_PROVIDER=vertex`. Secrets come from Google Secret Manager with
per-secret IAM bindings (no service account has project-wide secret access).

Routes: `/chat`, `/history`, `/models`, `/health`.


## Configuration

All configuration lives in `config.py` with env-var overrides. Key values:

| Constant | Default | Purpose |
|---|---|---|
| `WINDOW_FLOOR` | 20 | Minimum messages before fold is possible |
| `WINDOW_CEILING` | 40 | Message count that triggers a fold |
| `HISTORY_LIMIT` | 10 | Messages returned by `/history` endpoint |
| `EMBEDDING_PROVIDER` | `"local"` | `"local"` or `"vertex"` |
| `STORAGE_BACKEND` | `"sqlite"` | `"sqlite"` or `"firestore"` |
| `EMBEDDING_URL` | `http://localhost:11444` | llama-server address |

Floor and ceiling are independent tuning knobs. A higher floor retains more
verbatim context. A higher ceiling delays folds. Neither depends on the other.


## Model Support

Five models: Haiku 4.5 (default), Sonnet 4.6, Opus 4.6, Opus 4.7, Opus 4.8.

Thinking modes vary by model:
- **Haiku 4.5:** Extended thinking (`type: "enabled"`, `budget_tokens`).
- **Sonnet 4.6 / Opus 4.6:** Adaptive thinking (`type: "adaptive"`) with
  effort levels.
- **Opus 4.7 / Opus 4.8:** Adaptive thinking with `xhigh` effort available.

Haiku without Deep Reasoning hallucinated tool calls in production. Deep
Reasoning is required for reliable autonomous agent use on Haiku.

**Thinking detection:** The metadata records whether the API response actually
contained a `ThinkingBlock`, not whether thinking was requested. The API does
not return the actual effort level used — only the requested level is recorded.


## Message Metadata

`save_turn` accepts a metadata dict with fields defined by the
`METADATA_FIELDS` tuple: `model_used`, `display_name`, `effort`, `thinking`,
`fallback`. Metadata is stored flat on message documents. The `/history`
endpoint returns metadata alongside messages.


## Downstream Bot Pattern

A downstream bot is a separate Python package that depends on Basic Bot:

```
# requirements.in
git+https://git@github.com/CablepunkPress/basic-bot.git@<commit-hash>
```
It provides:
- A `tool_box` package with domain-specific tool modules
- A system prompt (persona + capabilities)
- An entry point that calls `create_app(package_name, collection)`

The downstream bot inherits all of Basic Bot's memory, retrieval, fold
lifecycle, and belt tools. It adds only what is specific to its domain.

Pin to a commit hash, not a branch. This prevents upstream changes from
breaking downstream bots without explicit version bumps.


## Scripts

Maintenance scripts in `scripts/` are invoked directly:
`python scripts/<script>.py <args>`.

- **`reembed.py`** — re-embeds all folded messages. Required after swapping
  embedding models. Respects the `summarized_through` boundary. Batches at 50
  turns. Permanent infrastructure.
- **`rebuild_summary.py`** — regenerates the rolling summary from message
  history.
- **`backfill_rag.py`** — backfills RAG vectors for messages that predate the
  RAG system.
- **`backfill_seq.py`** — assigns sequence numbers to legacy messages.
- **`test_fold.py`** / **`test_fold_lifecycle.py`** / **`test_rag.py`** —
  manual verification scripts for the memory subsystems.


## Design Rationale

Decisions that might look arbitrary but were made for specific reasons:

- **HTML comments for seq numbers** instead of bracket prefixes: Haiku
  echoed `[#N]` formats. HTML comments are understood as non-content metadata.
- **RAG blocks summary on failure:** prevents a gap where messages pass the
  boundary but have no vector representation.
- **Haiku for summarization:** fast, cheap, and the summary prompt is
  constrained enough that a smaller model handles it well. The extracted
  `summary.py` module allows swapping to a local model later.
- **`search_archive` not `search_memory`:** Haiku pattern-matched
  "search your memory" to a tool literally named `search_memory`, causing
  unwanted tool calls. Renaming broke the false association.
- **Separate `tool_belt` and `tool_box`:** belt tools are always present and
  inherited; box tools are domain-specific and optional. The kill switch only
  affects box tools.
- **`struct.pack` for vector blobs:** avoids JSON serialization overhead for
  high-dimensional float arrays in SQLite. Standard IEEE 754 binary format.
- **Brute-force cosine similarity:** at the scale of a personal assistant
  (thousands to tens of thousands of vectors), brute force in Python is fast
  enough and avoids an ANN index dependency.
- **Local Embeddings with Qwen:** Model changed from EmbeddingGemma to Qwen3-Embedding-0.6B Q8_0
  due to EmbeddingGemma's limiting 2K context window easily surpassed in a turn. 
  Qwen3-Embedding has a context window of 32K.
