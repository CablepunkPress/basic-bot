# Architecture

Basic Bot is a general-purpose AI assistant engine. It provides chat,
three-tier memory, a tool system, and a provider abstraction for
inference. Agents are directories of configuration that the engine
assembles at startup — see
[build-a-bot](https://github.com/CablepunkPress/build-a-bot) for the
agent template and
[basic-ui](https://github.com/CablepunkPress/basic-ui) for the
reference interface.

This document covers engine internals: how the pieces connect and why
they're built the way they are.

## Package Contents

```
basic_bot/
    __init__.py           # Package version (single source of truth)
    chat.py               # Chat loop — provider-agnostic orchestration, tool execution
    config.py             # Configuration constants and env-var overrides
    embeddings.py         # EmbeddingProvider protocol, LocalEmbedder, VertexEmbedder
    factory.py            # create_runtime() — assembles a BotRuntime from an agent directory
    fold.py               # Fold lifecycle — should_fold, fold_rag, fold_summary
    memory.py             # Memory orchestration — context building, turn storage
    rag.py                # RAG operations — turn pairing, vector storage, search
    runtime.py            # BotRuntime dataclass — the assembled agent
    secrets_env.py        # Load agent API keys from keyring into environment
    store.py              # MessageStore protocol — the storage interface
    store_firestore.py    # Firestore implementation (cloud)
    store_sqlite.py       # SQLite implementation (local, default)
    summary.py            # Summarization — provider-based, prompt construction
    tools.py              # Tool discovery and registry assembly
    
    infra/                # Shared Bountiful infrastructure at ~/.bountiful/
        __init__.py
        __main__.py       # Entry point for `python -m basic_bot.infra`
        llamacpp.py       # Clone and compile llama.cpp from source
        models.py         # Download and manage GGUF models
        server.py         # llama-server lifecycle — start, health check, stop
    
    instructions/
        capabilities.md   # Engine-owned system prompt section

    providers/            # Inference provider abstraction
        __init__.py
        protocol.py       # InferenceProvider protocol, ChatResponse, ModelInfo, ToolCall
        claude.py         # ClaudeProvider — wraps Anthropic SDK

    setup/                # Agent setup utilities (called by build-a-bot shims)
        __init__.py
        secrets.py        # Interactive keyring setup for agent API keys
        tools.py          # Install tool groups from extend-a-bot

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
7. Instantiates the inference provider
8. Returns a `BotRuntime` dataclass holding all of it

The factory builds runtimes, not web applications. Flask, FastAPI, or
any other serving layer wraps the runtime in routes. This separation
keeps the engine framework-agnostic.

**Identity flows from `dashboard.json`.** The `id` names the database,
the dot-directory, the keyring service, and the log prefix. One
declaration, everything derived.

## Provider Abstraction

The engine never imports provider-specific SDKs. All inference goes
through `InferenceProvider`, a protocol defined in
`providers/protocol.py`. Translation between the engine's internal
message format and any API happens inside the provider.

### Three-Level Model Taxonomy

Every model carries metadata in a `ModelInfo` dataclass:

- **Host** — `"api"` or `"local"` (where is it running?)
- **Provider** — `"Anthropic"`, `"Google"`, `"Alibaba"` (who provides it?)
- **Family** — `"Claude"`, `"Gemini"`, `"Qwen"` (which model family?)

Plus: `id`, `display_name`, `rank` (display ordering), `effort_levels`,
and `thinking_type`. The UI can slice models any way it wants — by host,
provider, family, or a flat ranked list — without restructuring code.

### Provider Contract

```python
class InferenceProvider(Protocol):
    def get_models(self) -> dict[str, ModelInfo]: ...
    def get_default_model(self) -> str: ...
    def get_fallback_model(self) -> str: ...
    def chat(self, *, messages, system, tools, model_id, effort, thinking) -> ChatResponse: ...
```

Providers return `ChatResponse` (text, model_used, thinking, tool_calls)
and `ToolCall` (name, input, id) — engine-owned types the chat loop
works with directly.

### Current Provider

`ClaudeProvider` in `providers/claude.py` wraps the Anthropic SDK.
It owns the Claude model catalog, token limits, API kwargs construction,
and response parsing. Adding a new provider means adding one file that
implements the same protocol.

File naming follows SDK constraints: `claude.py` not `anthropic.py`,
because a file named `anthropic.py` would shadow the pip package.

### Chat Orchestration

`chat.py` is provider-agnostic. It loads memory, builds the system
prompt, calls `provider.chat()`, and handles the tool execution loop.
The tool loop appends assistant messages with tool calls and tool
results in the engine's internal format; the provider translates at the
boundary.

Summarization also goes through the provider. `summary.py` calls
`provider.chat()` with the fallback model — no direct API calls exist
outside the provider.

## Infrastructure Package

`infra/` manages the shared Bountiful infrastructure at `~/.bountiful/`.
This is not agent-specific — it's built once per machine and shared by
every agent.

- **`llamacpp.py`** — clones and compiles llama.cpp from source (CPU).
  Includes prerequisite checks for git, cmake, and a C++ compiler.
- **`models.py`** — downloads GGUF models from Hugging Face. Currently
  the Qwen3-Embedding-0.6B Q8_0 model.
- **`server.py`** — llama-server lifecycle: start if not running, health
  check, stop on teardown. Used by basic-ui's `launch.py` to manage the
  embedding server.
- **`__main__.py`** — entry point for `python -m basic_bot.infra`,
  called by build-a-bot's `build.py` after pip install completes.

Build-a-bot's `build.py` delegates to `python -m basic_bot.infra` via
subprocess after creating the venv. This keeps build tool requirements
(cmake, compiler) in the engine where they belong — future GPU builds,
hardware detection, and model catalogs evolve here without touching the
template.

## Setup Package

`setup/` contains utilities called by build-a-bot's shim scripts:

- **`secrets.py`** — interactive keyring setup. Scans `dashboard.json`
  for agent identity and `tools/*/tool.json` for secret declarations.
  Called by `add_secrets.py`.
- **`tools.py`** — installs tool groups from extend-a-bot. Downloads
  the tarball, extracts the group, installs pip dependencies. Handles
  install, update (preserving config files), and list commands. Called
  by `add_tools.py`.

These live in the engine so pip updates deliver new behavior. The
build-a-bot scripts are stable shims that never need updating.

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

**Tool results** are returned to the provider in the engine's internal
format. The provider translates them to the API's expected format
(e.g., Anthropic expects tool results as user-role messages with
tool_result content blocks).

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
- Summary generates asynchronously in the background via the inference
  provider; the agent picks it up next turn.

### Fold Lifecycle

The fold moves messages from the window into long-term storage when the
window hits `WINDOW_CEILING`. `fold.py` provides sync functions; the
caller picks the async strategy (`asyncio` for FastAPI, `threading` for
Flask). The provider is passed through from the runtime so summarization
uses the same inference backend as chat.

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
extraction (inside the provider) and at context build — one leaked echo
stored in history becomes in-context training signal, so stripping
happens at both ends.

### Retrieval

- **`search_archive`** — semantic search by topic.
- **`recall_message`** — deterministic lookup: single seq (±2 context),
  seq range, or date range. Capped at 50 results, with the cap noted in
  responses so the agent knows more may exist.

Complementary modes: semantic search fails when you know the exact
turn; deterministic lookup fails when you only remember the gist.

### Summary Hardening

The summary is generated from raw transcripts, which can contain text
that looks like instructions. Two defenses: XML fencing around the
transcript and a summarizer system prompt that forbids following
transcript instructions.

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

The MODEL section is now provider-aware: it reads from `ModelInfo.family`
and `ModelInfo.display_name` rather than hardcoding "Claude."

## Configuration

All in `config.py` with env-var overrides:

| Constant | Default | Purpose |
|---|---|---|
| `WINDOW_FLOOR` | 20 | Minimum window size after a fold |
| `WINDOW_CEILING` | 40 | Message count that triggers a fold |
| `SUMMARY_MAX_TOKENS` | 2000 | Token budget for the rolling summary |
| `SUMMARY_MIN_CHARS` | 40 | Minimum viable summary length |
| `HISTORY_LIMIT` | 10 | Messages returned by `/history` |
| `EMBEDDING_PROVIDER` | `"local"` | `"local"` or `"vertex"` |
| `STORAGE_BACKEND` | `"sqlite"` | `"sqlite"` or `"firestore"` |
| `EMBEDDING_URL` | `http://localhost:11444` | llama-server address |
| `TOOL_BOX_ENABLED` | `true` | Kill switch for plugin tools |

Agent-level settings (Flask port, tool repo overrides) live in
`config.json` in the agent directory. The engine provides defaults for
any missing field.

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
- **Provider abstraction at the boundary:** the engine works with its
  own message format. Providers translate at the edge. Adding a
  provider means adding one file — no engine changes.
- **Three-level model taxonomy (host/provider/model):** supports
  mixed local+API deployments where a user has Qwen models locally
  and Claude models via API, all in one registry.
- **Rank on ModelInfo:** controls display order in the UI independent
  of alphabetical sort. Works across providers.
- **Filesystem tool loading:** plugin tools are user-owned files, like
  editor plugins. No package structure means drag-and-drop installation
  and portability between agents.
- **`~/.{id}/{id}.db` derived from dashboard.json:** one identity
  declaration drives everything; no per-agent storage configuration.
- **HTML comments for seq numbers:** Haiku echoed bracket formats;
  comments read as non-content metadata.
- **RAG blocks summary on failure:** prevents messages passing the
  boundary without vectors.
- **Summarization through the provider:** uses the same inference
  backend as chat. When a local model replaces the API, summarization
  switches automatically.
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
- **Setup and infra in the engine package:** build-a-bot scripts are
  stable shims. Logic lives behind pip so engine updates deliver new
  behavior to every agent.
- **`infra/` as a package:** GPU builds, hardware detection, model
  catalogs, and multi-server management all evolve here without
  touching the template.
- **`claude.py` not `anthropic.py`:** a file named `anthropic.py`
  shadows the pip package on import.
