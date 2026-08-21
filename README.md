# Basic Bot

A general-purpose AI assistant engine.

This is the **engine repository**, the pip-installable core that provides
chat, memory, and tools. It is not meant to be run directly.

**To create and run your own agent, start at
[build-a-bot](https://github.com/CablepunkPress/build-a-bot).**

## The Ecosystem

| Repo | What it is |
|------|------------|
| [basic-bot](https://github.com/CablepunkPress/basic-bot) | The engine (this repo). Chat loop, memory, tool system. |
| [basic-ui](https://github.com/CablepunkPress/basic-ui) | Reference Flask chat interface. |
| [build-a-bot](https://github.com/CablepunkPress/build-a-bot) | Template for creating your own local agent. |
| [extend-a-bot](https://github.com/CablepunkPress/extend-a-bot) | Drop-in plugin tool groups. |

## What the Engine Provides

- **Chat loop** — sends messages to assistant, executes tool calls, handles
  fallbacks. Currently, supports Haiku, Sonnet, and Opus models with per-model
  thinking modes.
- **Three-tier memory** — a sliding window of recent messages, a rolling
  summary of older conversation, and a permanent vector archive searchable
  by the agent itself. Conversations are stored locally in SQLite.
- **Tool system** — two built-in tools (`search_archive`,
  `recall_message`) and filesystem-based discovery of plugin tools from
  an agent's `tools/` directory.
- **Storage and embedding abstractions** — SQLite or Firestore for
  storage, local llama.cpp or Vertex AI for embeddings, selected by
  environment variable.

## Installing as a Dependency

Agents depend on the engine in their `pyproject.toml`:

```toml
dependencies = [
    "basic-bot[local] @ git+https://github.com/CablepunkPress/basic-bot.git@v0.6.0",
]
```

Extras:

- `[local]` — local deployment (keyring for secrets)
- `[cloud]` — cloud deployment (FastAPI, Firestore, Vertex AI)
- `[dev]` — everything, for development environments

## Usage

The engine's entry point is the factory:

```python
from pathlib import Path
from basic_bot.factory import create_runtime

runtime = create_runtime(Path("/path/to/agent"))
```

The agent directory contains `dashboard.json` (identity), `persona.md`
(personality), and optionally `tools/` (plugin tool groups). The factory
returns a `BotRuntime` holding the store, persona, and tool registry.
A UI or server layer wraps the runtime in routes —
[basic-ui](https://github.com/CablepunkPress/basic-ui) is the reference
implementation.

## Memory System

Basic Bot uses a three-tier memory architecture across two axes:
persistence and timing.

- **Short-term (sliding window):** The most recent messages verbatim,
  between 20 and 40 messages before a fold compresses the oldest batch.
- **Intermediate-term (rolling summary):** A compressed narrative of
  everything that has left the window. Preserves durable facts without
  the token cost of raw messages.
- **Long-term (RAG archive):** Every turn pair embedded as vectors and
  stored permanently. The summary knows the facts; RAG has the receipts.

When the window fills, a fold moves the oldest messages into the archive
and summary. RAG embeds synchronously; the summary generates in the
background. If embedding fails, the fold retries the same batch — no
message ever passes the boundary without being indexed. Raw messages
are never deleted.

See [ARCHITECTURE.md](ARCHITECTURE.md) for internals and design
rationale.

## License

MIT
