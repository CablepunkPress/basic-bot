# TODO

Covers basic-bot (engine), plus basic-ui, build-a-bot, extend-a-bot.

## v0.7.0 — Open Source Release

### README
User-facing documentation: what Basic Bot is, how to install, how to
run, how to build a downstream bot. Feature-focused, not architecture-
focused (ARCHITECTURE.md covers internals).

This should be done in conjunction with Bountiful GUI. Basic Bot is the engine; Bountiful is the UI. Basic Bot `web\` (UI) and `build.py` and `run.py` will be be deprecated in favor of Bountiful. Basic Bot is "House Agent" of Bountiful. Deferred and revised: see v0.8.0

v0.6.0 README update is engine-focused and link to build-a-bot new repo for end users

### License
MIT. Confirm license file is present and `pyproject.toml` declares it.


## v0.8.0 — Local Deployment, Local Models

### Deployment
Local vs Cloud

Local deployment: build-a-bot
Cloud deployment: build-a-bot-cloud

### Inference provider selection for local deployment
Add inference provider switch: API or Local

Local first. Same provider. Idea is to run on minimal hardware AND not need to use add_secrets.py. Target: Nvidia 3060 12GB + 32GB. Future target: build for desktop agent computers (Nvidia DGX Spark, AMD Ryzen AI Halo)
[local] = local deployment, local models
Embeddings:
Summarization:
Inference: 
Inference (coding): GPT-OSS-20B

[hybrid] = local deployment, cloud models
API second. Same provider. Idea is to not run any model locally and rely entirely on API calls.
Embeddings:
Summarization:
Inference:
Inference (optional): Claude API, etc.

Currently, embeddings are Qwen3-Embedding running on CPU, summarization is Haiku 4.5 API, and inference is Claude API. Default should be one provider of the full stack (embeddings, summarization, inference) for each deployment scenario. One API key for the full stack, not 2-3. Add additional keys for inference choice. Embeddings and summarization needs to be opionated.


### Chat abstraction

chat_with_model (currently chat_with_claude)

messages: list of {role, content}

response: {text, tool_calls, thinking, model_used}

tool_call: {name, input, id}

tool_result: {id, content}

### Build-A-Bot Shims
Script logic into engine package; template scripts become stable shims
`run.py` calls `basic_bot.launch(ROOT)`
`add_secrets.py` calls `basic_bot.setup_secrets(ROOT)`

### Local summarization
Replace Haiku API call for summary generation with a local model via
llama.cpp. The infrastructure is already in place. `summary.py` is
extracted as its own module for exactly this swap.

Summary model needs replaced with local model. Possible candidate: Qwen3 14B Q4_K_M
For folds on local on limited hardware, summary needs switched back to sync: 
both embeddings and summary need to finish before next turn. Have to accept the delay.
`FOLD_MODE = "sync"` or `FOLD_MODE = "async"` in config.

Bonus: UI should indicate a fold. Opportune time to integrate planned "what's happening" window in UI.


## v0.9.0 — Plugin Tools

### Tool Box --> extend-a-bot
Web search tool
GitHub repo tools (completed)
central.db read-only tool 

### Write tool confirmation gate
Tools with `requires_confirmation: True` in their definition return a
preview instead of executing. The chat handler checks this flag before
dispatching. The UI presents the preview with approve/decline. This is
an engine-level feature — every downstream bot inherits it.

Read tools execute immediately. Write tools go through the gate. The
classification lives on the tool definition, not in a separate config.

### Tool search meta-tool
When the tool count exceeds the reliable range (~20+), a `search_tools`
meta-tool finds the right tool by description and loads its full
definition on demand. Same pattern Claude Code uses internally. Build
when the tool count actually causes selection problems, not before.


## Unscheduled

### API key expiration handling
Graceful error message when the Anthropic API key is expired or
invalid, with a hint to run `build.py` again. Currently the error
surfaces as an unhandled API exception.

### Context window bloat without embeddings
When llama-server is unavailable, RAG embedding fails and the window
grows past the ceiling indefinitely. Expected behavior, but worth
documenting for users and potentially surfacing a warning in the UI.

### extract_reply edge case
Sonnet can return thinking-only responses with no `TextBlock`. Current
Haiku fallback is correct recovery but the error log is noisy. Clean
up the logging path.

### Proxy timeout
Raise `/api/chat` timeout from 30s to 120s. Sonnet with high effort,
Deep Reasoning, and tools exceeds 30s regularly.

### Repeated fold-failure logging
First failure logs the full traceback. Subsequent retries of the same
batch should log a single-line message until the fold succeeds or a
new error occurs.

### Model order
Currently, models display in alphabetical order. Should be in ascending order (Haiku then Sonnet then Opus). When selecting, should always default to low effort level as well.

### Scripts
Scripts needed reviewed for relevancy and updated.

### Streaming (SSE)
Deferred. Bot → proxy → frontend SSE chain. Stream only final text,
not thinking blocks.

### Attachments
Cablepunk Bot needs file attachments for some tool
workflows. Basic Bot doesn't need them but the engine should support
them for downstream bots.

### Cloud Deployment
Resume.

### Firestore-to-SQLite migration script
Permanent infrastructure in `scripts/`. Reads messages, state, and
summaries from a Firestore collection and writes them into a SQLite
database. Skips vectors (run `reembed.py` after). Preserves original
sequence numbers and timestamps. Maps user IDs as needed.

Currently exists in Cablepunk Bot's repo (Bountiful prototype) as a one-off. Worth
generalizing and moving to Basic Bot for any downstream bot migrating
off Firestore.
