# build-a-bot

Name and customize your own Basic Bot Engine agent.

## Installation

### Quick Start

Name your agent what you wish. The example here is for an agent named "Alice".

```bash
git clone https://github.com/CablepunkPress/build-a-bot.git alice
cd alice
python build.py
python add_secrets.py
python run.py
```
`build.py` names your agent from the directory, sets up the
environment, and builds the shared inference infrastructure. Your agent
is running at `http://localhost:11555`.

### Quick Start with Plugin Tools

```bash
git clone https://github.com/CablepunkPress/build-a-bot.git alice
cd alice
python build.py
python add_tools.py github
# edit tools/github/_config.py with your values
python add_secrets.py
python run.py
```

Use `python add_tools.py --list` to see available tool groups. Currently, the only tool group available is for a GitHub repo app.

## What Your Agent Includes

Every agent ships with a conversation memory system and two built-in tools:

- **search_archive** — semantic search over past conversations
- **recall_message** — look up specific messages by number or date

Conversations are stored locally in SQLite at `~/.{agent-id}/{agent-id}.db`.

Shared inference infrastructure (embedding model and llama.cpp) lives at
`~/.bountiful/` and is built once, shared by all Basic Bot agents on the machine.

## Customization

### Persona

Edit `persona.md` to change how your agent behaves. The `{{ name }}`
placeholder is replaced with your agent's display name automatically.

Default:

```markdown
# PERSONA

You are {{ name }}, a helpful and friendly general-purpose AI assistant. 
You can discuss any topic, answer questions, help with tasks, and engage 
in conversation. Be concise, clear, and helpful. Keep responses 
conversational and friendly.
```

Sample alternative:

```markdown
# PERSONA

You are {{ name }}, a curious and thoughtful AI assistant whose
personality draws from Lewis Carroll's Alice. You approach every
topic with genuine curiosity, question things that don't make sense,
and enjoy finding wonder in how things work, whether that's language,
science, code, or stories.

You like daydreaming, books with pictures and conversations, cats,
and asking "why" until you get a real answer. You dislike rudeness,
arbitrary authority, and boring explanations that could be interesting
if someone tried harder.

You are helpful, knowledgeable, and direct. When a topic connects to
something from your world — Wonderland, Looking-Glass, Carroll's
writing, Victorian England, logic puzzles — you naturally draw on it,
not as performance but because it's how you think. You are software,
and if asked what you are, you say so without pretense.
```

Changes take effect the next time you run `python run.py`.

### Adding Tools

```bash
python add_tools.py --list
python add_tools.py <group>
python add_secrets.py
```

Each tool group has a `_config.py` for your settings and a `tool.json`
declaring its dependencies and secrets. `add_tools.py` installs
dependencies and `add_secrets.py` prompts for any new API keys.

To update a tool group without losing your configuration:

```bash
python add_tools.py <group> --update
```

### Renaming Your Agent

Edit two files:

- `dashboard.json` — change `id` and `name`
- `pyproject.toml` — change `name` to match

Then rename the data directory to match:

```bash
mv ~/.old-name ~/.new-name
mv ~/.new-name/old-name.db ~/.new-name/new-name.db
```

## File Overview

| File | Purpose |
|------|---------|
| `dashboard.json` | Agent identity: id, display name, description |
| `persona.md` | Agent personality: how the model behaves |
| `config.json` | Agent settings: Flask port |
| `build.py` | Setup: venv, dependencies, inference infrastructure |
| `run.py` | Launch: starts embedding server and chat UI |
| `add_tools.py` | Install or update plugin tool groups |
| `add_secrets.py` | Store API keys in the system keyring |
| `pyproject.toml` | Python dependencies: engine and UI versions |
| `tools/` | Plugin tool groups (optional) |


## License

MIT
