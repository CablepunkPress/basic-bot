# Tools

This folder decides which tools this agent has, beyond the ones every
agent has (searching its archive and recalling messages).

## Curated tool groups

Tool groups that come with Bountiful, such as Wikipedia and Brave
Search, are turned on for this agent with:

    bountiful add <agent> wikipedia

That writes `wikipedia.toml` here, which holds this agent's settings
for the group, with each setting explained. The file being here is
what turns the group on; `bountiful remove <agent> wikipedia` turns it
off. The group's code stays with Bountiful and is updated with it.

Keys a group needs, such as an API key, are kept in your system's
keyring under this agent's id, never in this folder:

    bountiful keys <agent> brave

`bountiful tools <agent>` lists this agent's groups and anything
they're missing.

## Your own tool groups

A tool group of your own is a folder here with a `tool.json` manifest:

    tools/
    ├── example/
    │   ├── tool.json          ← makes the folder a tool group
    │   ├── _shared.py         ← shared code, not a tool
    │   └── example_tool.py    ← a tool
    └── example.toml           ← its settings (optional)

Each tool file defines a TOOL dict (its schema) and a function
`handler(context, **input)`. Files starting with _ are shared modules,
which tool files in the same group import directly:
`from _shared import helper`. `context["settings"]` holds the group's
settings from `example.toml`, and `context["agent_id"]` the agent's
id. A description can name a setting as {{ name }}, filled in when
the agent starts.

A folder of your own with the same name as a curated group takes its
place for this agent.
