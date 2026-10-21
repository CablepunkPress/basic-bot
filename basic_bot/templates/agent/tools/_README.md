# Tools

Drop tool groups here. A tool group is a directory of Python files:

    tools/
    ├── github/
    │   ├── _auth.py            ← shared code, not a tool
    │   ├── create_branch.py    ← a tool
    │   └── read_file.py        ← a tool
    └── example/
        ├── _config.py
        ├── _auth.py
        └── example_tool.py

Each tool file defines a TOOL dict (the schema) and a handler function.
Files starting with _ are shared modules — tool files in the same group
import them directly: `from _auth import auth_headers`.

Tool groups that need API keys use their own keyring service name.
Add the group's keys to add_secrets.py KEYS and re-run it.

The agent discovers everything here at startup. No registration,
no imports, no configuration — the directory is the installation.
