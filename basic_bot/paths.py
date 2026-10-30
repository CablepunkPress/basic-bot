"""Where Bountiful keeps things on this machine.

Every agent lives in the home folder, at ~/ID, with its conversation
database, logs, and backups in its data folder. Everything else the
engine keeps lives under ~/.bountiful and is shared by every agent on
the machine: the models, the llama.cpp build, the hardware profile the
build chose (machine.toml), and the registered agents (agents.json).
These paths are defined here once, and every module that needs one
imports it from here.
"""

from pathlib import Path

# Where every agent's folder is: AGENTS_HOME / ID
AGENTS_HOME = Path.home()

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"
LLAMA_DIR = BOUNTIFUL_HOME / "llama.cpp"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"
PROFILE_RECORD = BOUNTIFUL_HOME / "machine.toml"
AGENTS_REGISTRY = BOUNTIFUL_HOME / "agents.json"
