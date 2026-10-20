"""Where Bountiful keeps its shared files on this machine.

Everything the engine keeps outside an agent's own folder lives under
~/.bountiful and is shared by every agent on the machine: the models
and the llama.cpp build. Those paths are defined here once, and every
module that needs one imports it from here.

Each agent's own data, its conversation database and engine log, lives
in ~/.{agent-id} instead, which the session sets up.

machine.toml records the hardware profile the build chose.
"""

from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"
LLAMA_DIR = BOUNTIFUL_HOME / "llama.cpp"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"
PROFILE_RECORD = BOUNTIFUL_HOME / "machine.toml"
