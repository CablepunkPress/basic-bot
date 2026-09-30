"""Load agent secrets from the OS keyring into environment variables.

The keyring library is optional. It is installed when the user adds
keys through add_secrets.py or installs a tool group that needs it.
Without it, no keys are loaded and the agent runs local-only.
"""

import json
import os
from pathlib import Path


def load(agent_path: Path) -> None:
    """Read the agent's API key from keyring and set ANTHROPIC_API_KEY."""
    try:
        import keyring
        import keyring.errors
    except ImportError:
        return

    agent_path = Path(agent_path)
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    try:
        api_key = keyring.get_password(agent_id, "anthropic_api_key")
    except keyring.errors.KeyringError:
        return

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
