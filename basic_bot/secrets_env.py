"""Load agent secrets from the OS keyring into environment variables."""

import json
import os
import sys
from pathlib import Path

import keyring
import keyring.errors


def load(agent_path: Path) -> None:
    """Read the agent's API key from keyring and set ANTHROPIC_API_KEY."""
    agent_path = Path(agent_path)
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    try:
        api_key = keyring.get_password(agent_id, "anthropic_api_key")
    except keyring.errors.KeyringError as e:
        sys.exit(
            f"Could not read system keyring: {e}\n"
            "Run 'python add_secrets.py' first."
        )

    if not api_key:
        sys.exit("No Anthropic API key found — run 'python add_secrets.py' first.")

    os.environ["ANTHROPIC_API_KEY"] = api_key
