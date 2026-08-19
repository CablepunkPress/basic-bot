"""Store an agent's API keys in the OS keyring.

Scans the agent's dashboard.json for identity and tools/*/tool.json
for secret declarations. Shows what's set, prompts for what's missing.
"""

import getpass
import json
import sys
from pathlib import Path

import keyring
import keyring.errors


def _discover_tool_secrets(agent_path: Path) -> list[tuple[str, str, str]]:
    """Scan tools/*/tool.json for secret declarations."""
    tools_dir = agent_path / "tools"
    if not tools_dir.is_dir():
        return []

    secrets = []
    for manifest_path in sorted(tools_dir.glob("*/tool.json")):
        manifest = json.loads(manifest_path.read_text())
        group_name = manifest_path.parent.name
        for s in manifest.get("secrets", []):
            secrets.append((
                s["service"],
                s["key"],
                f"{s['label']} ({group_name})",
            ))
    return secrets


def run(agent_path: Path) -> None:
    """Interactive secret setup for the agent at agent_path."""
    agent_path = Path(agent_path)
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    print(f"{agent_id} — API key setup\n")

    try:
        keyring.get_password(agent_id, "probe")
    except keyring.errors.KeyringError as e:
        sys.exit(
            f"Could not access your system keyring: {e}\n"
            "Make sure KWallet, GNOME Keyring, or Keychain is available."
        )

    keys = [
        (agent_id, "anthropic_api_key", "Anthropic API key"),
    ]
    keys.extend(_discover_tool_secrets(agent_path))

    for service, key_name, label in keys:
        existing = keyring.get_password(service, key_name)
        status = "set" if existing else "not set"
        print(f"  {label} [{status}]")

        if existing:
            replace = input("    Replace? [y/N] ").strip().lower()
            if replace != "y":
                continue

        value = getpass.getpass(f"    {label} (hidden, Enter to skip): ").strip()
        if not value:
            print("    skipped")
            continue

        keyring.set_password(service, key_name, value)
        if keyring.get_password(service, key_name) != value:
            sys.exit("    stored but could not be read back — keyring problem")
        print("    stored")

    print("\nDone. Start the agent with:\n\n    python run.py\n")
