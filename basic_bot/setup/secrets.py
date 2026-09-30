"""Store an agent's API keys in the OS keyring.

Scans the agent's dashboard.json for identity and tools/*/tool.json
for secret declarations. Shows what's set, prompts for what's missing.

Nothing here is part of the base install. The keyring library is
installed the first time this runs, and the anthropic package is
installed when an Anthropic API key is stored. Both return here
after a rebuild if they're needed again.
"""

import getpass
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


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


def _ensure_package(name: str, reason: str) -> None:
    """Install a package into this environment if it is missing."""
    if importlib.util.find_spec(name) is not None:
        return

    print(f"  Installing {name} {reason}")
    result = subprocess.run([sys.executable, "-m", "pip", "install", name])
    if result.returncode != 0:
        sys.exit(f"  {name} install failed — see output above")
    importlib.invalidate_caches()


def run(agent_path: Path) -> None:
    """Interactive secret setup for the agent at agent_path."""
    agent_path = Path(agent_path)
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    print(f"{agent_id} — API key setup\n")

    _ensure_package("keyring", "to store keys in your system keyring")
    import keyring
    import keyring.errors

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

    if keyring.get_password(agent_id, "anthropic_api_key"):
        _ensure_package("anthropic", "for Claude access")

    print("\nDone. Start the agent with:\n\n    python run.py\n")
