# basic_bot/setup/tools.py
"""Install tool groups from the extend-a-bot repository.

Downloads tool groups from CablepunkPress/extend-a-bot on GitHub
and copies them into an agent's tools/ directory. Handles install,
update, list, and tool dependency installation.
"""

import io
import json
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_OWNER = "CablepunkPress"
REPO_NAME = "extend-a-bot"
REPO_BRANCH = "main"
TARBALL_URL = (
    f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
    f"/archive/refs/heads/{REPO_BRANCH}.tar.gz"
)


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def _fetch_tarball() -> tarfile.TarFile:
    """Download the extend-a-bot repo as a tarball."""
    print(f"Fetching {REPO_OWNER}/{REPO_NAME}...")
    try:
        with urllib.request.urlopen(TARBALL_URL) as response:
            data = response.read()
    except urllib.error.URLError as e:
        _fail(f"Could not download extend-a-bot: {e}")

    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def _list_groups(tar: tarfile.TarFile) -> list[str]:
    """Find all tool groups in the tarball (directories with tool.json)."""
    prefix = f"{REPO_NAME}-{REPO_BRANCH}/"
    groups = set()

    for member in tar.getmembers():
        if not member.name.startswith(prefix):
            continue
        relative = member.name[len(prefix):]
        parts = relative.split("/")
        if len(parts) == 2 and parts[1] == "tool.json":
            groups.add(parts[0])

    return sorted(groups)


def _extract_group(tar: tarfile.TarFile, group_name: str,
                   tools_dir: Path) -> None:
    """Extract one tool group into tools/."""
    prefix = f"{REPO_NAME}-{REPO_BRANCH}/{group_name}/"
    dest = tools_dir / group_name

    members = [
        m for m in tar.getmembers()
        if m.name.startswith(prefix) and m.name != prefix
    ]

    if not members:
        _fail(f"Tool group '{group_name}' not found in extend-a-bot")

    dest.mkdir(parents=True, exist_ok=True)

    for member in members:
        relative = member.name[len(prefix):]
        target = dest / relative

        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
            src.close()

    print(f"Installed tools/{group_name}/")


def _install_group_deps(tools_dir: Path, group_name: str) -> None:
    """Install pip dependencies declared in a tool group's tool.json."""
    manifest_path = tools_dir / group_name / "tool.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    deps = manifest.get("dependencies", [])
    if not deps:
        return

    print(f"    installing dependencies: {', '.join(deps)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + deps,
    )
    if result.returncode != 0:
        _fail("tool dependency install failed — see output above")


def _print_next_steps(group_dir: Path) -> None:
    """Read tool.json and print what the user does next."""
    manifest_path = group_dir / "tool.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())

    print(f"\n--- {manifest.get('name', group_dir.name)} ---")
    print(f"{manifest.get('description', '')}")

    config_items = manifest.get("config", [])
    secrets = manifest.get("secrets", [])

    steps = []

    if config_items:
        files = sorted(set(c["file"] for c in config_items))
        lines = []
        for c in config_items:
            lines.append(f"     {c['name']} — {c['label']}")
        steps.append(
            f"Edit tools/{group_dir.name}/{files[0]}:\n" + "\n".join(lines)
        )

    if secrets:
        steps.append(
            f"Run: python add_secrets.py   ({len(secrets)} key(s) needed)"
        )

    if steps:
        print("\nNext steps:")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
    print()


def _cmd_list(tools_dir: Path) -> None:
    """List available tool groups."""
    tar = _fetch_tarball()
    groups = _list_groups(tar)

    if not groups:
        print("No tool groups found in extend-a-bot.")
        return

    print(f"\nAvailable tool groups ({len(groups)}):\n")
    for name in groups:
        prefix = f"{REPO_NAME}-{REPO_BRANCH}/{name}/tool.json"
        member = None
        try:
            member = tar.getmember(prefix)
        except KeyError:
            pass

        description = ""
        if member:
            f = tar.extractfile(member)
            if f:
                manifest = json.loads(f.read())
                description = manifest.get("description", "")
                f.close()

        installed = (tools_dir / name).exists()
        status = " (installed)" if installed else ""
        print(f"  {name}{status}")
        if description:
            print(f"    {description}")
    print(f"\nInstall with: python add_tools.py <name>")


def _cmd_install(tools_dir: Path, group_name: str,
                 force: bool = False) -> None:
    """Install a tool group."""
    dest = tools_dir / group_name

    if dest.exists() and not force:
        _fail(
            f"tools/{group_name}/ already exists. Your edits would be lost.\n"
            f"To overwrite: python add_tools.py {group_name} --force"
        )

    tar = _fetch_tarball()
    groups = _list_groups(tar)

    if group_name not in groups:
        _fail(
            f"Tool group '{group_name}' not found.\n"
            f"Available: {', '.join(groups) or 'none'}\n"
            f"Run: python add_tools.py --list"
        )

    _extract_group(tar, group_name, tools_dir)
    _install_group_deps(tools_dir, group_name)
    _print_next_steps(dest)


def _cmd_update(tools_dir: Path, group_name: str) -> None:
    """Update a tool group, preserving user config files."""
    dest = tools_dir / group_name

    if not dest.exists():
        _fail(
            f"tools/{group_name}/ does not exist. Install it first:\n"
            f"  python add_tools.py {group_name}"
        )

    # Read the existing manifest to find protected files
    manifest_path = dest / "tool.json"
    protected = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        protected = {c["file"] for c in manifest.get("config", [])}

    tar = _fetch_tarball()
    groups = _list_groups(tar)

    if group_name not in groups:
        _fail(f"Tool group '{group_name}' not found in extend-a-bot")

    prefix = f"{REPO_NAME}-{REPO_BRANCH}/{group_name}/"
    members = [
        m for m in tar.getmembers()
        if m.name.startswith(prefix) and m.name != prefix and m.isfile()
    ]

    updated = []
    skipped = []

    for member in members:
        relative = member.name[len(prefix):]
        target = dest / relative

        if relative in protected:
            skipped.append(relative)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        src = tar.extractfile(member)
        if src is None:
            continue
        target.write_bytes(src.read())
        src.close()
        updated.append(relative)

    print(f"Updated tools/{group_name}/:")
    for f in updated:
        print(f"  updated: {f}")
    for f in skipped:
        print(f"  preserved: {f}")

    _install_group_deps(tools_dir, group_name)

    # Check if new manifest has config entries the old one didn't
    new_manifest_path = dest / "tool.json"
    if new_manifest_path.exists():
        new_manifest = json.loads(new_manifest_path.read_text())
        new_config = {c["file"] for c in new_manifest.get("config", [])}
        added_config = new_config - protected
        if added_config:
            print(f"\n  New config file(s) added: {', '.join(added_config)}")
            print("  Review and fill in your values.")


def run(agent_path: Path, args: list[str]) -> None:
    """CLI entry point — called by the add_tools.py shim."""
    agent_path = Path(agent_path)
    tools_dir = agent_path / "tools"

    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  python add_tools.py --list             Show available groups\n"
            "  python add_tools.py <group>            Install a tool group\n"
            "  python add_tools.py <group> --update   Update code, preserve config\n"
            "  python add_tools.py <group> --force    Overwrite existing"
            
        )
        return

    if args[0] == "--list":
        _cmd_list(tools_dir)
        return

    group_name = args[0]

    if "--update" in args:
        _cmd_update(tools_dir, group_name)
        return

    force = "--force" in args
    _cmd_install(tools_dir, group_name, force)
