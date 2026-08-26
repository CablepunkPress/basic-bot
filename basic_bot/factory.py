"""Factory for building bot runtimes.

create_runtime() assembles a BotRuntime from an agent directory.
The agent directory contains persona.md, dashboard.json, config.json,
and an optional tools/ directory. Web framework setup (Flask, FastAPI)
is the caller's responsibility — the factory builds the runtime,
not the routes.
"""

import json
import logging
import tomllib
from pathlib import Path

from basic_bot.config import STORAGE_BACKEND
from basic_bot.runtime import BotRuntime
from basic_bot.tools import build_registry


logger = logging.getLogger(__name__)


def _read_config(agent_path: Path) -> dict:
    """Read agent config.toml, returning empty dict if absent.

    The engine provides defaults for missing fields — agents only
    need to include values that differ from defaults.
    """
    config_file = agent_path / "config.toml"
    if config_file.exists():
        return tomllib.loads(config_file.read_text())
    return {}


def _build_provider(config: dict):
    """Create the inference provider from agent config.

    Imports are deferred so a local-only install never pulls in the
    Anthropic SDK and an API-only install never imports LocalProvider.
    """
    provider_name = config.get("inference_provider", "claude")

    if provider_name == "local":
        from basic_bot.providers.local import LocalProvider

        model_id = config.get("default_model", "qwen3-8b")
        kwargs: dict = {"model_id": model_id}

        max_tokens = config.get("max_tokens")
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        return LocalProvider(**kwargs)

    if provider_name == "claude":
        from basic_bot.providers.claude import ClaudeProvider

        return ClaudeProvider()

    raise ValueError(f"Unknown inference provider: {provider_name!r}")


def _build_store(backend: str, collection: str):
    """Create the MessageStore for the configured backend.

    Imports are deferred so a local install never pulls in google.cloud
    and a cloud deploy never pulls in sqlite3 internals.

    SQLite databases live at ~/.{agent_id}/{agent_id}.db — the agent ID
    is the collection, and the dot-directory is created if absent.
    """
    if backend == "firestore":
        from basic_bot.store_firestore import FirestoreMessageStore
        return FirestoreMessageStore(collection)

    if backend == "sqlite":
        from basic_bot.store_sqlite import SQLiteMessageStore
        sqlite_dir = Path.home() / f".{collection}"
        sqlite_dir.mkdir(exist_ok=True)
        return SQLiteMessageStore(str(sqlite_dir / f"{collection}.db"))

    raise ValueError(f"Unknown storage backend: {backend!r}")


def create_runtime(agent_path: str | Path) -> BotRuntime:
    """Build a fully configured bot runtime from an agent directory.

    Args:
        agent_path: Directory containing persona.md, dashboard.json,
            and optionally config.json and tools/.

    Returns:
        A BotRuntime with store, persona, tools, and identity configured.
    """

    agent_path = Path(agent_path).resolve()

    # Dashboard — agent identity, loaded first so agent_id drives logging
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    logging.basicConfig(
        format=f"%(asctime)s - [{agent_id}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    # Config — agent settings, defaults for missing fields
    config = _read_config(agent_path)

    # Persona — user-authored, in the agent directory
    persona_text = (agent_path / "persona.md").read_text().strip()
    persona_text = persona_text.replace("{{ name }}", dashboard.get("name", agent_id))

    # Context — user-added domain knowledge files (optional)
    context_dir = agent_path / "context"
    if context_dir.is_dir():
        context_parts = []
        for md_file in sorted(context_dir.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            context_parts.append(md_file.read_text().strip())
        if context_parts:
            persona_text += "\n\n" + "\n\n".join(context_parts)

    # Capabilities — engine-owned, always from the basic_bot package
    capabilities_path = Path(__file__).parent / "instructions" / "capabilities.md"
    capabilities_text = capabilities_path.read_text().strip()

    persona = persona_text + "\n\n" + capabilities_text

    # Storage — collection is the agent ID
    store = _build_store(STORAGE_BACKEND, agent_id)

    # Tools — belt from engine, box from agent_path/tools/
    tool_registry = build_registry(agent_path)

    # Provider — selected by config, defaults to Claude
    provider = _build_provider(config)

    return BotRuntime(
        agent_id=agent_id,
        agent_path=agent_path,
        store=store,
        persona=persona,
        tool_registry=tool_registry,
        dashboard=dashboard,
        provider=provider,
    )
