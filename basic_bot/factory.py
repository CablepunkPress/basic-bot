"""Factory for building bot runtimes.

create_runtime() assembles a BotRuntime from an agent directory.
This is the seam between deployment-specific configuration (profiles,
hardware detection) and engine-core code that never knows what it's
running on.

Everything deployment-specific gets resolved here and injected into
the runtime. Engine-core code reads from the runtime, never reaches
outward.
"""

import json
import logging
import tomllib
from pathlib import Path

from basic_bot.runtime import BotRuntime
from basic_bot.tools import build_registry

logger = logging.getLogger(__name__)


def _read_config(agent_path: Path) -> dict:
    """Read agent config.toml, returning empty dict if absent."""
    config_file = agent_path / "config.toml"
    if config_file.exists():
        return tomllib.loads(config_file.read_text())
    return {}


def _build_summary_provider():
    """Summary always runs on the local model defined in the profile."""
    from basic_bot.config import SUMMARY_URL
    from basic_bot.profile import get_summary_config
    from basic_bot.providers.local import LocalProvider

    config = get_summary_config()
    model_id = config["alias"]
    max_tokens = config["max_tokens"]
    return LocalProvider(model_id, base_url=SUMMARY_URL, max_tokens=max_tokens)


def _build_summary_sampling() -> dict:
    """Load summary sampling from the hardware profile."""
    from basic_bot.profile import get_summary_config
    return get_summary_config()["sampling"]


def _build_chat_provider(config: dict):
    provider_name = config.get("inference_provider", "local")

    if provider_name == "local":
        from basic_bot.config import CHAT_URL
        from basic_bot.profile import get_default_chat_model
        from basic_bot.providers.local import LocalProvider
        from basic_bot.providers.protocol import ModelInfo

        model_id, model_config = get_default_chat_model()

        model_info = ModelInfo(
            id=model_id,
            display_name=model_config["display_name"],
            provider=model_config["provider"],
            family=model_config["family"],
            host="local",
            rank=model_config.get("rank", 0),
            thinking_type=model_config.get("thinking_type"),
        )

        return LocalProvider(
            model_id,
            base_url=CHAT_URL,
            max_tokens=model_config["max_tokens"],
            model_info=model_info,
            sampling=model_config.get("sampling", {}),
        )

    if provider_name == "claude":
        from basic_bot.providers.claude import ClaudeProvider
        return ClaudeProvider()

    raise ValueError(f"Unknown inference provider: {provider_name!r}")


def _build_store(agent_id: str):
    """Create the SQLite store for the agent."""
    from basic_bot.store_sqlite import SQLiteMessageStore

    sqlite_dir = Path.home() / f".{agent_id}"
    sqlite_dir.mkdir(exist_ok=True)
    return SQLiteMessageStore(str(sqlite_dir / f"{agent_id}.db"))


def create_runtime(agent_path: str | Path) -> BotRuntime:
    """Build a fully configured bot runtime from an agent directory."""

    agent_path = Path(agent_path).resolve()

    # Dashboard — agent identity, loaded first so agent_id drives logging
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    logging.basicConfig(
        format=f"%(asctime)s - [{agent_id}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    # Config — agent settings
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

    # Capabilities — engine-owned
    capabilities_path = Path(__file__).parent / "instructions" / "capabilities.md"
    capabilities_text = capabilities_path.read_text().strip()

    persona = persona_text + "\n\n" + capabilities_text

    # Storage
    store = _build_store(agent_id)

    # Tools
    tool_registry = build_registry(agent_path)

    # Providers
    summary_provider = _build_summary_provider()
    chat_provider = _build_chat_provider(config)

    # Sampling — resolved from profile, carried on runtime
    summary_sampling = _build_summary_sampling()

    return BotRuntime(
        agent_id=agent_id,
        agent_path=agent_path,
        store=store,
        persona=persona,
        tool_registry=tool_registry,
        dashboard=dashboard,
        chat_provider=chat_provider,
        summary_provider=summary_provider,
        summary_sampling=summary_sampling,
    )
