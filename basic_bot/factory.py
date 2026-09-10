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


def _build_embedder(chat_registry):
    """Build a ManagedEmbedder with lifecycle callbacks.

    The chat_registry reference lets the restart closure use the
    correct local model — not just the profile default.
    """
    import basic_bot.config as config
    from basic_bot.embeddings import LocalEmbedder, ManagedEmbedder
    from basic_bot.infrastructure.server import start, stop, is_running, CHAT, EMBEDDING

    _chat_was_running = False

    def start_embedding():
        nonlocal _chat_was_running
        _chat_was_running = is_running(CHAT)
        if _chat_was_running:
            stop(CHAT)
        start(EMBEDDING)

    def stop_embedding():
        stop(EMBEDDING)
        if _chat_was_running:
            model_id = chat_registry.active_local_model
            start(CHAT, model_id)

    embedder = LocalEmbedder(config.EMBEDDING_URL)
    return ManagedEmbedder(
        embedder,
        start_fn=start_embedding,
        stop_fn=stop_embedding,
    )


def _build_summary_provider():
    """Summary always runs on the local model defined in the profile."""
    import basic_bot.config as config
    from basic_bot.profile import get_summary_config
    from basic_bot.providers.local import LocalProvider
    from basic_bot.providers.protocol import ModelInfo

    summary_config = get_summary_config()
    model_id = summary_config["alias"]
    max_tokens = summary_config["max_tokens"]

    model_info = ModelInfo(
        id=model_id,
        display_name=summary_config.get("alias", model_id),
        provider=summary_config["provider"],
        family=summary_config["family"],
        host="local",
        thinking_type=summary_config.get("thinking_type"),
    )

    return LocalProvider(
        model_id,
        base_url=config.SUMMARY_URL,
        max_tokens=max_tokens,
        model_info=model_info,
    )


def _build_summary_sampling() -> dict:
    """Load summary sampling from the hardware profile."""
    from basic_bot.profile import get_summary_config
    return get_summary_config()["sampling"]


def _build_local_chat_providers() -> dict:
    """Build a LocalProvider for each available local chat model."""
    import basic_bot.config as config
    from basic_bot.profile import get_available_chat_models
    from basic_bot.providers.local import LocalProvider
    from basic_bot.providers.protocol import ModelInfo

    providers = {}
    for model_id, model_config in get_available_chat_models().items():
        model_info = ModelInfo(
            id=model_id,
            display_name=model_config["display_name"],
            provider=model_config["provider"],
            family=model_config["family"],
            host="local",
            rank=model_config.get("rank", 0),
            thinking_type=model_config.get("thinking_type"),
        )
        providers[model_id] = LocalProvider(
            model_id,
            base_url=config.CHAT_URL,
            max_tokens=model_config["max_tokens"],
            model_info=model_info,
            sampling=model_config.get("sampling", {}),
        )
    return providers


def _build_api_chat_provider():
    """Build ClaudeProvider if an API key is available. None otherwise."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from basic_bot.providers.claude import ClaudeProvider
    return ClaudeProvider()


def _build_chat_registry(agent_config: dict):
    """Build the composite chat provider with lifecycle management.

    Merges local and API models into one catalog. Lifecycle closures
    follow the same pattern as _build_embedder — factory captures
    infrastructure imports, registry calls through closures.
    """
    from basic_bot.infrastructure.server import start, stop, CHAT
    from basic_bot.profile import get_default_chat_model
    from basic_bot.providers.registry import ChatProviderRegistry

    local_providers = _build_local_chat_providers()
    api_provider = _build_api_chat_provider()

    local_default_id, _ = get_default_chat_model()
    provider_name = agent_config.get("inference_provider", "local")

    if provider_name == "claude" and api_provider:
        default_model = api_provider.get_default_model()
        initial_active = "api"
        initial_local_model = None
    else:
        default_model = local_default_id
        initial_active = "local"
        initial_local_model = local_default_id

    def start_local(model_id):
        start(CHAT, model_id)

    def stop_local():
        stop(CHAT)

    return ChatProviderRegistry(
        local_providers=local_providers,
        api_provider=api_provider,
        default_model=default_model,
        start_local=start_local,
        stop_local=stop_local,
        initial_active=initial_active,
        initial_local_model=initial_local_model,
    )


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

    # Providers — registry first, embedder needs the reference
    summary_provider = _build_summary_provider()
    chat_provider = _build_chat_registry(config)

    # Embedder — captures registry for correct model restart
    embedder = _build_embedder(chat_provider)

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
        embedder=embedder,
    )
