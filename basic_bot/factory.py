"""Factory for building bot runtimes.

create_runtime() assembles a BotRuntime from an agent directory.
This is the seam between deployment-specific configuration (profiles,
hardware detection) and engine-core code that never knows what it's
running on.

Everything deployment-specific gets resolved here and injected into
the runtime. Engine-core code reads from the runtime, never reaches
outward.

Logging is configured by whoever opens the runtime, normally
basic_bot.session, so each front end decides where logs go.
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


def _read_markdown_dir(directory: Path) -> list[str]:
    """Read every .md file in a directory, alphabetically.

    Files starting with an underscore are skipped, so a directory
    can carry a README or notes that never reach the model.
    Returns an empty list if the directory does not exist.
    """
    if not directory.is_dir():
        return []
    return [
        md_file.read_text().strip()
        for md_file in sorted(directory.glob("*.md"))
        if not md_file.name.startswith("_")
    ]


def _local_url(port: int) -> str:
    """Address of a llama-server this machine runs."""
    return f"http://localhost:{port}"


def _local_model_info(model_id: str, entry: dict, display_name: str):
    """Describe a local model for the UI from its profile entry.

    The [reasoning] table's mode and defaults become ModelInfo fields.
    Its template keys stay with the provider.
    """
    from basic_bot.providers.protocol import (
        REASONING_ALWAYS,
        REASONING_NONE,
        ModelInfo,
    )

    reasoning = entry.get("reasoning") or {}
    mode = reasoning.get("mode", REASONING_NONE)

    return ModelInfo(
        id=model_id,
        display_name=display_name,
        provider=entry["provider"],
        family=entry["family"],
        host="local",
        rank=entry.get("rank", 0),
        reasoning=mode,
        reasoning_default=(
            mode == REASONING_ALWAYS or reasoning.get("default", False)
        ),
        effort_levels=reasoning.get("effort_levels"),
        effort_default=reasoning.get("effort_default"),
        effort_needs_reasoning=reasoning.get("effort_needs_reasoning", False),
    )


def _build_embedder(chat_router):
    """Build a ManagedEmbedder that pauses chat while it runs.

    The embedding server needs the memory the chat server is using,
    so chat is suspended through the router, its single owner,
    rather than stopped directly.
    """
    import basic_bot.config as config
    from basic_bot.embeddings import LocalEmbedder, ManagedEmbedder
    from basic_bot.profile import get_embedding_config
    from basic_bot.infrastructure.server import start, stop, EMBEDDING

    embedding_config = get_embedding_config()

    def start_embedding():
        chat_router.suspend()
        try:
            start(EMBEDDING)
        except Exception:
            chat_router.resume()
            raise

    def stop_embedding():
        try:
            stop(EMBEDDING)
        finally:
            chat_router.resume()

    embedder = LocalEmbedder(
        _local_url(config.EMBEDDING_PORT),
        ctx_size=embedding_config["ctx_size"],
        query_prefix=embedding_config.get("query_prefix", ""),
        passage_prefix=embedding_config.get("passage_prefix", ""),
        model_name=embedding_config.get("alias", ""),
    )
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

    summary_config = get_summary_config()
    model_id = summary_config["alias"]

    return LocalProvider(
        model_id,
        base_url=_local_url(config.SUMMARY_PORT),
        max_tokens=summary_config["max_tokens"],
        model_info=_local_model_info(model_id, summary_config, model_id),
        reasoning=summary_config.get("reasoning"),
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

    providers = {}
    for model_id, model_config in get_available_chat_models().items():
        providers[model_id] = LocalProvider(
            model_id,
            base_url=_local_url(config.CHAT_PORT),
            max_tokens=model_config["max_tokens"],
            model_info=_local_model_info(
                model_id, model_config, model_config["display_name"],
            ),
            reasoning=model_config.get("reasoning"),
            sampling=model_config.get("sampling", {}),
        )
    return providers


def _build_api_chat_provider():
    """Build ClaudeProvider if an Anthropic API key is stored. None otherwise."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from basic_bot.providers.claude import ClaudeProvider
    except ImportError:
        logger.warning(
            "An Anthropic API key is stored, but the anthropic package is not "
            "installed. Run 'python3 add_secrets.py' to restore Claude access."
        )
        return None
    return ClaudeProvider()


def _build_chat_router(agent_config: dict):
    """Build the composite chat provider.

    Nothing starts here. The session selects the starting model.
    """
    from basic_bot.infrastructure.server import start, stop, CHAT
    from basic_bot.profile import get_default_chat_model
    from basic_bot.providers.router import (
        HOST_API,
        HOST_LOCAL,
        ChatRouter,
    )

    local_default, _ = get_default_chat_model()
    provider_name = agent_config.get("inference_provider", "local")
    starting_host = HOST_API if provider_name == "claude" else HOST_LOCAL

    def start_local(model_id):
        start(CHAT, model_id)

    def stop_local():
        stop(CHAT)

    return ChatRouter(
        local_providers=_build_local_chat_providers(),
        api_provider=_build_api_chat_provider(),
        local_default=local_default,
        starting_host=starting_host,
        start_local=start_local,
        stop_local=stop_local,
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

    # Dashboard — agent identity
    dashboard = json.loads((agent_path / "dashboard.json").read_text())
    agent_id = dashboard["id"]

    # Config — agent settings
    config = _read_config(agent_path)

    # Stable prompt prefix, assembled once at startup. Order runs from
    # the user's own words to engine instructions to optional domain
    # knowledge. chat.py appends the TOOLS and MEMORY sections.

    # Persona — user-authored, in the agent directory
    persona_text = (agent_path / "persona.md").read_text().strip()
    persona_text = persona_text.replace("{{ name }}", dashboard.get("name", agent_id))
    sections = [persona_text]

    # Instructions — engine-owned, every .md file in instructions/
    instructions_dir = Path(__file__).parent / "instructions"
    sections.extend(_read_markdown_dir(instructions_dir))

    # Context — user-added domain knowledge, only if files exist
    context_parts = _read_markdown_dir(agent_path / "context")
    if context_parts:
        sections.append("# CONTEXT\n\n" + "\n\n".join(context_parts))

    # Prompt prefix: persona + instructions + context (should be renamed)
    persona = "\n\n".join(sections)

    # Storage
    store = _build_store(agent_id)

    # Tools
    tool_registry = build_registry(agent_path)

    # Providers — registry first, embedder needs the reference
    summary_provider = _build_summary_provider()
    chat_provider = _build_chat_router(config)

    # Embedder — suspends chat through the router
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
