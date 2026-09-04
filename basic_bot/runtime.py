"""Bot runtime — the identity and state of a running bot instance."""

from dataclasses import dataclass, field
from pathlib import Path

from basic_bot.embeddings import LocalEmbedder
from basic_bot.providers.protocol import InferenceProvider
from basic_bot.store import MessageStore


@dataclass
class BotRuntime:
    """Everything that distinguishes one bot from another at runtime.

    Built once by create_runtime() and passed through the chat functions.
    The factory assembles this from the agent directory, config, and
    hardware profile. Engine-core code reads from the runtime — it
    never reaches outward to profiles or infrastructure.
    """

    agent_id: str
    agent_path: Path
    store: MessageStore
    persona: str
    tool_registry: dict
    dashboard: dict
    chat_provider: InferenceProvider
    summary_provider: InferenceProvider
    summary_sampling: dict = field(default_factory=dict)
    embedder: LocalEmbedder | None = None
