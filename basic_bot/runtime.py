"""Bot runtime — the identity and state of a running bot instance."""

from dataclasses import dataclass
from pathlib import Path

from basic_bot.store import MessageStore


@dataclass
class BotRuntime:
    """Everything that distinguishes one bot from another at runtime.

    Built once by create_runtime() and passed through the chat functions.
    Downstream bots never construct this directly — the factory does it.
    """

    agent_id: str
    agent_path: Path
    store: MessageStore
    persona: str
    tool_registry: dict
    dashboard: dict
    provider: InferenceProvider
