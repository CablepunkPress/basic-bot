"""Bot runtime — the identity and state of a running bot instance."""

from dataclasses import dataclass

from basic_bot.store import MessageStore


@dataclass
class BotRuntime:
    """Everything that distinguishes one bot from another at runtime.

    Built once by create_app() and passed through the chat functions.
    Downstream bots never construct this directly — the factory does it.
    """

    package_name: str
    store: MessageStore
    persona: str
    tool_registry: dict