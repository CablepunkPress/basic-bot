"""Session: a running agent, as a front end sees it.

A front end opens a session, drives it, and closes it. It never
touches the runtime, the registry, servers, the store, or folding.
The session holds the current host, model, Deep Reasoning setting,
and effort, and decides which controls apply, so every front end
shows the same thing.

    with Session.open(agent_path) as session:
        session.start()
        reply = session.send("Hello")

send() returns only after the turn is saved and, when due, memory
is folded. A front end shows one long wait rather than an answer the
user can't yet reply to.

Engine logs always go to ~/.{agent-id}/{agent-id}.log, beside the
conversation database, and rotate so they never grow past a fixed
size. A front end can also echo them to its terminal. The log
records each session's start, with versions, its end, and every
settings change, so it reads as a complete account of what happened.

Engine modules read config values when their functions run, never
when they're imported. That's what lets open() apply config.toml
overrides after everything here has been imported. A module that
reads config at import, including through a default argument, would
capture the value before the override.
"""

import json
import logging
import threading
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler
from pathlib import Path

import basic_bot.config as config
from basic_bot.chat import chat_with_model
from basic_bot.factory import create_runtime
from basic_bot.fold import build_metadata, should_fold
from basic_bot.infrastructure.llamacpp import LLAMA_COMMIT, LLAMA_VERSION
from basic_bot.infrastructure.orchestration import fold_sequential
from basic_bot.infrastructure.server import ServerError, stop_all
from basic_bot.memory import get_messages
from basic_bot.profile import detect_hardware
from basic_bot.providers.protocol import (
    REASONING_ALWAYS,
    REASONING_NONE,
    ModelInfo,
)
from basic_bot.providers.registry import HOST_API, HOST_LOCAL, NoModelsError
from basic_bot.secrets_env import load as load_secrets

logger = logging.getLogger(__name__)

DEFAULT_USER = "local"

# Log rotation: when the log reaches LOG_MAX_BYTES, it becomes
# {agent-id}.log.1, older files move up a number, and the oldest
# beyond LOG_BACKUPS is deleted. Total size stays under about
# LOG_MAX_BYTES * (LOG_BACKUPS + 1).
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3


class SessionError(Exception):
    """Something to tell the user. The message is user-facing."""


class SessionBusy(SessionError):
    """The session is still answering; try again when it's done."""


@dataclass
class Reply:
    """The answer to one message, with what actually produced it."""
    text: str
    seq: int
    model_used: str
    display_name: str
    effort: str | None
    thinking: bool
    fallback: bool


@dataclass
class Controls:
    """What a front end should show, and the settings in effect."""
    host: str
    hosts: list[str]
    model: ModelInfo
    models: list[ModelInfo]          # the current host's models, by rank
    reasoning_shown: bool            # show the Deep Reasoning toggle
    reasoning_locked: bool           # shown checked, and can't be changed
    reasoning_on: bool
    effort_levels: list[str] | None  # None: don't show effort
    effort: str | None


def _read_config(agent_path: Path) -> dict:
    config_file = agent_path / "config.toml"
    if config_file.exists():
        return tomllib.loads(config_file.read_text())
    return {}


def _configure_logging(agent_id: str, log_path: Path, to_terminal: bool) -> None:
    """Send engine logs to the rotating file, and optionally the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        ),
    ]
    if to_terminal:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        format=f"%(asctime)s - [{agent_id}] %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


def _versions() -> str:
    """What's running: basic-bot, the llama.cpp pin, and the profile."""
    try:
        engine = version("basic-bot")
    except PackageNotFoundError:
        engine = "unknown"

    return (
        f"basic-bot {engine}, "
        f"llama.cpp {LLAMA_VERSION} ({LLAMA_COMMIT[:7]}), "
        f"profile {detect_hardware()}"
    )


class Session:
    """A running agent. Create with Session.open()."""

    def __init__(self, runtime, agent_config: dict, log_path: Path):
        self._runtime = runtime
        self._registry = runtime.chat_provider
        self.agent_config = agent_config
        self.log_path = log_path
        self._lock = threading.Lock()

        # The user's choices. None means "this model's default."
        self._model_id: str | None = None
        self._thinking: bool | None = None
        self._effort: str | None = None

    # --- Opening and closing ---

    @classmethod
    def open(
        cls, agent_path: str | Path, *, log_to_terminal: bool = False,
    ) -> "Session":
        """Apply config, start logging, load secrets, and build the runtime.

        Nothing is started yet; call start() for that. Logs always go
        to ~/.{agent-id}/{agent-id}.log. With log_to_terminal, they
        also go to the terminal, for front ends that run in one, like
        the web launcher.
        """
        agent_path = Path(agent_path).resolve()
        dashboard = json.loads((agent_path / "dashboard.json").read_text())
        agent_id = dashboard["id"]

        # Overrides before anything reads config
        agent_config = _read_config(agent_path)
        config.apply_overrides(agent_config)

        log_path = Path.home() / f".{agent_id}" / f"{agent_id}.log"
        _configure_logging(agent_id, log_path, log_to_terminal)

        load_secrets(agent_path)
        try:
            runtime = create_runtime(agent_path)
        except NoModelsError as e:
            logger.error("Session could not open: %s", e)
            raise SessionError(str(e)) from e

        logger.info("Session opened: %s", _versions())
        return cls(runtime, agent_config, log_path)

    def start(self) -> None:
        """Start the starting host's default model."""
        with self._busy():
            self._select(self._registry.get_default_model())

    def close(self) -> None:
        """Stop every server this session started."""
        stop_all()
        logger.info("Session closed")

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- Identity ---

    @property
    def agent_id(self) -> str:
        return self._runtime.agent_id

    @property
    def name(self) -> str:
        return self._runtime.dashboard.get("name", self.agent_id)

    # --- Controls ---

    def controls(self) -> Controls:
        """What to show, from the current choices and the model's rules."""
        models = self._registry.get_models()
        model_id = self._model_id or self._registry.get_default_model()
        info = models[model_id]

        reasoning_on, effort = info.resolve(self._thinking, self._effort)

        effort_levels = info.effort_levels
        if info.effort_needs_reasoning and not reasoning_on:
            effort_levels = None

        return Controls(
            host=info.host,
            hosts=self._registry.hosts(),
            model=info,
            models=sorted(
                (m for m in models.values() if m.host == info.host),
                key=lambda m: m.rank,
            ),
            reasoning_shown=info.reasoning != REASONING_NONE,
            reasoning_locked=info.reasoning == REASONING_ALWAYS,
            reasoning_on=reasoning_on,
            effort_levels=effort_levels,
            effort=effort,
        )

    def select_host(self, host: str) -> None:
        """Switch host, right away, to that host's default model."""
        if host not in self._registry.hosts():
            if host == HOST_API:
                raise SessionError(
                    "No API key is stored, so API models aren't available.\n"
                    "Run 'python3 add_secrets.py' to add one."
                )
            if host == HOST_LOCAL:
                raise SessionError(
                    "No local models are downloaded.\n"
                    "Run 'python3 build.py' to download them."
                )
            raise SessionError(f"Unknown host '{host}'. Choose local or api.")

        with self._busy():
            self._select(self._registry.default_for(host))

    def select_model(self, model_id: str) -> None:
        """Switch model, right away. Settings reset to its defaults."""
        if model_id not in self._registry.get_models():
            raise SessionError(f"Unknown model '{model_id}'.")
        with self._busy():
            self._select(model_id)

    def set_reasoning(self, on: bool) -> None:
        info = self.controls().model
        if info.reasoning == REASONING_ALWAYS:
            raise SessionError(
                f"{info.display_name} always reasons. "
                f"Deep Reasoning can't be turned off."
            )
        if info.reasoning == REASONING_NONE:
            raise SessionError(f"{info.display_name} doesn't use Deep Reasoning.")
        self._thinking = on
        logger.info("Deep Reasoning turned %s", "on" if on else "off")

    def set_effort(self, level: str) -> None:
        controls = self.controls()
        info = controls.model
        if not info.effort_levels:
            raise SessionError(f"{info.display_name} doesn't use effort levels.")
        if controls.effort_levels is None:
            raise SessionError(
                f"Effort applies to {info.display_name} only with "
                f"Deep Reasoning on."
            )
        if level not in info.effort_levels:
            raise SessionError(
                f"Effort for {info.display_name} can be: "
                f"{', '.join(info.effort_levels)}."
            )
        self._effort = level
        logger.info("Effort set to %s", level)

    def _select(self, model_id: str) -> None:
        """Select a model and reset settings. Caller holds the lock."""
        self._model_id = model_id
        self._thinking = None
        self._effort = None

        info = self._registry.get_models()[model_id]
        logger.info(
            "Selecting %s (%s); settings reset to its defaults",
            info.display_name, info.host,
        )
        try:
            self._registry.select(model_id)
        except ServerError as e:
            logger.error("Could not start %s: %s", info.display_name, e)
            raise SessionError(str(e)) from e

    # --- Conversation ---

    def send(self, message: str) -> Reply:
        """Answer a message, save the turn, and fold memory if due.

        Returns only when all three are done.
        """
        message = message.strip()
        if not message:
            raise SessionError("Type a message to send.")

        with self._busy():
            store = self._runtime.store
            model_id = self._model_id or self._registry.get_default_model()

            try:
                result = chat_with_model(
                    self._runtime, DEFAULT_USER, message,
                    model_id, self._effort, self._thinking,
                )
            except ServerError as e:
                raise SessionError(str(e)) from e
            except Exception as e:
                logger.exception("Chat failed")
                raise SessionError(f"The model couldn't answer: {e}") from e

            seq = store.save_turn(
                DEFAULT_USER, message, result["reply"],
                metadata=build_metadata(result),
            )
            logger.info("Saved turn (seq %d–%d)", seq, seq + 1)

            fold_state = should_fold(store, DEFAULT_USER)
            if fold_state:
                try:
                    fold_sequential(self._runtime, store, DEFAULT_USER, fold_state)
                except Exception:
                    # The turn is saved. The boundary didn't move, so the
                    # fold runs again after the next message.
                    logger.exception("Fold failed — will retry next turn")

            return Reply(
                text=result["reply"],
                seq=seq,
                model_used=result["model_used"],
                display_name=result["display_name"],
                effort=result["effort"],
                thinking=result["thinking"],
                fallback=result["fallback"],
            )

    def history(self, limit: int | None = None) -> list[dict]:
        """Recent messages, oldest first: role, content, seq, metadata."""
        if limit is None:
            limit = config.HISTORY_LIMIT
        limit = min(limit, config.WINDOW_CEILING)
        return get_messages(self._runtime.store, DEFAULT_USER, limit)

    # --- Locking ---

    @contextmanager
    def _busy(self):
        """One operation at a time. A second caller gets SessionBusy."""
        if not self._lock.acquire(blocking=False):
            raise SessionBusy(
                "Still working on the last request. "
                "Try again when it finishes."
            )
        try:
            yield
        finally:
            self._lock.release()
