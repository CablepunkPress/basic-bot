"""llama-server lifecycle management.

Three server roles on dedicated ports set in config.py

Sequential mode: only one server runs at a time.
At startup, only the chat server loads.
During fold: chat stops → embedding runs → summary runs → chat restarts.

All model paths, launch args, and port assignments come from the
hardware profile loaded by basic_bot.profile.

Failures raise ServerError with a message meant for the user. A
server can start because of a click, not only at launch, so a failed
start must be reportable instead of ending the process.
"""

import logging
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import basic_bot.config as config
from basic_bot.infrastructure.llamacpp import SERVER_BIN

logger = logging.getLogger(__name__)

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"

# --- Server roles ---

EMBEDDING = "embedding"
SUMMARY = "summary"
CHAT = "chat"

_ROLE_PORTS = {
    EMBEDDING: config.EMBEDDING_PORT,
    SUMMARY: config.SUMMARY_PORT,
    CHAT: config.CHAT_PORT,
}

LOG_FILES = {
    EMBEDDING: BOUNTIFUL_HOME / "llama-embedding.log",
    SUMMARY: BOUNTIFUL_HOME / "llama-summary.log",
    CHAT: BOUNTIFUL_HOME / "llama-chat.log",
}

# Generous on purpose: start() returns as soon as the server is healthy,
# so this only limits how long a broken start takes to fail. The first
# Metal launch of a large model can be slow.
HEALTH_TIMEOUT = 180


class ServerError(RuntimeError):
    """A llama-server could not be started. The message is user-facing."""


# --- Active process tracking ---

_active: dict[str, subprocess.Popen] = {}


# --- Build launch command from profile config ---

def _build_launch_args(config: dict) -> list[str]:
    """Translate profile config dict into llama-server CLI flags."""
    args = []

    if "gpu_layers" in config:
        args += ["--n-gpu-layers", str(config["gpu_layers"])]
    if config.get("flash_attn"):
        args += ["--flash-attn", config["flash_attn"]]
    if "ctx_size" in config:
        args += ["--ctx-size", str(config["ctx_size"])]
    if "batch_size" in config:
        args += ["--batch-size", str(config["batch_size"])]
    if "ubatch_size" in config:
        args += ["--ubatch-size", str(config["ubatch_size"])]
    if "n_cpu_moe" in config:
        args += ["--n-cpu-moe", str(config["n_cpu_moe"])]
    if config.get("no_mmap"):
        args += ["--no-mmap"]
    if "alias" in config:
        args += ["--alias", config["alias"]]
    if config.get("embeddings", False):
        args += ["--embeddings"]

    # Speculative decoding: a drafter proposes tokens the model checks
    # in one pass. Speeds up generation without changing the output.
    draft = config.get("draft")
    if draft:
        if draft.get("type") == "mtp":
            args += ["--spec-type", "draft-mtp"]
        elif "file" in draft:
            draft_path = MODELS_DIR / draft["file"]
            if draft_path.exists():
                args += ["-md", str(draft_path)]
                if "gpu_layers" in draft:
                    args += ["-ngld", str(draft["gpu_layers"])]
            else:
                logger.warning(
                    "Draft model %s not found — starting without it. "
                    "Run 'python3 build.py' to download it.",
                    draft["file"],
                )

    # Use the chat template embedded in each GGUF. Muse Glimmer
    # requires this; it is harmless for the other models.
    args += ["--jinja"]

    # One slot, so a single conversation gets the full ctx_size.
    # llama-server divides --ctx-size across slots.
    args += ["--parallel", "1"]

    return args


def _config_for_role(role: str, model_id: str | None = None) -> dict:
    """Load the profile config for a server role.

    For chat, model_id selects which chat model entry to use.
    For embedding and summary, the profile has a single entry.
    """
    from basic_bot.profile import (
        get_embedding_config,
        get_summary_config,
        get_chat_models,
        get_default_chat_model,
    )

    if role == EMBEDDING:
        config = dict(get_embedding_config())
        config["embeddings"] = True
        return config

    if role == SUMMARY:
        return get_summary_config()

    if role == CHAT:
        if model_id:
            models = get_chat_models()
            if model_id not in models:
                raise ServerError(f"Unknown chat model: {model_id}")
            return models[model_id]
        _, config = get_default_chat_model()
        return config

    raise ServerError(f"Unknown server role: {role}")


# --- Port and health utilities ---

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=2,
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


# --- Lifecycle ---

def start(role: str, model_id: str | None = None) -> subprocess.Popen | None:
    """Start a server by role.

    For chat, model_id selects which model to load. Defaults to the
    profile's default chat model.

    Stops any existing server on this role first.
    Returns the process, or None if reusing an existing server.
    Raises ServerError if the server can't be started.
    """
    if role in _active:
        stop(role)

    config = _config_for_role(role, model_id)
    port = _ROLE_PORTS[role]
    model_path = MODELS_DIR / config["file"]
    label = f"{role.capitalize()} server"

    # Reuse an existing healthy server on this port
    if _port_in_use(port):
        if _healthy(port):
            logger.info("%s already running on port %d", label, port)
            return None
        raise ServerError(
            f"Port {port} is in use by something other than llama-server."
        )

    if not SERVER_BIN.exists():
        raise ServerError(
            "llama-server is missing from ~/.bountiful/. "
            "Run 'python3 build.py'."
        )
    if not model_path.exists():
        raise ServerError(
            f"Model file not found: {model_path}. "
            f"Run 'python3 build.py' to download it."
        )

    # Build command from profile
    launch_args = _build_launch_args(config)
    cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
    ] + launch_args

    # Start and wait for health. The child keeps its own handle to
    # the log, so ours can close once it starts.
    log_path = LOG_FILES[role]
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
        )
    logger.info("%s starting on port %d (log: %s)", label, port, log_path)

    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServerError(
                f"{label} exited during startup — check {log_path}"
            )
        if _healthy(port):
            logger.info("%s ready", label)
            _active[role] = process
            return process
        time.sleep(0.5)

    _terminate(process)
    raise ServerError(
        f"{label} not ready in {HEALTH_TIMEOUT}s — check {log_path}"
    )


def _terminate(process: subprocess.Popen) -> None:
    """Stop a process, forcefully if it doesn't exit in time."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def stop(role: str) -> None:
    """Stop a server by role. No-op if not tracked."""
    process = _active.pop(role, None)
    if process is None:
        return
    _terminate(process)
    logger.info("%s server stopped", role.capitalize())


def stop_all() -> None:
    """Stop all tracked servers. Called on teardown."""
    for role in list(_active):
        stop(role)
