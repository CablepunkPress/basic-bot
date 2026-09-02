"""llama-server lifecycle management.

Three server roles on dedicated ports set in config.py

Sequential mode: only one server runs at a time.
At startup, only the chat server loads.
During fold: chat stops → embedding runs → summary runs → chat restarts.

All model paths, launch args, and port assignments come from the
hardware profile loaded by basic_bot.profile.
"""

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from basic_bot.config import EMBEDDING_PORT, SUMMARY_PORT, CHAT_PORT
from basic_bot.infrastructure.llamacpp import SERVER_BIN

logger = logging.getLogger(__name__)

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"

# --- Server roles ---

EMBEDDING = "embedding"
SUMMARY = "summary"
CHAT = "chat"

_ROLE_PORTS = {
    EMBEDDING: EMBEDDING_PORT,
    SUMMARY: SUMMARY_PORT,
    CHAT: CHAT_PORT,
}

LOG_FILES = {
    EMBEDDING: BOUNTIFUL_HOME / "llama-embedding.log",
    SUMMARY: BOUNTIFUL_HOME / "llama-summary.log",
    CHAT: BOUNTIFUL_HOME / "llama-chat.log",
}

HEALTH_TIMEOUT = 60


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

    args += ["--parallel", "1"]

    return args


def _build_env(config: dict) -> dict | None:
    """Build environment overrides from profile config."""
    if config.get("hide_gpu"):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env
    return None


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
        config = get_embedding_config()
        config["embeddings"] = True
        return config

    if role == SUMMARY:
        return get_summary_config()

    if role == CHAT:
        if model_id:
            models = get_chat_models()
            if model_id not in models:
                sys.exit(f"Unknown chat model: {model_id}")
            return models[model_id]
        _, config = get_default_chat_model()
        return config

    sys.exit(f"Unknown server role: {role}")


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
            print(f"{label} already running on port {port}")
            return None
        sys.exit(
            f"Port {port} in use but not responding as llama-server."
        )

    if not SERVER_BIN.exists():
        sys.exit("llama-server missing from ~/.bountiful/. Run 'python build.py'.")
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    # Build command from profile
    launch_args = _build_launch_args(config)
    cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
    ] + launch_args

    # Build environment
    env = _build_env(config)

    # Start and wait for health
    log_path = LOG_FILES[role]
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    print(f"{label} starting on port {port} (log: {log_path})")

    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.exit(f"{label} exited during startup — check {log_path}")
        if _healthy(port):
            print(f"{label} ready")
            _active[role] = process
            return process
        time.sleep(0.5)

    process.terminate()
    sys.exit(f"{label} not ready in {HEALTH_TIMEOUT}s — check {log_path}")


def stop(role: str) -> None:
    """Stop a server by role. No-op if not tracked."""
    process = _active.pop(role, None)
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    logger.info(f"{role.capitalize()} server stopped")


def stop_all() -> None:
    """Stop all tracked servers. Called on teardown."""
    for role in list(_active):
        stop(role)


def is_running(role: str) -> bool:
    """Check if a server role is tracked and alive."""
    process = _active.get(role)
    if process is None:
        return False
    if process.poll() is not None:
        _active.pop(role, None)
        return False
    return True
