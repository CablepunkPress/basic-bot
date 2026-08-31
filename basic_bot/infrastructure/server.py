"""llama-server lifecycle management.

Three server roles on dedicated ports:
  - Embedding (CPU, port 11333) — Qwen3-Embedding-0.6B Q8_0
  - Summary  (GPU, port 11444) — Qwen3-8B Q4_K_M
  - Chat     (GPU, port 11555) — Qwen3.6-35B-A3B

Sequential mode: only one server runs at a time.
At startup, only the chat server loads.
During fold: chat stops → embedding runs → summary runs → chat restarts.
"""

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from basic_bot.infrastructure.llamacpp import SERVER_BIN

logger = logging.getLogger(__name__)

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"

# --- Server roles ---

EMBEDDING = "embedding"
SUMMARY = "summary"
CHAT = "chat"

PORTS = {
    EMBEDDING: 11333,
    SUMMARY: 11444,
    CHAT: 11555,
}

LOG_FILES = {
    EMBEDDING: BOUNTIFUL_HOME / "llama-embedding.log",
    SUMMARY: BOUNTIFUL_HOME / "llama-summary.log",
    CHAT: BOUNTIFUL_HOME / "llama-chat.log",
}

HEALTH_TIMEOUT = 60


# --- Server configuration ---

@dataclass
class ServerConfig:
    """Everything needed to start a llama-server instance."""
    role: str
    model_path: Path
    port: int
    launch_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] | None = None

    @property
    def label(self) -> str:
        return f"{self.role.capitalize()} server"


# --- Hardware profile: NVIDIA RTX 3060 12GB + 32GB DDR4 ---
# Models and launch args are fixed per hardware target.
# Mac Mini 32GB is next target.

def _embedding_config() -> ServerConfig:
    return ServerConfig(
        role=EMBEDDING,
        model_path=MODELS_DIR / "Qwen3-Embedding-0.6B-Q8_0.gguf",
        port=PORTS[EMBEDDING],
        launch_args=[
            "--embeddings",
            "--ubatch-size", "8192",
            "--ctx-size", "8192",
            "--parallel", "1",
            "--n-gpu-layers", "0",
        ],
        env_overrides={"CUDA_VISIBLE_DEVICES": ""},
    )


def _summary_config() -> ServerConfig:
    return ServerConfig(
        role=SUMMARY,
        model_path=MODELS_DIR / "Qwen3-8B-Q4_K_M.gguf",
        port=PORTS[SUMMARY],
        launch_args=[
            "--n-gpu-layers", "-1",
            "--flash-attn", "on",
            "--ctx-size", "16384",
            "--parallel", "1",
            "--alias", "qwen3-8b-q4_k_m",
        ],
    )


def _chat_config() -> ServerConfig:
    # TODO: Exact filename TBD after download and testing
    return ServerConfig(
        role=CHAT,
        model_path=MODELS_DIR / "Qwen3.6-35B-A3B-IQ4_NL.gguf",
        port=PORTS[CHAT],
        launch_args=[
            "--n-gpu-layers", "-1",
            "--n-cpu-moe", "25",
            "--flash-attn", "on",
            "--ctx-size", "32768",
            "--parallel", "1",
            "--no-mmap",
            "--alias", "qwen3.6-35b-a3b-iq4_nl",
        ],
    )


_CONFIGS = {
    EMBEDDING: _embedding_config,
    SUMMARY: _summary_config,
    CHAT: _chat_config,
}


# --- Active process tracking ---

_active: dict[str, subprocess.Popen] = {}


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

def start(role: str) -> subprocess.Popen | None:
    """Start a server by role.

    Stops any existing server on this role first.
    Exits if the binary or model file is missing.
    Returns the process, or None if reusing an existing server.
    """
    if role in _active:
        stop(role)

    config = _CONFIGS[role]()

    # Reuse an existing healthy server on this port
    if _port_in_use(config.port):
        if _healthy(config.port):
            print(f"{config.label} already running on port {config.port}")
            return None
        sys.exit(
            f"Port {config.port} in use but not responding as llama-server."
        )

    if not SERVER_BIN.exists():
        sys.exit("llama-server missing from ~/.bountiful/. Run 'python build.py'.")
    if not config.model_path.exists():
        sys.exit(f"Model file not found: {config.model_path}")

    # Build command
    cmd = [
        str(SERVER_BIN),
        "-m", str(config.model_path),
        "--port", str(config.port),
    ] + config.launch_args

    # Build environment
    env = None
    if config.env_overrides:
        env = os.environ.copy()
        env.update(config.env_overrides)

    # Start and wait for health
    log_path = LOG_FILES[role]
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    print(f"{config.label} starting on port {config.port} (log: {log_path})")

    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.exit(f"{config.label} exited during startup — check {log_path}")
        if _healthy(config.port):
            print(f"{config.label} ready")
            _active[role] = process
            return process
        time.sleep(0.5)

    process.terminate()
    sys.exit(f"{config.label} not ready in {HEALTH_TIMEOUT}s — check {log_path}")


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
