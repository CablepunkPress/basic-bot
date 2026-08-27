"""llama-server lifecycle management.

Manages two server instances:
  - Embedding server (CPU, port 11444) — always running for local embeddings
  - Inference server (GPU, port 11445) — runs when inference_provider is local

Each has its own log file for independent debugging.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from basic_bot.infrastructure.llamacpp import SERVER_BIN

BOUNTIFUL_HOME = Path.home() / ".bountiful"
EMBEDDING_LOG = BOUNTIFUL_HOME / "llama-embedding.log"
INFERENCE_LOG = BOUNTIFUL_HOME / "llama-inference.log"

HEALTH_TIMEOUT = 60


def _port_from_url(url: str) -> int:
    parsed = urllib.parse.urlsplit(url)
    if parsed.port is None:
        sys.exit(f"URL has no port: {url}")
    return parsed.port


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


def _check_port(port: int, label: str) -> bool:
    """Check if a server is already running on this port.

    Returns True if an existing healthy server is reused.
    Exits if the port is occupied by something else.
    """
    if _port_in_use(port):
        if _healthy(port):
            print(f"{label} already running on port {port}")
            return True
        sys.exit(
            f"Port {port} is in use but not responding as llama-server.\n"
            f"Something else may be running there."
        )
    return False


def _start_and_wait(
    cmd: list[str],
    port: int,
    log_path: Path,
    label: str,
    env: dict | None = None,
) -> subprocess.Popen:
    """Start a llama-server process and wait for it to become healthy."""
    log = open(log_path, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"{label} starting on port {port} (log: {log_path})")

    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.exit(f"{label} exited during startup — check {log_path}")
        if _healthy(port):
            print(f"{label} ready")
            return process
        time.sleep(0.5)

    process.terminate()
    sys.exit(f"{label} did not become ready in {HEALTH_TIMEOUT}s — check {log_path}")


def ensure_embedding(embedding_url: str) -> subprocess.Popen | None:
    """Start the embedding server (CPU-only) unless already running.

    Returns the process if this call started it, or None if reusing
    an existing server.
    """
    from basic_bot.infrastructure.models import EMBEDDING_MODEL_FILE

    port = _port_from_url(embedding_url)

    if _check_port(port, "Embedding server"):
        return None

    if not SERVER_BIN.exists() or not EMBEDDING_MODEL_FILE.exists():
        sys.exit(
            "llama-server or the embedding model is missing from ~/.bountiful/.\n"
            "Run 'python build.py' first."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    return _start_and_wait(
        cmd=[
            str(SERVER_BIN),
            "-m", str(EMBEDDING_MODEL_FILE),
            "--embeddings",
            "--port", str(port),
            "--ubatch-size", "8192",
            "--ctx-size", "8192",
            "--parallel", "1",
            "--n-gpu-layers", "0",
        ],
        port=port,
        log_path=EMBEDDING_LOG,
        label="Embedding server",
        env=env,
    )


def ensure_inference(model_id: str, inference_url: str) -> subprocess.Popen | None:
    """Start the inference server (GPU) unless already running.

    Returns the process if this call started it, or None if reusing
    an existing server.
    """
    from basic_bot.infrastructure.models import (
        get_inference_model_path,
        get_inference_launch_args,
    )

    port = _port_from_url(inference_url)

    if _check_port(port, "Inference server"):
        return None

    model_path = get_inference_model_path(model_id)

    if not SERVER_BIN.exists():
        sys.exit(
            "llama-server is missing from ~/.bountiful/.\n"
            "Run 'python build.py' first."
        )

    if not model_path.exists():
        sys.exit(
            f"Model file not found: {model_path}\n"
            f"Download it first:\n\n"
            f"    wget -P ~/.bountiful/models/ "
            f"<model-url>\n"
        )

    launch_args = get_inference_launch_args(model_id)

    return _start_and_wait(
        cmd=[
            str(SERVER_BIN),
            "-m", str(model_path),
            "--port", str(port),
            "--alias", model_id,
        ] + launch_args,
        port=port,
        log_path=INFERENCE_LOG,
        label=f"Inference server ({model_id})",
    )


def stop(process: subprocess.Popen) -> None:
    """Terminate a llama-server process."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
