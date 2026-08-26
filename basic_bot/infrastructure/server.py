"""llama-server lifecycle management.

Start, health-check, and stop the shared llama-server process
for local embeddings (and eventually summarization/inference).
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
from basic_bot.infrastructure.models import MODEL_FILE

BOUNTIFUL_HOME = Path.home() / ".bountiful"
LLAMA_LOG = BOUNTIFUL_HOME / "llama-server.log"

HEALTH_TIMEOUT = 60  # seconds to wait for llama-server to come up


def _port_from_url(url: str) -> int:
    parsed = urllib.parse.urlsplit(url)
    if parsed.port is None:
        sys.exit(f"EMBEDDING_URL has no port: {url}")
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


def ensure(embedding_url: str) -> subprocess.Popen | None:
    """Start llama-server unless one is already running.

    Returns the process if this call started it (caller tears it down),
    or None if an existing server is being reused.
    """
    port = _port_from_url(embedding_url)

    if _port_in_use(port):
        if _healthy(port):
            print(f"llama-server already running on port {port} — reusing it")
            return None
        sys.exit(
            f"Port {port} is in use but not responding as llama-server.\n"
            f"Something else may be running there — stop it, or set\n"
            f"EMBEDDING_URL in your environment to use a different port."
        )

    if not SERVER_BIN.exists() or not MODEL_FILE.exists():
        sys.exit(
            "llama-server or the embedding model is missing from ~/.bountiful/.\n"
            "Run 'python build.py' first."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    log = open(LLAMA_LOG, "w")
    process = subprocess.Popen(
        [
            str(SERVER_BIN),
            "-m", str(MODEL_FILE),
            "--embeddings",
            "--port", str(port),
            "--ubatch-size", "8192",
            "--ctx-size", "8192",
            "--parallel", "1",
            "--n-gpu-layers", "0",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"llama-server starting on port {port} (log: {LLAMA_LOG})")

    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.exit(f"llama-server exited during startup — check {LLAMA_LOG}")
        if _healthy(port):
            print("llama-server ready — embeddings available")
            return process
        time.sleep(0.5)

    process.terminate()
    sys.exit(f"llama-server did not become ready in {HEALTH_TIMEOUT}s — check {LLAMA_LOG}")


def stop(process: subprocess.Popen) -> None:
    """Terminate a llama-server process started by ensure()."""
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    print("llama-server stopped")
