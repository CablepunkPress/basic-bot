"""Launch Basic Bot locally.

Starts llama-server (local embeddings) as a subprocess, waits for it to
be ready, then starts the Flask UI on port 5084. Ctrl+C stops both.

Run with any Python — it re-executes itself inside .venv automatically:

    python run.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


# Re-exec inside the venv if we aren't already. Everything above this
# line must be stdlib-only — venv packages aren't importable yet.
if sys.prefix == sys.base_prefix:
    interpreter = _venv_python()
    if not interpreter.exists():
        sys.exit("No .venv found — run 'python build.py' first.")
    os.execv(str(interpreter), [str(interpreter), *sys.argv])

# --- From here on we're inside the venv ---

import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import keyring
import keyring.errors

KEYRING_SERVICE = "basic-bot"
KEYRING_USERNAME = "anthropic_api_key"

try:
    api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
except keyring.errors.KeyringError as e:
    sys.exit(f"Could not read system keyring: {e}\nRun 'python build.py' first.")

if not api_key:
    sys.exit("No Anthropic API key found — run 'python build.py' first.")

os.environ["ANTHROPIC_API_KEY"] = api_key

from basic_bot.config import EMBEDDING_PROVIDER, EMBEDDING_URL

HOME = Path.home() / ".basic-bot"
SERVER_BIN = HOME / "llama.cpp" / "build" / "bin" / "llama-server"
MODEL_FILE = HOME / "models" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
LLAMA_LOG = HOME / "llama-server.log"

FLASK_PORT = 5084
HEALTH_TIMEOUT = 60  # seconds to wait for llama-server to come up


def _port_from_url(url: str) -> int:
    parsed = urllib.parse.urlsplit(url)
    if parsed.port is None:
        sys.exit(f"EMBEDDING_URL has no port: {url}")
    return parsed.port


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_llama_server() -> subprocess.Popen:
    if not SERVER_BIN.exists() or not MODEL_FILE.exists():
        sys.exit(
            "llama-server or the embedding model is missing.\n"
            "Run 'python build.py' first."
        )

    port = _port_from_url(EMBEDDING_URL)
    if _port_in_use(port):
        sys.exit(
            f"Port {port} is already in use — another llama-server, or Ollama,\n"
            f"may be running. Stop it, or set EMBEDDING_URL in your\n"
            f"environment to use a different port."
        )

    log = open(LLAMA_LOG, "w")
    process = subprocess.Popen(
        [
            str(SERVER_BIN),
            "-m", str(MODEL_FILE),
            "--embeddings",
            "--port", str(port),
            "--ubatch-size", "8192",    # max tokens per embedding input
            "-c", "8192",               # context window — matches ubatch; max for model is 32768
            "--np", "1",                # single slot — only one client calls this server
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    print(f"llama-server starting on port {port} (log: {LLAMA_LOG})")

    deadline = time.monotonic() + HEALTH_TIMEOUT
    health_url = f"http://localhost:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.exit(f"llama-server exited during startup — check {LLAMA_LOG}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print("llama-server ready — embeddings available")
                    return process
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)

    process.terminate()
    sys.exit(f"llama-server did not become ready in {HEALTH_TIMEOUT}s — check {LLAMA_LOG}")


def main() -> None:
    llama = start_llama_server() if EMBEDDING_PROVIDER == "local" else None

    try:
        from web.server import create_local_app
        app = create_local_app()
        # use_reloader=False: the reloader re-runs this script in a child
        # process, which would spawn a second llama-server.
        app.run(port=FLASK_PORT, debug=True, use_reloader=False)
    finally:
        if llama is not None:
            llama.terminate()
            try:
                llama.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llama.kill()
            print("llama-server stopped")


if __name__ == "__main__":
    main()
