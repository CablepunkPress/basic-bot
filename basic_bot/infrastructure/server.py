"""llama-server lifecycle management.

Three server roles, each on its own port from basic_bot.config.
Every server runs on this machine, and the engine owns all of them.

Sequential mode: only one server runs at a time, because the models
don't fit in memory together. The chat router decides when the chat
server runs; a fold suspends chat through it, runs the embedding
server, then the summary server, and resumes chat.

All model paths and launch args come from the hardware profile loaded
by basic_bot.profile. Ports are read from config each time they're
needed, so config.toml overrides apply however early this module is
imported.

Each llama-server runs under a small watchdog (watchdog.py), in its
own session, separate from the terminal:

- The separate session means a Ctrl-C in the terminal reaches only
  the front end. The engine alone decides when a server stops: on a
  model switch, a fold, or session close.
- The watchdog stops llama-server if the engine ends without stopping
  it, as after a force quit or a crash. The engine holds the write end
  of the watchdog's stdin; when the engine ends, the operating system
  closes it, and the watchdog acts. Each process object keeps its pipe
  open, so a server's process must stay referenced while it runs.

To the rest of the engine, the watchdog is invisible: same port, same
health check, same log, and it exits when llama-server exits, with the
same code. Only a forced stop differs: a force-kill can't be passed
on, so it goes to the whole session, reaching llama-server directly.

Once shutdown begins (see basic_bot.shutdown), start() declines to
start anything, so cleanup code still running, such as a fold resuming
chat, can't launch a server that would outlive the program. It
declines quietly rather than raising, so an error can't replace the
exception that is shutting the program down.

Failures raise ServerError with a message meant for the user. A
server can start because of a click, not only at launch, so a failed
start must be reportable instead of ending the process.

Every start writes a new log, so a restart never erases an earlier
run's: a fold restarts the chat server, and its timings used to be
lost with the old log. The session says where the logs go, the
agent's logs/server/ folder, since a server runs for one agent at a
time; without that, they go in ~/.bountiful/logs/. Each log is named
for when its server started, in UTC, and what it ran, and begins with
the exact command. The newest LOGS_KEPT of each role are kept.
"""

import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import basic_bot.config as config
from basic_bot.paths import BOUNTIFUL_HOME, MODELS_DIR, SERVER_BIN
from basic_bot.profile import (
    get_chat_models,
    get_default_chat_model,
    get_embedding_config,
    get_summary_config,
)
from basic_bot.shutdown import shutting_down

logger = logging.getLogger(__name__)

# Run as a file with the agent's own Python, never imported
WATCHDOG = Path(__file__).with_name("watchdog.py")

# --- Server roles ---

EMBEDDING = "embedding"
SUMMARY = "summary"
CHAT = "chat"

# Where server logs go when no session has said: the machine's own
LOG_DIR_DEFAULT = BOUNTIFUL_HOME / "logs"

# How many of each role's logs to keep. A fold starts three servers,
# so this covers several folds' worth.
LOGS_KEPT = 20

_log_dir: Path | None = None


def set_log_dir(directory: Path) -> None:
    """Send server logs to this folder, such as an agent's logs/server/."""
    global _log_dir
    _log_dir = Path(directory)


def _new_log_path(role: str, entry: dict) -> Path:
    """A new log for one server run, named for its time, role, and model.

    Older logs of the same role beyond LOGS_KEPT are deleted.
    """
    directory = _log_dir or LOG_DIR_DEFAULT
    directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%SZ")
    name = f"{stamp}-{role}"
    if role == CHAT and entry.get("alias"):
        name += f"-{entry['alias']}"

    # Sorted by name is sorted by time, since every name starts with it
    earlier = sorted(
        p for p in directory.glob("*.log")
        if p.stem.split("Z-", 1)[-1].split("-", 1)[0] == role
    )
    for old in earlier[:max(0, len(earlier) - (LOGS_KEPT - 1))]:
        try:
            old.unlink()
        except OSError:
            pass

    return directory / f"{name}.log"

# Generous on purpose: start() returns as soon as the server is healthy,
# so this only limits how long a broken start takes to fail. The first
# Metal launch of a large model can be slow.
HEALTH_TIMEOUT = 180

# How long a server gets to stop before it's killed. Longer than the
# watchdog's own GRACE, so the watchdog normally finishes the stop.
STOP_TIMEOUT = 10


class ServerError(RuntimeError):
    """A llama-server could not be started. The message is user-facing."""


# --- Active process tracking ---

# Each entry is a watchdog process. Keeping it referenced keeps its
# stdin pipe open, which is what keeps its llama-server running.
_active: dict[str, subprocess.Popen] = {}


def _role_port(role: str) -> int:
    """The port for a server role, read from config at call time."""
    ports = {
        EMBEDDING: config.EMBEDDING_PORT,
        SUMMARY: config.SUMMARY_PORT,
        CHAT: config.CHAT_PORT,
    }
    if role not in ports:
        raise ServerError(f"Unknown server role: {role}")
    return ports[role]


# --- Build launch command from hardware profile model entry ---

def _build_launch_args(entry: dict) -> list[str]:
    """Translate a profile entry into llama-server CLI flags."""
    args = []

    if "gpu_layers" in entry:
        args += ["--n-gpu-layers", str(entry["gpu_layers"])]
    if entry.get("flash_attn"):
        args += ["--flash-attn", entry["flash_attn"]]
    if "ctx_size" in entry:
        args += ["--ctx-size", str(entry["ctx_size"])]
    if "batch_size" in entry:
        args += ["--batch-size", str(entry["batch_size"])]
    if "ubatch_size" in entry:
        args += ["--ubatch-size", str(entry["ubatch_size"])]
    if "n_cpu_moe" in entry:
        args += ["--n-cpu-moe", str(entry["n_cpu_moe"])]
    if entry.get("no_mmap"):
        args += ["--no-mmap"]
    if "alias" in entry:
        args += ["--alias", entry["alias"]]
    if entry.get("embeddings", False):
        args += ["--embeddings"]
    # llama-server keeps copies of earlier prompts' states in ordinary
    # memory, up to 8 GB by default, in case a request returns to one.
    # A single conversation rarely does, and on a Mac that memory is
    # what the model needs, so a profile can lower it or turn it off (0).
    if "cache_ram_mib" in entry:
        args += ["--cache-ram", str(entry["cache_ram_mib"])]
    # The KV cache holds every token of the current prompt, so the
    # server's memory follows the longest prompt of a session. Storing
    # it at 8 bits (q8_0) instead of 16 roughly halves that. A
    # quantized V cache needs flash attention.
    if "cache_type_k" in entry:
        args += ["--cache-type-k", entry["cache_type_k"]]
    if "cache_type_v" in entry:
        args += ["--cache-type-v", entry["cache_type_v"]]

    # Speculative decoding: a drafter proposes tokens the model checks
    # in one pass. Speeds up generation without changing the output.
    # A drafter file alone: llama.cpp reads its type from the file.
    # type = "mtp" alone: drafting layers built into the model file.
    # Both: a drafter file whose type llama.cpp can't read from it,
    # such as Gemma 4's MTP assistants.
    draft = entry.get("draft")
    if draft:
        spec_type = ["--spec-type", "draft-mtp"] if draft.get("type") == "mtp" else []
        if "file" in draft:
            draft_path = MODELS_DIR / draft["file"]
            if draft_path.exists():
                args += ["-md", str(draft_path)] + spec_type
                if "gpu_layers" in draft:
                    args += ["-ngld", str(draft["gpu_layers"])]
                # A drafter file keeps its own KV cache
                if "cache_type_k" in draft:
                    args += ["--cache-type-k-draft", draft["cache_type_k"]]
                if "cache_type_v" in draft:
                    args += ["--cache-type-v-draft", draft["cache_type_v"]]
            else:
                logger.warning(
                    "Draft model %s not found — starting without it. "
                    "Run 'python3 build.py' to download it.",
                    draft["file"],
                )
        else:
            args += spec_type

    # Use the chat template embedded in each GGUF. Muse Glimmer
    # requires this; it is harmless for the other models.
    args += ["--jinja"]

    # One slot, so a single conversation gets the full ctx_size.
    # llama-server divides --ctx-size across slots.
    args += ["--parallel", "1"]

    return args


def _profile_entry(role: str, model_id: str | None = None) -> dict:
    """The hardware profile's entry for a server role.

    For chat, model_id selects which chat model's entry. For embedding
    and summary, the profile has a single entry.
    """
    if role == EMBEDDING:
        entry = dict(get_embedding_config())
        entry["embeddings"] = True
        return entry

    if role == SUMMARY:
        return get_summary_config()

    if role == CHAT:
        if model_id:
            models = get_chat_models()
            if model_id not in models:
                raise ServerError(f"Unknown chat model: {model_id}")
            return models[model_id]
        _, entry = get_default_chat_model()
        return entry

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


def _port_taken_message(port: int) -> str:
    """Explain a busy port, and how to find what's holding it."""
    if _healthy(port):
        return (
            f"Port {port} already has a llama-server that this session "
            f"didn't start. Another agent may be running, or an earlier "
            f"run didn't shut down cleanly.\n\n"
            f"Close any other agents. If none are running, the server is "
            f"left over. To stop it, open Activity Monitor, search for "
            f"llama-server, select it, and click the stop button.\n\n"
            f"Or, from a terminal, find it with:\n"
            f"  lsof -i :{port}\n"
            f"then stop it with:\n"
            f"  kill <PID>"
        )
    return (
        f"Port {port} is in use by another program.\n\n"
        f"This shows which one:\n"
        f"  lsof -i :{port}\n"
        f"Quit that program, then try again."
    )


# --- Lifecycle ---

def start(role: str, model_id: str | None = None) -> subprocess.Popen | None:
    """Start a server by role.

    For chat, model_id selects which model to load. Defaults to the
    profile's default chat model.

    Stops this role's existing server first. Never reuses a server it
    didn't start: one already on the port may hold a different model
    or settings, and couldn't be stopped to switch models.

    If startup is interrupted for any reason, including Ctrl-C or a
    shutdown signal, the half-started server is stopped before the
    interruption carries on.

    Returns the process, or None if shutdown has begun, in which case
    nothing is started. Raises ServerError if the server can't start.
    """
    if shutting_down():
        logger.info("Not starting %s server: shutting down", role)
        return None

    if role in _active:
        stop(role)

    entry = _profile_entry(role, model_id)
    port = _role_port(role)
    model_path = MODELS_DIR / entry["file"]
    label = f"{role.capitalize()} server"

    if _port_in_use(port):
        raise ServerError(_port_taken_message(port))

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
    launch_args = _build_launch_args(entry)
    server_cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
    ] + launch_args

    # Start llama-server under its watchdog, in their own session, so a
    # Ctrl-C in the terminal reaches only the front end. stdin is the
    # watchdog's link to the engine: held open here, never written.
    # close_fds keeps each watchdog from inheriting another's link.
    # The children keep their own handle to the log, so ours can close.
    cmd = [sys.executable, str(WATCHDOG), "--"] + server_cmd
    log_path = _new_log_path(role, entry)
    with open(log_path, "w") as log_file:
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log_file.write(f"# {label} started {started}\n# {shlex.join(server_cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    logger.info("%s starting on port %d (log: %s)", label, port, log_path)

    try:
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if shutting_down():
                # Shutdown began while this server was loading
                _terminate(process)
                logger.info("%s stopped while starting: shutting down", label)
                return None
            if process.poll() is not None:
                _close_link(process)
                raise ServerError(
                    f"{label} exited during startup — check {log_path}"
                )
            if _healthy(port):
                logger.info("%s ready", label)
                _active[role] = process
                return process
            time.sleep(0.5)

        raise ServerError(
            f"{label} not ready in {HEALTH_TIMEOUT}s — check {log_path}"
        )
    except BaseException:
        # Not yet tracked, so nothing else would stop it
        if process.poll() is None:
            logger.info("%s stopped during startup", label)
            _terminate(process)
        raise


def _close_link(process: subprocess.Popen) -> None:
    """Close our end of a watchdog's stdin, once it's no longer needed."""
    if process.stdin and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass


def _terminate(process: subprocess.Popen) -> None:
    """Stop a server through its watchdog, forcefully if it takes too long.

    A polite stop goes to the watchdog, which passes it on. A force-kill
    can't be passed on, so it goes to the whole session, which reaches
    llama-server directly: the watchdog started its session, so the
    session's group ID is the watchdog's process ID.
    """
    try:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    finally:
        _close_link(process)


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
