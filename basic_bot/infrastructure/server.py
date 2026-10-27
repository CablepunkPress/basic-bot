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

The chat server's cache is saved and restored across restarts, so a
conversation isn't reread from the start every time the server stops.
The session says where the files go, the agent's slots/ folder; without
that, nothing is saved. Before the chat server stops (a model switch,
an archive search or a fold needing its memory, or the session
closing), its cache is saved there with llama-server's slot save, which
in llama.cpp v0.7.0 also keeps the drafter's state and any context
checkpoints. Right after a chat server starts, a file saved by that
same server setup is restored, so the next request reads only what's
new. A file is named for the model and a short hash of everything it
depends on (the llama.cpp commit and the exact command), so a cache is
never restored into a different setup. Saving or restoring can fail
without harm: the server then reads the conversation as before.
"""

import hashlib
import json
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
from basic_bot.infrastructure.llamacpp import LLAMA_COMMIT
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
_slot_dir: Path | None = None


def set_log_dir(directory: Path) -> None:
    """Send server logs to this folder, such as an agent's logs/server/."""
    global _log_dir
    _log_dir = Path(directory)


def set_slot_dir(directory: Path) -> None:
    """Save and restore the chat server's cache in this folder, such as
    an agent's slots/. Without one, nothing is saved."""
    global _slot_dir
    _slot_dir = Path(directory)


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

# How long saving or restoring the chat server's cache may take. A
# save waits for a request in progress to finish, so this also limits
# how long a stop can be held up by one.
SLOT_TIMEOUT = 30


class ServerError(RuntimeError):
    """A llama-server could not be started. The message is user-facing."""


# --- Active process tracking ---

# Each entry is a watchdog process. Keeping it referenced keeps its
# stdin pipe open, which is what keeps its llama-server running.
_active: dict[str, subprocess.Popen] = {}

# The cache file for each running server that saves one (chat only)
_slot_files: dict[str, str] = {}


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
    # Context checkpoints. A model whose cache can't be cut back to an
    # earlier point (sliding-window attention, as in Gemma 4, or a
    # recurrent state, as in Qwen3.8) gets copies of its cache saved, so
    # the server can return to an earlier point; up to 32 by default.
    # Gemma 4 31B's are about 425 MiB each, two or three per request, and
    # llama.cpp doesn't log making them: llama-server grew to 14 GB in nine
    # short turns. With none, a turn that goes back rereads the whole
    # conversation. For chat, an agent's config.toml can override the
    # profile (CTX_CHECKPOINTS in config.py; see start()).
    if "ctx_checkpoints" in entry:
        args += ["--ctx-checkpoints", str(entry["ctx_checkpoints"])]

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


# --- Saving and restoring the chat server's cache ---

def _slot_file(alias: str, server_cmd: list[str]) -> str:
    """The cache file's name for this exact server setup.

    A saved cache is only valid for the build and settings that made
    it, so the name carries a hash of the llama.cpp commit and the
    command. Changing either starts a new file, and the old one is
    removed the next time this model's cache is saved.
    """
    key = hashlib.sha256(
        (LLAMA_COMMIT + "\n" + shlex.join(server_cmd)).encode("utf-8")
    ).hexdigest()[:12]
    return f"{alias}-{key}.bin"


def _slot_request(port: int, action: str, filename: str) -> dict:
    """Ask llama-server to save or restore slot 0's cache."""
    request = urllib.request.Request(
        f"http://localhost:{port}/slots/0?action={action}",
        data=json.dumps({"filename": filename}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=SLOT_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _save_slot(role: str) -> None:
    """Save a running server's cache, if it keeps one. Never raises."""
    filename = _slot_files.get(role)
    if filename is None or _slot_dir is None:
        return
    try:
        result = _slot_request(_role_port(role), "save", filename)
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("Couldn't save the %s server's cache: %s", role, e)
        return

    saved = result.get("n_saved", 0)
    if not saved:
        # Nothing worth keeping, and an older file would now be stale
        (_slot_dir / filename).unlink(missing_ok=True)
        return
    logger.info(
        "Saved the %s server's cache: %s tokens, %.0f MB in %.1f s",
        role, f"{saved:,}", result.get("n_written", 0) / 1e6,
        result.get("timings", {}).get("save_ms", 0) / 1000,
    )

    # Older files for the same model, from an earlier setup
    alias = filename.rsplit("-", 1)[0]
    for old in _slot_dir.glob(f"{alias}-*.bin"):
        if old.name != filename and old.stem.rsplit("-", 1)[0] == alias:
            old.unlink(missing_ok=True)


def _restore_slot(role: str) -> None:
    """Restore a just-started server's saved cache, if there is one.

    Never raises: without the cache, the server reads the conversation
    as it always has. A file it can't load is removed.
    """
    filename = _slot_files.get(role)
    if filename is None or _slot_dir is None:
        return
    path = _slot_dir / filename
    if not path.exists():
        return
    try:
        result = _slot_request(_role_port(role), "restore", filename)
    except urllib.error.HTTPError as e:
        logger.warning(
            "Couldn't restore the %s server's cache (HTTP %d); removing it",
            role, e.code,
        )
        path.unlink(missing_ok=True)
        return
    except (urllib.error.URLError, OSError, ValueError) as e:
        logger.warning("Couldn't restore the %s server's cache: %s", role, e)
        return
    logger.info(
        "Restored the %s server's cache: %s tokens, %.0f MB in %.1f s",
        role, f"{result.get('n_restored', 0):,}", result.get("n_read", 0) / 1e6,
        result.get("timings", {}).get("restore_ms", 0) / 1000,
    )


def _checkpoints_override() -> int:
    """The agent's own context checkpoint count for its chat model."""
    value = config.CTX_CHECKPOINTS
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ServerError(
            f"ctx_checkpoints in this agent's config.toml is {value!r}. "
            f"It must be a whole number, 0 or more, or left out to use "
            f"the profile's."
        )
    return value


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
    if role == CHAT and config.CTX_CHECKPOINTS is not None:
        entry = {**entry, "ctx_checkpoints": _checkpoints_override()}
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

    # The chat server can save and restore its cache (see the module
    # docstring). llama-server requires the folder to exist.
    _slot_files.pop(role, None)
    if role == CHAT and _slot_dir is not None:
        _slot_dir.mkdir(parents=True, exist_ok=True)
        server_cmd += ["--slot-save-path", str(_slot_dir)]
        slot_file = _slot_file(entry["alias"], server_cmd)
    else:
        slot_file = None

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
                if slot_file is not None:
                    _slot_files[role] = slot_file
                    _restore_slot(role)
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
    """Stop a server by role, saving its cache first if it keeps one.

    No-op if not tracked.
    """
    if role not in _active:
        return
    _save_slot(role)
    _slot_files.pop(role, None)
    process = _active.pop(role)
    _terminate(process)
    logger.info("%s server stopped", role.capitalize())


def stop_all() -> None:
    """Stop all tracked servers. Called on teardown."""
    for role in list(_active):
        stop(role)
