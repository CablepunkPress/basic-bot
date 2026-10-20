"""Build orchestrator.

Called by build.py in the bountiful repo, as `python -m basic_bot`.

Takes the GPU backend from the operating system, checks the build
tools for it, and compiles the pinned llama.cpp with that backend.
Then detects the machine's memory, which on Linux needs the
llama-server just built, chooses and records the hardware profile,
and downloads the models it requires.

Pinned models, those with sha256 and size in the profile, are checked
against their checksum and downloaded again when the pin changes. An
interrupted download keeps its partial file and resumes on the next
run. Unpinned models are accepted if present, but never downloaded.

Which models are downloaded follows a rule. The embedding model, the
summary model, and the default chat model, with its drafter, are
mandatory. Every other chat model is optional: the build asks about
each one once, defaulting to no, and records the answer beside the
models in .choices.json, since every agent on the machine shares
them. An optional model already on disk counts as accepted.
--models asks about every optional model again. With no one at the
keyboard, as when an app or a hosted service runs the build, only
mandatory models are downloaded and nothing is recorded, so the
questions wait for the next build someone runs by hand.

Model files the profile doesn't use, or that belong to a declined
model, are listed with their sizes and offered for deletion,
defaulting to no.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.paths import MODELS_DIR
from basic_bot.profile import (
    ProfileError,
    choose_profile,
    get_downloadable_models,
    platform_backend,
)

# Which files have passed their checksum, so a rebuild doesn't rehash
# 40GB. Only a cache: the checksum in the profile is the truth.
VERIFIED_FILE = MODELS_DIR / ".verified.json"

# The user's answers about optional models: model ids accepted and
# declined.
CHOICES_FILE = MODELS_DIR / ".choices.json"

# On a machine that's already set up, mandatory downloads larger than
# this ask first, as when an update makes a new chat model the default.
CONFIRM_BYTES = 5 * 1000 ** 3

CHUNK = 1024 * 1024
TIMEOUT = 60          # seconds without data before a stalled download fails
SPACE_MARGIN = 2 * 1024 ** 3
PROGRESS_INTERVAL = 0.5

# Waits between retries after a stalled or dropped connection. The
# count resets whenever a retry makes progress, so a long download
# survives any number of separate blips.
RETRY_WAITS = [10, 30, 60, 120, 300]

ROLE_NAMES = {
    "embedding": "embedding model",
    "summary": "summary model",
    "chat": "chat model",
    "draft": "draft model",
}


def _fail(message: str) -> NoReturn:
    sys.exit(f"\nERROR: {message}")


# --- Describing models and numbers ---

def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _describe(model: dict) -> str:
    """'chat model: Muse Glimmer 30B Q4_K_XL' — role, then name."""
    name = model.get("display_name") or model["file"]
    return f"{ROLE_NAMES[model['role']]}: {name}"


def _source(url: str) -> str:
    """Where a model comes from, e.g. 'Hugging Face, Abiray/Nemotron-...'."""
    parts = urlparse(url)
    if parts.netloc == "huggingface.co":
        segments = parts.path.strip("/").split("/")
        if len(segments) >= 2:
            return f"Hugging Face, {segments[0]}/{segments[1]}"
        return "Hugging Face"
    return parts.netloc or url


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


# --- Verification ---

def _load_verified() -> dict:
    try:
        return json.loads(VERIFIED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_verified(verified: dict) -> None:
    VERIFIED_FILE.write_text(json.dumps(verified, indent=2) + "\n")


def _sha256(path: Path) -> str:
    print("      verifying checksum...", end="", flush=True)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK * 8):
            digest.update(block)
    print(" done")
    return digest.hexdigest()


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _is_current(model: dict, verified: dict) -> bool:
    """True if the file on disk is the one the profile asks for."""
    dest = MODELS_DIR / model["file"]
    if not dest.exists():
        return False

    sha256 = model["sha256"]
    if sha256 is None:
        return True   # unpinned: presence is all that can be checked

    if dest.stat().st_size != model["size"]:
        return False
    if verified.get(model["file"]) == sha256:
        return True

    # Present but never verified, e.g. downloaded before pinning
    if _sha256(dest) == sha256:
        verified[model["file"]] = sha256
        _save_verified(verified)
        return True
    return False


# --- Downloading ---

def _partial_path(model: dict) -> Path:
    dest = MODELS_DIR / model["file"]
    return dest.with_name(dest.name + ".partial")


def _fetch(url: str, partial: Path, size: int | None) -> None:
    """Download url into partial, resuming from any bytes already there."""
    have = partial.stat().st_size if partial.exists() else 0
    if size is not None and have > size:
        partial.unlink()
        have = 0
    if size is not None and have == size:
        return

    request = urllib.request.Request(url)
    if have:
        print(f"      resuming at {_gb(have)}")
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if have and response.status != 206:
            print("      the server can't resume, starting over")
            have = 0

        total = size
        if total is None:
            length = response.headers.get("Content-Length")
            total = have + int(length) if length else None

        started = time.monotonic()
        last_print = 0.0
        done = have

        with open(partial, "ab" if have else "wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                done += len(chunk)

                now = time.monotonic()
                if now - last_print >= PROGRESS_INTERVAL:
                    last_print = now
                    _progress(done, total, done - have, now - started)

        _progress(done, total, done - have, time.monotonic() - started)
    print()


def _progress(done: int, total: int | None, session: int, elapsed: float) -> None:
    """One progress line, rewritten in place."""
    if total:
        line = f"      {done * 100 // total}% ({_gb(done)} of {_gb(total)})"
    else:
        line = f"      {_gb(done)}"

    # Rate and time left, once there's enough to measure
    if elapsed >= 5 and session > 0:
        rate = session / elapsed
        line += f" — {rate / 1e6:.1f} MB/s"
        if total and done < total:
            line += f", about {_duration((total - done) / rate)} left"

    print("\r" + line.ljust(76), end="", flush=True)


def _download(model: dict, number: int, count: int, verified: dict) -> None:
    dest = MODELS_DIR / model["file"]
    partial = _partial_path(model)

    print()
    print(f"    downloading {number} of {count}, {_describe(model)}")
    print(f"      from {_source(model['url'])}")

    retries = 0
    while True:
        before = _size(partial)
        error = None
        try:
            _fetch(model["url"], partial, model["size"])
        except urllib.error.HTTPError as e:
            if e.code < 500:
                # The server refused the request itself, e.g. a missing
                # file. Waiting won't change that.
                _fail(
                    f"the download of the {_describe(model)} failed: {e}\n\n"
                    f"URL: {model['url']}"
                )
            print()
            error = e
        except Exception as e:
            print()
            error = e

        after = _size(partial)
        complete = model["size"] is None or after == model["size"]
        if error is None and complete:
            break
        if error is None:
            # The stream ended normally but the file is short: the
            # connection was closed partway through.
            error = "the connection closed early"

        # Progress since the last attempt earns a fresh set of retries
        if after > before:
            retries = 0

        if retries >= len(RETRY_WAITS):
            _fail(
                f"the download of the {_describe(model)} stopped: {error}\n\n"
                f"The partial file is kept.\n"
                f"Run 'python3 build.py' again to resume."
            )

        wait = RETRY_WAITS[retries]
        retries += 1
        print(f"      connection lost ({error}); retrying in {wait} seconds")
        time.sleep(wait)

    if model["sha256"] is not None:
        if _sha256(partial) != model["sha256"]:
            partial.unlink()
            _fail(
                f"the {_describe(model)} failed its checksum and was "
                f"deleted.\n\n"
                f"Run 'python3 build.py' again to download it fresh."
            )

    partial.replace(dest)
    if model["sha256"] is not None:
        verified[model["file"]] = model["sha256"]
        _save_verified(verified)
    print(f"      saved to {dest}")


def _remaining_bytes(needed: list[dict]) -> int:
    """Bytes still to download, counting any partial files already there."""
    remaining = 0
    for model in needed:
        partial = _partial_path(model)
        have = partial.stat().st_size if partial.exists() else 0
        remaining += max(model["size"] - have, 0)
    return remaining


# --- Choosing optional models ---

def _ask(question: str, default: bool) -> bool:
    """A yes-or-no question, answered at the keyboard."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"    {question} {hint} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _load_choices() -> dict[str, list[str]]:
    try:
        data = json.loads(CHOICES_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    return {key: list(data.get(key, [])) for key in ("accepted", "declined")}


def _record_choice(choices: dict, model_id: str, accepted: bool) -> None:
    for key in ("accepted", "declined"):
        if model_id in choices[key]:
            choices[key].remove(model_id)
    choices["accepted" if accepted else "declined"].append(model_id)
    CHOICES_FILE.write_text(json.dumps(choices, indent=2) + "\n")


def _wants(model: dict, size: int, choices: dict,
           ask_again: bool, interactive: bool) -> bool:
    """Whether an optional chat model should be on this machine."""
    model_id = model["model_id"]
    name = model.get("display_name") or model["file"]

    if ask_again and interactive:
        if model["exists"]:
            wanted = _ask(f"Keep {name} ({_gb(size)}, downloaded)?", True)
        else:
            wanted = _ask(f"Download {name} ({_gb(size)})?", False)
        _record_choice(choices, model_id, wanted)
        return wanted

    if model_id in choices["accepted"]:
        return True
    if model_id in choices["declined"]:
        return False

    # Already here, e.g. an old default that became optional
    if model["exists"]:
        _record_choice(choices, model_id, True)
        return True

    if not interactive:
        return False
    wanted = _ask(f"{name} is available ({_gb(size)}). Download it?", False)
    _record_choice(choices, model_id, wanted)
    return wanted


def _offer_cleanup(unused: list[Path], interactive: bool, verified: dict) -> None:
    """List model files nothing uses, and offer to delete them."""
    if not unused:
        return
    total = sum(p.stat().st_size for p in unused)
    print()
    print("    model files this profile doesn't use:")
    for path in unused:
        print(f"      {path.name} ({_gb(path.stat().st_size)})")
    if not interactive:
        print(f"    delete them from {MODELS_DIR} to free {_gb(total)}")
        return
    if not _ask(f"Delete them, freeing {_gb(total)}?", False):
        return
    for path in unused:
        path.unlink()
        verified.pop(path.name, None)
    _save_verified(verified)
    print("    deleted")


# --- Models ---

def _download_models(ask_again: bool) -> None:
    """Decide which models belong here, then download what's missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    verified = _load_verified()
    already_set_up = bool(verified)
    choices = _load_choices()
    interactive = sys.stdin.isatty()

    models = get_downloadable_models()
    drafter_size = {
        m["for_model"]: m["size"] or 0 for m in models if m["role"] == "draft"
    }

    wanted_ids: set[str] = set()
    declined_files: set[str] = set()
    needed = []

    for model in models:
        description = _describe(model)

        if model["role"] == "draft":
            # Follows its chat model, which was just decided and listed
            if not model["mandatory"] and model["for_model"] not in wanted_ids:
                if model["sha256"] is not None:
                    declined_files.add(model["file"])
                continue
        elif not model["mandatory"]:
            if model["sha256"] is None:
                print(f"    {description} — optional, not pinned yet")
                continue
            size = model["size"] + drafter_size.get(model["model_id"], 0)
            if not _wants(model, size, choices, ask_again, interactive):
                print(f"    {description} — optional, not downloaded")
                declined_files.add(model["file"])
                continue
            wanted_ids.add(model["model_id"])

        if _is_current(model, verified):
            status = "downloaded and verified" if model["sha256"] else "present"
            print(f"    {description} — {status}")
            continue

        if model["sha256"] is None:
            _fail(
                f"the {description} is required but isn't pinned.\n\n"
                f"Add its sha256 and size to the profile, with a url that "
                f"names a commit. 'python -m basic_bot.pin' prints them.\n"
                f"Then run 'python3 build.py' again."
            )

        print(f"    {description} — needs download ({_gb(model['size'])})")
        needed.append(model)

    named = {m["file"] for m in models}
    unused = sorted(
        p for p in MODELS_DIR.glob("*.gguf")
        if p.name not in named or p.name in declined_files
    )
    _offer_cleanup(unused, interactive, verified)

    if not needed:
        print("    nothing to download")
        return

    required = _remaining_bytes([m for m in needed if m["mandatory"]])
    if already_set_up and interactive and required > CONFIRM_BYTES:
        print()
        if not _ask(
            f"The required models need {_gb(required)} of new downloads. "
            f"Download them now?",
            True,
        ):
            sys.exit(
                "\nStopped before downloading. Run 'python3 build.py' "
                "again when you're ready."
            )

    remaining = _remaining_bytes(needed)
    free = shutil.disk_usage(MODELS_DIR).free
    if remaining + SPACE_MARGIN > free:
        _fail(
            f"not enough disk space. The models need {_gb(remaining)} "
            f"more, and {_gb(free)} is free.\n\n"
            f"Free up space, then run 'python3 build.py' again."
        )

    hosts = {urlparse(m["url"]).netloc for m in needed}
    source = " from Hugging Face" if hosts == {"huggingface.co"} else ""
    noun = "model" if len(needed) == 1 else "models"
    print()
    print(f"    {len(needed)} {noun} to download, {_gb(remaining)},{source}")

    for number, model in enumerate(needed, start=1):
        _download(model, number, len(needed), verified)


# --- Build ---

# Flags for every backend. By default, llama.cpp's build downloads
# llama-server's built-in web interface from Hugging Face, falling back
# to a moving "latest". Bountiful talks to llama-server's API and has
# its own interface, so the download is turned off. The build then
# warns "no assets available - building without an embedded UI",
# which is expected.
COMMON_FLAGS = ["-DLLAMA_USE_PREBUILT_UI=OFF"]


def _build_config(backend: str) -> tuple[list[str], str]:
    """cmake flags and backend name for this machine's backend.

    Metal is llama.cpp's default on macOS, so it needs no backend flag.
    """
    if backend == "metal":
        return COMMON_FLAGS, "Metal"
    return ["-DGGML_VULKAN=ON", *COMMON_FLAGS], "Vulkan"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 build.py",
        description="Build llama.cpp and download the AI models.",
    )
    parser.add_argument(
        "--models", action="store_true",
        help="ask again about every optional chat model",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    try:
        backend = platform_backend()
    except ProfileError as e:
        _fail(str(e))
    flags, backend_name = _build_config(backend)

    print(f"  checking build tools for {backend_name}")
    check_prerequisites(backend)

    print("  llama.cpp, the inference server")
    build(flags, backend_name)

    print("  choosing a hardware profile")
    try:
        machine, name = choose_profile()
    except ProfileError as e:
        _fail(str(e))
    print(f"    {backend_name}, {machine.memory_gb} GB for models")
    print(f"    profile: {name}")

    print("  AI models")
    _download_models(ask_again=args.models)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\n\nStopped. Partial downloads are kept.\n"
            "Run 'python3 build.py' again to resume."
        )
        sys.exit(130)
