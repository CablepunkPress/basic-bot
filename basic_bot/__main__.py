"""Build orchestrator.

Called by build.py in the bountiful repo, as `python -m basic_bot`.

Detects hardware, checks the build tools for it, compiles the pinned
llama.cpp with the right backend, and downloads the models the
hardware profile requires.

Pinned models, those with sha256 and size in the profile, are checked
against their checksum and downloaded again when the pin changes. An
interrupted download keeps its partial file and resumes on the next
run. Unpinned models are accepted if present, but never downloaded.
"""

import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.profile import (
    detect_hardware,
    get_downloadable_models,
    MODELS_DIR,
)

# Which files have passed their checksum, so a rebuild doesn't rehash
# 40GB. Only a cache: the checksum in the profile is the truth.
VERIFIED_FILE = MODELS_DIR / ".verified.json"

CHUNK = 1024 * 1024
TIMEOUT = 60          # seconds without data before a stalled download fails
SPACE_MARGIN = 2 * 1024 ** 3
PROGRESS_INTERVAL = 0.5

ROLE_NAMES = {
    "embedding": "embedding model",
    "summary": "summary model",
    "chat": "chat model",
}


def _fail(message: str) -> None:
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

    try:
        _fetch(model["url"], partial, model["size"])
    except Exception as e:
        _fail(
            f"the download of the {_describe(model)} stopped: {e}\n\n"
            f"The partial file is kept.\n"
            f"Run 'python3 build.py' again to resume."
        )

    if model["size"] is not None and partial.stat().st_size != model["size"]:
        _fail(
            f"the {_describe(model)} is incomplete.\n\n"
            f"Run 'python3 build.py' again to resume."
        )

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


def _download_models() -> None:
    """List every model's status, then download what's missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    verified = _load_verified()

    needed = []
    for model in get_downloadable_models():
        description = _describe(model)

        if not model["download"]:
            print(f"    {description} — optional, not downloaded")
            continue

        if _is_current(model, verified):
            status = "downloaded and verified" if model["sha256"] else "present"
            print(f"    {description} — {status}")
            continue

        if model["sha256"] is None:
            _fail(
                f"the {description} is marked for download but isn't "
                f"pinned.\n\n"
                f"Add its sha256 and size to the profile, with a url that "
                f"names a commit, or set download = false.\n"
                f"Then run 'python3 build.py' again."
            )

        print(f"    {description} — needs download ({_gb(model['size'])})")
        needed.append(model)

    if not needed:
        print("    nothing to download")
        return

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


def _build_config(hardware: str) -> tuple[list[str], str]:
    """cmake flags and backend name for the detected hardware.

    Metal is llama.cpp's default on macOS, so Apple Silicon needs no
    backend flag. Detection has already stopped on unsupported hardware.
    """
    if hardware.startswith("apple"):
        return COMMON_FLAGS, "Metal"
    return ["-DGGML_CUDA=ON", *COMMON_FLAGS], "CUDA"


def main() -> None:
    print("  detecting hardware")
    hw = detect_hardware()
    print(f"    hardware: {hw}")

    print("  checking build tools")
    check_prerequisites(hw)

    print("  llama.cpp, the inference server")
    flags, backend = _build_config(hw)
    build(flags, backend)

    print("  AI models")
    _download_models()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\n\nStopped. Partial downloads are kept.\n"
            "Run 'python3 build.py' again to resume."
        )
        sys.exit(130)
