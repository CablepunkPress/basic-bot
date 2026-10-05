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
import urllib.request
from pathlib import Path

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


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# --- Verification ---

def _load_verified() -> dict:
    try:
        return json.loads(VERIFIED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_verified(verified: dict) -> None:
    VERIFIED_FILE.write_text(json.dumps(verified, indent=2) + "\n")


def _sha256(path: Path, label: str) -> str:
    print(f"    {label}: verifying checksum...", end="", flush=True)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK * 8):
            digest.update(block)
    print(" done")
    return digest.hexdigest()


def _is_current(model: dict, label: str, verified: dict) -> bool:
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
    if _sha256(dest, label) == sha256:
        verified[model["file"]] = sha256
        _save_verified(verified)
        return True
    return False


# --- Downloading ---

def _partial_path(model: dict) -> Path:
    dest = MODELS_DIR / model["file"]
    return dest.with_name(dest.name + ".partial")


def _progress(label: str, done: int, total: int | None) -> None:
    if total:
        percent = done * 100 // total
        print(
            f"\r    {label}: {percent}% ({_gb(done)} of {_gb(total)})",
            end="", flush=True,
        )
    else:
        print(f"\r    {label}: {_gb(done)}", end="", flush=True)


def _fetch(label: str, url: str, partial: Path, size: int | None) -> None:
    """Download url into partial, resuming from any bytes already there."""
    have = partial.stat().st_size if partial.exists() else 0
    if size is not None and have > size:
        partial.unlink()
        have = 0
    if size is not None and have == size:
        return

    request = urllib.request.Request(url)
    if have:
        print(f"    {label}: resuming at {_gb(have)}")
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if have and response.status != 206:
            print(f"    {label}: server can't resume, starting over")
            have = 0

        total = size
        if total is None:
            length = response.headers.get("Content-Length")
            total = have + int(length) if length else None

        done = have
        with open(partial, "ab" if have else "wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                _progress(label, done, total)
    print()


def _download(model: dict, label: str, verified: dict) -> None:
    dest = MODELS_DIR / model["file"]
    partial = _partial_path(model)

    try:
        _fetch(label, model["url"], partial, model["size"])
    except Exception as e:
        _fail(
            f"download of {label} stopped: {e}\n"
            f"The partial file is kept. Run 'python3 build.py' again "
            f"to resume.\nURL: {model['url']}"
        )

    if model["size"] is not None and partial.stat().st_size != model["size"]:
        _fail(
            f"{label} is incomplete. Run 'python3 build.py' again to resume."
        )

    if model["sha256"] is not None:
        if _sha256(partial, label) != model["sha256"]:
            partial.unlink()
            _fail(
                f"{label} failed its checksum and was deleted. "
                f"Run 'python3 build.py' again to download it fresh."
            )

    partial.replace(dest)
    if model["sha256"] is not None:
        verified[model["file"]] = model["sha256"]
        _save_verified(verified)
    print(f"    {label}: saved to {dest}")


def _check_space(needed: list[dict]) -> None:
    """Stop before downloading if the disk can't hold what's needed."""
    remaining = 0
    for model in needed:
        partial = _partial_path(model)
        have = partial.stat().st_size if partial.exists() else 0
        remaining += max(model["size"] - have, 0)

    free = shutil.disk_usage(MODELS_DIR).free
    if remaining + SPACE_MARGIN > free:
        _fail(
            f"not enough disk space. The models need {_gb(remaining)} "
            f"more, and {_gb(free)} is free."
        )
    if remaining:
        print(f"    {_gb(remaining)} to download")


def _download_models() -> None:
    """Download and verify every model the profile marks for download."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    verified = _load_verified()

    needed = []
    for model in get_downloadable_models():
        if not model["download"]:
            continue
        label = model.get("model_id", model["role"])

        if _is_current(model, label, verified):
            print(f"    {label}: already downloaded")
            continue

        if model["sha256"] is None:
            _fail(
                f"{label} is marked for download but isn't pinned.\n"
                f"Add its sha256 and size to the profile, with a url that "
                f"names a commit, or set download = false."
            )
        needed.append((model, label))

    if not needed:
        return

    _check_space([model for model, _ in needed])
    for model, label in needed:
        _download(model, label, verified)


# --- Build ---

def _build_config(hardware: str) -> tuple[list[str], str]:
    """cmake flags and backend name for the detected hardware.

    Metal is llama.cpp's default on macOS, so Apple Silicon needs no
    flags. Detection has already stopped on unsupported hardware.
    """
    if hardware.startswith("apple"):
        return [], "Metal"
    return ["-DGGML_CUDA=ON"], "CUDA"


def main() -> None:
    print("  detecting hardware")
    hw = detect_hardware()
    print(f"    hardware: {hw}")

    print("  checking build tools")
    check_prerequisites(hw)

    print("  llama.cpp")
    flags, backend = _build_config(hw)
    build(flags, backend)

    print("  models")
    _download_models()


if __name__ == "__main__":
    main()
