"""Model files: checking them and downloading them.

Shared by the build (python -m basic_bot), which keeps the models this
machine uses, and 'bountiful models', which offers the optional ones.

Users see models, not files. A chat model's drafter is a separate
file because that's how model makers publish it, but it belongs to its
model: it's counted in the model's size and downloaded as "file 2 of 2"
of that model.

Pinned files, those with sha256 and size in the profile, are checked
against their checksum and downloaded again when the pin changes. An
interrupted download keeps its partial file and resumes on the next
run, retrying on its own after a dropped connection.
"""

import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

from basic_bot.paths import MODELS_DIR
from basic_bot.profile import get_downloadable_models

# Which files have passed their checksum, so a rebuild doesn't rehash
# 40 GB. Only a cache: the checksum in the profile is the truth.
VERIFIED_FILE = MODELS_DIR / ".verified.json"

CHUNK = 1024 * 1024
TIMEOUT = 60          # seconds without data before a stalled download fails
SPACE_MARGIN = 2 * 1024 ** 3
PROGRESS_INTERVAL = 0.5

# Waits between retries after a stalled or dropped connection. The
# count resets whenever a retry makes progress, so a long download
# survives any number of separate blips.
RETRY_WAITS = [10, 30, 60, 120, 300]

# On a machine that's already set up, downloads larger than this ask
# first, as when an update makes a new chat model the default.
CONFIRM_BYTES = 5 * 1000 ** 3

ROLE_NAMES = {
    "embedding": "embedding model",
    "summary": "summary model",
    "chat": "chat model",
}


def fail(message: str) -> NoReturn:
    sys.exit(f"\nERROR: {message}")


def ask(question: str, default: bool, indent: str = "    ") -> bool:
    """A yes-or-no question, answered at the keyboard."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{indent}{question} {hint} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


# --- Numbers and names ---

def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def _source(url: str) -> str:
    """Where a file comes from, e.g. 'Hugging Face, Abiray/Nemotron-...'."""
    parts = urlparse(url)
    if parts.netloc == "huggingface.co":
        segments = parts.path.strip("/").split("/")
        if len(segments) >= 2:
            return f"Hugging Face, {segments[0]}/{segments[1]}"
        return "Hugging Face"
    return parts.netloc or url


# --- Models ---

@dataclass
class Model:
    """One model as the user sees it: its own file, then its drafter's.
    
    model_id is the chat model's id in the profile, or the role
    ("embedding", "summary") for the two models that have none.
    """
    role: str
    model_id: str
    name: str
    mandatory: bool
    files: list[dict]

    @property
    def label(self) -> str:
        return ROLE_NAMES[self.role]

    @property
    def size(self) -> int:
        return sum(f["size"] or 0 for f in self.files)

    @property
    def pinned(self) -> bool:
        return all(f["sha256"] is not None for f in self.files)

    @property
    def on_disk(self) -> bool:
        """True if the model's own file is here, finished."""
        return (MODELS_DIR / self.files[0]["file"]).exists()


def profile_models() -> list[Model]:
    """Every model the profile names, each with its drafter's file."""
    models: list[Model] = []
    for entry in get_downloadable_models():
        if entry["role"] == "draft":
            # The profile lists a drafter right after its chat model
            owner = next(m for m in models if m.model_id == entry["for_model"])
            owner.files.append(entry)
            continue
        models.append(Model(
            role=entry["role"],
            model_id=entry.get("model_id") or entry["role"],
            name=entry.get("display_name") or entry["file"],
            mandatory=entry["mandatory"],
            files=[entry],
        ))
    return models


# --- Verification ---

def load_verified() -> dict:
    try:
        return json.loads(VERIFIED_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_verified(verified: dict) -> None:
    VERIFIED_FILE.write_text(json.dumps(verified, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK * 8):
            digest.update(block)
    return digest.hexdigest()


def _is_current(file: dict, verified: dict, name: str, checked: list) -> bool:
    """True if the file on disk is the one the profile asks for.

    A file that has to be hashed is announced and added to checked.
    """
    dest = MODELS_DIR / file["file"]
    if not dest.exists():
        return False

    sha256 = file["sha256"]
    if sha256 is None:
        return True   # unpinned: presence is all that can be checked

    if dest.stat().st_size != file["size"]:
        return False
    if verified.get(file["file"]) == sha256:
        return True

    # Present but never verified, e.g. after the models folder was
    # moved here from another machine. This takes a minute or two.
    print(f"    checking {name}, {file['file']}...", end="", flush=True)
    checked.append(file["file"])
    matches = _sha256(dest) == sha256
    print(" done" if matches else " it has changed")
    if matches:
        verified[file["file"]] = sha256
        save_verified(verified)
    return matches


# --- Downloading ---

def _partial_path(file: dict) -> Path:
    dest = MODELS_DIR / file["file"]
    return dest.with_name(dest.name + ".partial")


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _remaining_bytes(files: list[dict]) -> int:
    """Bytes still to download, counting any partial files already there."""
    return sum(
        max(f["size"] - _size(_partial_path(f)), 0) for f in files
    )


def _fetch(url: str, partial: Path, size: int | None) -> None:
    """Download url into partial, resuming from any bytes already there."""
    have = _size(partial)
    if size is not None and have > size:
        partial.unlink()
        have = 0
    if size is not None and have == size:
        return

    request = urllib.request.Request(url)
    if have:
        print(f"      resuming at {gb(have)}")
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
        line = f"      {done * 100 // total}% ({gb(done)} of {gb(total)})"
    else:
        line = f"      {gb(done)}"

    # Rate and time left, once there's enough to measure
    if elapsed >= 5 and session > 0:
        rate = session / elapsed
        line += f" — {rate / 1e6:.1f} MB/s"
        if total and done < total:
            line += f", about {_duration((total - done) / rate)} left"

    print("\r" + line.ljust(76), end="", flush=True)


def _download(model: Model, file: dict, verified: dict, rerun: str) -> None:
    """Download one file of a model, retrying, then check its checksum."""
    dest = MODELS_DIR / file["file"]
    partial = _partial_path(file)
    which = f"the {model.label}, {model.name}"

    heading = f"    downloading {which}"
    if len(model.files) > 1:
        heading += f" (file {model.files.index(file) + 1} of {len(model.files)})"
    print()
    print(heading)
    print(f"      from {_source(file['url'])}")

    retries = 0
    while True:
        before = _size(partial)
        error = None
        try:
            _fetch(file["url"], partial, file["size"])
        except urllib.error.HTTPError as e:
            if e.code < 500:
                # The server refused the request itself, e.g. a missing
                # file. Waiting won't change that.
                print()
                fail(f"the download of {which} failed: {e}\n\nURL: {file['url']}")
            print()
            error = f"connection lost ({e})"
        except Exception as e:
            print()
            error = f"connection lost ({e})"

        after = _size(partial)
        complete = file["size"] is None or after == file["size"]
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
            fail(
                f"the download of {which} stopped: {error}\n\n"
                f"The partial file is kept. {rerun} to resume."
            )

        wait = RETRY_WAITS[retries]
        retries += 1
        print(f"      {error}; retrying in {wait} seconds")
        time.sleep(wait)

    if file["sha256"] is not None:
        print("      verifying checksum...", end="", flush=True)
        if _sha256(partial) != file["sha256"]:
            print()
            partial.unlink()
            fail(
                f"{which} failed its checksum, so the download was "
                f"deleted. {rerun} to download it fresh."
            )
        print(" done")

    partial.replace(dest)
    if file["sha256"] is not None:
        verified[file["file"]] = file["sha256"]
        save_verified(verified)


# --- Installing ---

def _table(rows: list[tuple[Model, str]]) -> None:
    width = max(len(model.name) for model, _ in rows)
    for model, status in rows:
        print(
            f"    {model.label.capitalize():<16} {model.name:<{width}}  "
            f"{gb(model.size):>8}  {status}"
        )


def install(models: list[Model], rerun: str, confirm: bool = False) -> None:
    """Show the models as a table, then download whatever isn't current.

    rerun is how to carry on later, as in "Run 'python3 build.py'
    again". With confirm, downloads larger than CONFIRM_BYTES ask first.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    verified = load_verified()

    rows: list[tuple[Model, str]] = []
    needed: list[tuple[Model, dict]] = []
    checked: list[str] = []
    for model in models:
        stale = [
            f for f in model.files
            if not _is_current(f, verified, model.name, checked)
        ]
        for file in stale:
            if file["sha256"] is None:
                fail(
                    f"the {model.label}, {model.name}, is needed but isn't "
                    f"pinned.\n\nAdd its sha256 and size to the profile, with "
                    f"a url that names a commit. 'python -m basic_bot.pin' "
                    f"prints them."
                )
            needed.append((model, file))
        if stale:
            rows.append((model, "to download"))
        else:
            rows.append((model, "verified" if model.pinned else "present"))

    if checked:
        print()
    if rows:
        _table(rows)
    if not needed:
        print("\n    nothing to download")
        return

    files = [file for _, file in needed]
    remaining = _remaining_bytes(files)
    count = len({id(model) for model, _ in needed})
    noun = "model" if count == 1 else "models"
    hosts = {urlparse(f["url"]).netloc for f in files}
    source = ", from Hugging Face" if hosts == {"huggingface.co"} else ""
    print(f"\n    {count} {noun} to download, {gb(remaining)}{source}")

    if confirm and remaining > CONFIRM_BYTES and sys.stdin.isatty():
        if not ask(f"That's {gb(remaining)} of new downloads. Download now?", True):
            sys.exit(f"\nStopped before downloading. {rerun} when you're ready.")

    free = shutil.disk_usage(MODELS_DIR).free
    if remaining + SPACE_MARGIN > free:
        fail(
            f"not enough disk space. The models need {gb(remaining)} more, "
            f"with {gb(SPACE_MARGIN)} to spare, and {gb(free)} is free.\n\n"
            f"Free up space, then {rerun[0].lower()}{rerun[1:]}."
        )

    for model, file in needed:
        _download(model, file, verified, rerun)
