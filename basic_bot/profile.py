"""Hardware profiles: choosing one when the machine is built, and
loading it when an agent opens.

Profiles ship with the engine package in basic_bot/profiles/. Each is
a TOML file describing the models, launch args, reasoning controls,
speculative decoding, sampling parameters, and UI metadata for one
kind of machine, and its [needs] table states what that machine must
have.

The profile is chosen once, by the build (python -m basic_bot), and
recorded in ~/.bountiful, which every agent on the machine shares.
Opening an agent reads that record and runs no detection, so it costs
nothing extra, the models on disk always match the profile in use,
and a hosted engine never detects anything.

Choosing works from facts about the machine. The GPU backend comes
from the operating system: Metal on macOS, Vulkan on Linux. The memory
models can use comes from sysctl on a Mac and, on Linux, from the
llama-server the build has just compiled. Selection takes the largest
profile for that backend that fits the memory. No code names a
machine, so a profile for new hardware needs no new code.

The profile is checked when it's chosen and again when it loads, so a
mistake in a hand-edited file is reported with a clear message instead
of failing in the middle of a conversation. Problems raise
ProfileError rather than exiting, so a web server or a hosted service
can report them.

Two gates determine whether a chat model appears in the UI:
it must be in the profile and downloaded to disk.
"""

import json
import logging
import platform
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from basic_bot.paths import MODELS_DIR, PROFILE_RECORD, SERVER_BIN

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"

SETUP = "Run 'python3 build.py' to set it up."


class ProfileError(RuntimeError):
    """No profile fits this machine, or a profile is malformed.

    The message is user-facing.
    """


@dataclass(frozen=True)
class Machine:
    """What detection found: the GPU backend and the memory for models."""
    backend: str
    memory_gb: int


_machine: Machine | None = None
_loaded: tuple[str, dict] | None = None


# --- Machine detection ---

def platform_backend() -> str:
    """The GPU backend for this operating system: metal or vulkan.

    Known before anything is built, so the build knows which llama.cpp
    to compile.
    """
    if sys.platform == "darwin":
        # Also catches Python running under Rosetta, which reports x86_64
        if platform.machine() != "arm64":
            raise ProfileError(
                "Bountiful needs a Mac with Apple Silicon, and a Python "
                "built for it. If this Mac has an M-series chip, Python "
                "may be running under Rosetta: install Python with "
                "Homebrew and run 'python3 build.py' again."
            )
        return "metal"
    if sys.platform == "linux":
        return "vulkan"
    raise ProfileError("Bountiful runs on an Apple Silicon Mac or on Linux.")


def detect_machine() -> Machine:
    """This machine's backend and model memory, detected once per process.

    On Linux this asks llama-server, so llama.cpp must be built first.
    """
    global _machine
    if _machine is None:
        backend = platform_backend()
        if backend == "metal":
            memory_gb = _mac_memory_gb()
        else:
            memory_gb = _vulkan_memory_gb()
        _machine = Machine(backend, memory_gb)
        logger.info(
            "Detected %s with %d GB for models",
            backend.capitalize(), memory_gb,
        )
    return _machine


def _mac_memory_gb() -> int:
    """A Mac's unified memory, in whole gigabytes."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // (1024 ** 3)
    except (ValueError, OSError):
        pass
    raise ProfileError("Couldn't read this Mac's memory size.")


# One line of `llama-server --list-devices`, as printed by llama.cpp's
# common_print_available_devices():
#   Vulkan0: NVIDIA GeForce RTX 3060 (12288 MiB, 11873 MiB free)
_DEVICE_LINE = re.compile(
    r"^\s*(Vulkan\d+): (.*) \((\d+) MiB, \d+ MiB free\)\s*$"
)


def _vulkan_memory_gb() -> int:
    """The largest Vulkan graphics card's memory, in whole gigabytes.

    Read from the llama-server the build compiled, so no other tool is
    needed. Cards report a little under their size, so this rounds.
    """
    if not SERVER_BIN.exists():
        raise ProfileError(
            "llama.cpp must be built before the graphics card can be "
            "detected. " + SETUP
        )
    try:
        result = subprocess.run(
            [str(SERVER_BIN), "--list-devices"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ProfileError(f"Couldn't list graphics cards: {e}") from e

    cards = []
    for line in result.stdout.splitlines():
        match = _DEVICE_LINE.match(line)
        if match:
            name, description, mib = match.groups()
            logger.info("Found %s: %s (%s MiB)", name, description, mib)
            cards.append(int(mib))

    if not cards:
        raise ProfileError(
            "No Vulkan graphics card was found. Check that its driver is "
            "installed: 'vulkaninfo --summary' should list it."
        )
    return round(max(cards) / 1024)


# --- Validation ---

PROFILE_FORMAT = 1
BACKENDS = ("metal", "vulkan")

_REQUIRED = {
    "embedding": ("file", "url", "ctx_size"),
    "summary": ("file", "url", "alias", "publisher", "family",
                "max_tokens", "sampling"),
    "chat": ("file", "url", "alias", "display_name", "publisher",
             "family", "max_tokens"),
}


def _raise_problems(path: Path, problems: list[str]) -> NoReturn:
    raise ProfileError(
        f"Problems in {path.name}:\n"
        + "\n".join(f"  - {p}" for p in problems)
    )


def _check_format(profile: dict) -> list[str]:
    """Check the profile's format version and its [needs] table."""
    problems = []

    fmt = profile.get("format")
    if not isinstance(fmt, int) or isinstance(fmt, bool):
        problems.append(f"missing format = {PROFILE_FORMAT} before the first table")
    elif fmt != PROFILE_FORMAT:
        problems.append(
            f"format {fmt} isn't one this engine reads "
            f"(it reads format {PROFILE_FORMAT})"
        )

    needs = profile.get("needs")
    if not isinstance(needs, dict):
        problems.append("missing [needs] table")
        return problems
    if needs.get("backend") not in BACKENDS:
        problems.append(f"needs.backend must be one of: {', '.join(BACKENDS)}")
    memory = needs.get("memory_gb")
    if not (isinstance(memory, int) and not isinstance(memory, bool)
            and memory > 0):
        problems.append("needs.memory_gb must be a positive whole number")

    return problems


def _check_reasoning(name: str, table: dict) -> list[str]:
    """Check one [reasoning] table against the rules in the profile header."""
    problems = []
    mode = table.get("mode")

    if mode not in ("optional", "always"):
        problems.append(f'{name}.reasoning: mode must be "optional" or "always"')

    if mode == "optional":
        if not isinstance(table.get("default"), bool):
            problems.append(
                f"{name}.reasoning: optional reasoning needs "
                f"default = true or false"
            )
        if not table.get("switch_key"):
            problems.append(
                f"{name}.reasoning: optional reasoning needs switch_key"
            )

    levels = table.get("effort_levels")
    if levels is not None:
        if not isinstance(levels, list) or not levels:
            problems.append(
                f"{name}.reasoning: effort_levels must be a non-empty list"
            )
        elif table.get("effort_default") not in levels:
            problems.append(
                f"{name}.reasoning: effort_default must be one of {levels}"
            )
        if not table.get("effort_key"):
            problems.append(f"{name}.reasoning: effort_levels needs effort_key")
    elif "effort_default" in table or "effort_key" in table:
        problems.append(
            f"{name}.reasoning: effort_default and effort_key "
            f"need effort_levels"
        )

    if table.get("effort_needs_reasoning") and mode != "optional":
        problems.append(
            f"{name}.reasoning: effort_needs_reasoning applies only "
            f"to optional reasoning"
        )

    return problems


def _check_pin(name: str, entry: dict) -> list[str]:
    """Check a pinned download: sha256, size, and a commit url, together."""
    sha256 = entry.get("sha256")
    size = entry.get("size")
    if sha256 is None and size is None:
        return []

    problems = []
    hex_digits = set("0123456789abcdef")
    if not (isinstance(sha256, str) and len(sha256) == 64
            and set(sha256) <= hex_digits):
        problems.append(f"{name}: sha256 must be 64 lowercase hex characters")
    if not (isinstance(size, int) and not isinstance(size, bool) and size > 0):
        problems.append(f"{name}: size must be a positive number of bytes")
    if "/resolve/main/" in entry.get("url", ""):
        problems.append(f"{name}: a pinned url must name a commit, not main")
    return problems


def _check_draft(name: str, table: dict) -> list[str]:
    """Check one [draft] table: a drafter file, a type, or both.

    type = "mtp" alone means drafting layers built into the model
    file. A file is a separate drafter, pinned like any model. Both
    together are a separate drafter whose type llama.cpp can't tell
    from the file, such as Gemma 4's MTP assistants.
    """
    draft_name = f"{name}.draft"

    if "type" not in table and "file" not in table:
        return [f'{draft_name}: needs type = "mtp", a file, or both']

    problems = []
    if "type" in table and table["type"] != "mtp":
        problems.append(f'{draft_name}: type must be "mtp"')
    if "file" in table:
        if "url" not in table:
            problems.append(f"{draft_name}: missing url")
        problems += _check_pin(draft_name, table)
    return problems


def _check_entry(name: str, entry: dict, required: tuple) -> list[str]:
    """Check one model entry: required keys, pin, reasoning, and draft tables."""
    problems = [f"{name}: missing {key}" for key in required if key not in entry]
    if "download" in entry:
        problems.append(
            f"{name}: download is no longer used. The embedding, summary, "
            f"and default chat models are always downloaded, and the build "
            f"offers the other chat models."
        )
    problems += _check_pin(name, entry)
    reasoning = entry.get("reasoning")
    if reasoning is not None:
        problems += _check_reasoning(name, reasoning)
    draft = entry.get("draft")
    if draft is not None:
        problems += _check_draft(name, draft)
    return problems


def _validate(profile: dict, profile_path: Path) -> None:
    """Raise ProfileError listing every problem, if the profile is malformed."""
    problems = _check_format(profile)

    for role in ("embedding", "summary"):
        if role not in profile:
            problems.append(f"missing [{role}] section")
        else:
            problems += _check_entry(role, profile[role], _REQUIRED[role])

    chat = profile.get("chat", {})
    if not chat:
        problems.append("no [chat] models defined")
    for model_id, entry in chat.items():
        problems += _check_entry(f'chat."{model_id}"', entry, _REQUIRED["chat"])
        if not isinstance(entry.get("default", False), bool):
            problems.append(f'chat."{model_id}": default must be true or false')

    # The default chat model is always downloaded, so there must be
    # exactly one, or which models are mandatory would be ambiguous.
    defaults = [m for m, e in chat.items() if e.get("default") is True]
    if chat and len(defaults) != 1:
        problems.append(
            f"exactly one chat model must have default = true "
            f"(found {len(defaults)})"
        )

    if problems:
        _raise_problems(profile_path, problems)


# --- Selection ---

def _read(path: Path) -> dict:
    """Parse one profile file."""
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ProfileError(f"Couldn't read {path.name}: {e}") from e


def _select(machine: Machine) -> tuple[str, dict]:
    """The largest profile for this machine's backend that fits its memory.

    Returns (name, profile). Only each profile's format and [needs] are
    checked here; the caller checks the chosen profile in full.
    """
    candidates = []
    for path in sorted(PROFILES_DIR.glob("*.toml")):
        profile = _read(path)
        problems = _check_format(profile)
        if problems:
            _raise_problems(path, problems)
        needs = profile["needs"]
        if needs["backend"] == machine.backend:
            candidates.append((needs["memory_gb"], path.stem, profile))

    backend = machine.backend.capitalize()
    if not candidates:
        raise ProfileError(f"No profile is available for {backend}.")

    fitting = [c for c in candidates if c[0] <= machine.memory_gb]
    if not fitting:
        smallest = min(c[0] for c in candidates)
        raise ProfileError(
            f"This machine has {machine.memory_gb} GB for models. "
            f"Bountiful needs at least {smallest} GB with {backend}: "
            f"its models don't fit in less."
        )

    memory_gb, name, profile = max(fitting, key=lambda c: c[0])
    if machine.memory_gb > memory_gb:
        logger.warning(
            "This machine has %d GB for models. Bountiful's largest %s "
            "profile, %s, is made for %d GB, so it will use that. It "
            "should work, but this configuration is untested.",
            machine.memory_gb, backend, name, memory_gb,
        )
    return name, profile


# --- Choosing and recording ---

def choose_profile() -> tuple[Machine, str]:
    """Detect the machine, choose its profile, and record the choice.

    Called by the build. Returns the machine and the profile's name.
    """
    global _loaded
    machine = detect_machine()
    name, profile = _select(machine)
    _validate(profile, PROFILES_DIR / f"{name}.toml")

    PROFILE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_RECORD.write_text(
        "# The hardware profile this machine uses, chosen by\n"
        "# 'python3 build.py'. Run it again to choose again.\n"
        f"profile = {json.dumps(name)}\n"
        f"backend = {json.dumps(machine.backend)}\n"
        f"memory_gb = {machine.memory_gb}\n"
    )
    logger.info("Chose hardware profile: %s", name)
    _loaded = (name, profile)
    return machine, name


def _recorded_name() -> str:
    """The profile the build chose for this machine."""
    if not PROFILE_RECORD.exists():
        raise ProfileError(
            "This machine hasn't been set up for Bountiful yet. " + SETUP
        )
    try:
        name = tomllib.loads(PROFILE_RECORD.read_text())["profile"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
        raise ProfileError(
            f"{PROFILE_RECORD} couldn't be read ({e}). " + SETUP
        ) from e
    if not isinstance(name, str):
        raise ProfileError(f"{PROFILE_RECORD} names no profile. " + SETUP)
    return name


# --- Loading ---

def _load() -> tuple[str, dict]:
    """Load and check the recorded profile, once per process."""
    global _loaded
    if _loaded is None:
        name = _recorded_name()
        path = PROFILES_DIR / f"{name}.toml"
        if not path.exists():
            raise ProfileError(
                f"This machine was set up with the {name} profile, which "
                f"this version of Bountiful doesn't have. Run "
                f"'python3 build.py' to choose again."
            )
        profile = _read(path)
        _validate(profile, path)
        logger.info("Loaded hardware profile: %s", name)
        _loaded = (name, profile)
    return _loaded


def get_profile() -> dict:
    """The profile in use."""
    return _load()[1]


def profile_name() -> str:
    """The name of the profile in use, such as metal_32gb."""
    return _load()[0]


# --- Profile helpers ---

def get_embedding_config() -> dict:
    """Embedding model configuration from the profile."""
    return get_profile()["embedding"]


def get_summary_config() -> dict:
    """Summary model configuration from the profile."""
    return get_profile()["summary"]


def get_chat_models() -> dict:
    """All chat model entries from the profile."""
    return get_profile().get("chat", {})


def get_default_chat_model() -> tuple[str, dict]:
    """The chat model marked default = true.

    The validator ensures there is exactly one. Returns (model_id,
    config).
    """
    for model_id, config in get_chat_models().items():
        if config.get("default") is True:
            return model_id, config
    raise ProfileError("The profile has no default chat model.")


def get_available_chat_models() -> dict:
    """Chat models that are in the profile and downloaded to disk.

    Profile is the menu. Filesystem is the filter.
    If the GGUF file exists, the model is available.
    """
    return {
        model_id: config
        for model_id, config in get_chat_models().items()
        if (MODELS_DIR / config["file"]).exists()
    }


def get_downloadable_models() -> list[dict]:
    """Every model file the profile names, and whether it's mandatory.

    The embedding model, the summary model, and the default chat model
    are mandatory: the build always downloads them. Every other chat
    model is optional, downloaded only if the user accepts it.

    Each dict has role, file, display_name, url, sha256, size,
    mandatory, and exists; sha256 and size are None for unpinned
    files. Chat models also carry model_id. A chat model's drafter
    file follows it, with role "draft" and for_model naming its model,
    and is mandatory exactly when its model is.
    """
    profile = get_profile()
    default_id, _ = get_default_chat_model()

    # (role, model_id, entry, mandatory)
    entries = [
        ("embedding", None, profile["embedding"], True),
        ("summary", None, profile["summary"], True),
    ]
    entries += [
        ("chat", model_id, config, model_id == default_id)
        for model_id, config in get_chat_models().items()
    ]

    models = []
    for role, model_id, entry, mandatory in entries:
        model = {
            "role": role,
            "file": entry["file"],
            "display_name": entry.get("display_name"),
            "url": entry["url"],
            "sha256": entry.get("sha256"),
            "size": entry.get("size"),
            "mandatory": mandatory,
            "exists": (MODELS_DIR / entry["file"]).exists(),
        }
        if model_id is not None:
            model["model_id"] = model_id
        models.append(model)

        draft = entry.get("draft") or {}
        if "file" in draft:
            models.append({
                "role": "draft",
                "file": draft["file"],
                "display_name": entry.get("display_name"),
                "url": draft["url"],
                "sha256": draft.get("sha256"),
                "size": draft.get("size"),
                "mandatory": mandatory,
                "exists": (MODELS_DIR / draft["file"]).exists(),
                "for_model": model_id,
            })

    return models
