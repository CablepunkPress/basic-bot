"""Hardware profile loader.

Detects the machine once, chooses the profile that fits it, loads and
checks that profile once, and caches the result for the lifetime of
the process. Every module that needs profile data imports from here.

Profiles ship with the engine package in basic_bot/profiles/. Each is
a TOML file describing the models, launch args, reasoning controls,
speculative decoding, sampling parameters, and UI metadata for one
kind of machine, and its [needs] table states what that machine must
have.

Detection reports facts about the machine: the GPU backend, which
comes from the operating system, and the memory models can use.
Selection takes the largest profile for that backend that fits the
memory. No code names a machine, so a profile for new hardware needs
no new code.

The profile is checked when it loads, so a mistake in a hand-edited
file is reported with a clear message instead of failing in the
middle of a conversation. Problems raise ProfileError rather than
exiting, so a web server or a hosted service can report them.

Two gates determine whether a chat model appears in the UI:
it must be in the profile and downloaded to disk.
"""

import logging
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from basic_bot.paths import MODELS_DIR

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"


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

def detect_machine() -> Machine:
    """This machine's backend and model memory, detected once per process."""
    global _machine
    if _machine is None:
        if sys.platform == "darwin":
            _machine = Machine("metal", _mac_memory_gb())
        elif sys.platform == "linux":
            _machine = Machine("vulkan", _vulkan_memory_gb())
        else:
            raise ProfileError(
                "Bountiful runs on an Apple Silicon Mac or on Linux."
            )
        logger.info(
            "Detected %s with %d GB for models",
            _machine.backend.capitalize(), _machine.memory_gb,
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


def _vulkan_memory_gb() -> int:
    """A graphics card's memory, read through llama.cpp's Vulkan build.

    Arrives with the Vulkan build, where it can be tested on Linux.
    """
    raise ProfileError(
        "Bountiful on Linux is moving to Vulkan and doesn't run there yet."
    )


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
    """Check one [draft] table: built-in drafting layers, or a drafter file."""
    draft_name = f"{name}.draft"

    if "type" not in table and "file" not in table:
        return [f'{draft_name}: needs type = "mtp" or a file']

    if "type" in table:
        problems = []
        if table["type"] != "mtp":
            problems.append(f'{draft_name}: type must be "mtp"')
        if "file" in table:
            problems.append(f"{draft_name}: use either type or file, not both")
        return problems

    problems = [] if "url" in table else [f"{draft_name}: missing url"]
    problems += _check_pin(draft_name, table)
    return problems


def _check_entry(name: str, entry: dict, required: tuple) -> list[str]:
    """Check one model entry: required keys, pin, reasoning, and draft tables."""
    problems = [f"{name}: missing {key}" for key in required if key not in entry]
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
    checked here; the chosen profile is checked in full when it loads.
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


# --- Loading ---

def _load() -> tuple[str, dict]:
    """Choose, load, and check the profile, once per process."""
    global _loaded
    if _loaded is None:
        name, profile = _select(detect_machine())
        _validate(profile, PROFILES_DIR / f"{name}.toml")
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

    Returns (model_id, config) tuple.
    """
    for model_id, config in get_chat_models().items():
        if config.get("default", False):
            return model_id, config

    # No default marked — use the first one
    models = get_chat_models()
    if models:
        first = next(iter(models))
        return first, models[first]

    raise ProfileError("No chat models are defined in the profile.")


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
    """All model files across all roles, with download status.

    Returns a list of dicts with role, file, url, sha256, size,
    download flag, and whether the file exists on disk. Chat models
    also carry model_id and default. A chat model's drafter file
    follows it, with role "draft", for_model naming its model, and
    the same download flag. sha256 and size are None for unpinned
    files.
    """
    profile = get_profile()

    # (role, model_id, entry, download by default)
    entries = [
        ("embedding", None, profile["embedding"], True),
        ("summary", None, profile["summary"], True),
    ]
    entries += [
        ("chat", model_id, config, False)
        for model_id, config in get_chat_models().items()
    ]

    models = []
    for role, model_id, entry, download_default in entries:
        download = entry.get("download", download_default)
        model = {
            "role": role,
            "file": entry["file"],
            "display_name": entry.get("display_name"),
            "url": entry["url"],
            "sha256": entry.get("sha256"),
            "size": entry.get("size"),
            "download": download,
            "exists": (MODELS_DIR / entry["file"]).exists(),
        }
        if model_id is not None:
            model["model_id"] = model_id
            model["default"] = entry.get("default", False)
        models.append(model)

        # A drafter file downloads whenever its model does
        draft = entry.get("draft") or {}
        if "file" in draft:
            models.append({
                "role": "draft",
                "file": draft["file"],
                "display_name": entry.get("display_name"),
                "url": draft["url"],
                "sha256": draft.get("sha256"),
                "size": draft.get("size"),
                "download": download,
                "exists": (MODELS_DIR / draft["file"]).exists(),
                "for_model": model_id,
            })

    return models
