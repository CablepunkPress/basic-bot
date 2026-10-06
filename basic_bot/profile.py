"""Hardware profile loader.

Detects hardware once, loads the matching profile TOML once,
caches the result for the lifetime of the process. Every module
that needs profile data imports from here.

Profiles ship with the engine package in basic_bot/profiles/.
Each profile is a TOML file describing the models, launch args,
reasoning controls, speculative decoding, sampling parameters, and
UI metadata for a specific hardware target.

The profile is checked when it loads, so a mistake in a hand-edited
file stops startup with a clear message instead of failing in the
middle of a conversation.

Two gates determine whether a chat model appears in the UI:
it must be in the profile and downloaded to disk.
"""

import logging
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

_hardware: str | None = None
_profile: dict | None = None

PROFILES_DIR = Path(__file__).parent / "profiles"
MODELS_DIR = Path.home() / ".bountiful" / "models"

# The only Apple memory size that has been tested.
APPLE_TESTED_GB = 32


# --- Hardware detection ---

def detect_hardware() -> str:
    """Detect GPU hardware. Returns a profile name.

    Results are cached — detection runs once per process.
    """
    global _hardware
    if _hardware is not None:
        return _hardware

    if sys.platform == "darwin":
        _hardware = _detect_apple()
        return _hardware

    if shutil.which("nvidia-smi"):
        _hardware = _detect_nvidia()
        return _hardware

    sys.exit(
        "No supported GPU detected.\n"
        "Bountiful runs on an Apple Silicon Mac with 32GB of memory."
    )


def _detect_nvidia() -> str:
    """Identify NVIDIA GPU by VRAM and return a profile name.

    The 12GB card was the engine's prototype hardware. It still works,
    but the Mac is the supported platform.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit("nvidia-smi failed. Check your GPU drivers.")

    lines = result.stdout.strip().split("\n")
    name, vram_str = lines[0].split(",")
    name = name.strip()
    vram_mb = int(vram_str.strip())
    vram_gb = vram_mb // 1024

    logger.info("Detected: %s (%d GB VRAM)", name, vram_gb)

    if vram_gb < 12:
        sys.exit(
            f"GPU has {vram_gb}GB VRAM. Bountiful requires at least 12GB."
        )

    logger.warning(
        "%s detected. This 12GB card was Bountiful's prototype hardware. "
        "It works, but its 9B chat model isn't suited to real daily use. "
        "The supported platform is an Apple Silicon Mac with 32GB.",
        name,
    )
    return "nvidia_12gb"


def _detect_apple() -> str:
    """Identify Apple Silicon memory and return a profile name.

    Only 32GB is tested. Larger Macs use the 32GB profile with a
    warning; smaller ones stop, because the models don't fit.
    """
    ram_gb = None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            ram_gb = int(result.stdout.strip()) // (1024 ** 3)
    except (ValueError, OSError):
        pass

    if ram_gb is None:
        logger.warning(
            "Could not read this Mac's memory size. Using the 32GB "
            "profile, the only one tested."
        )
        return "apple_32gb"

    logger.info("Detected Apple Silicon with %d GB memory", ram_gb)

    if ram_gb < APPLE_TESTED_GB:
        sys.exit(
            f"This Mac has {ram_gb}GB of memory. Bountiful needs 32GB: "
            f"its models don't fit in less."
        )

    if ram_gb > APPLE_TESTED_GB:
        logger.warning(
            "This Mac has %dGB of memory. Bountiful is tested only on "
            "32GB Macs, so it will use the 32GB profile. It should work, "
            "but this configuration is untested.",
            ram_gb,
        )

    return "apple_32gb"


# --- Validation ---

_REQUIRED = {
    "embedding": ("file", "url", "ctx_size"),
    "summary": ("file", "url", "alias", "provider", "family",
                "max_tokens", "sampling"),
    "chat": ("file", "url", "alias", "display_name", "provider",
             "family", "max_tokens"),
}


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
    """Stop with a list of problems if the profile is malformed."""
    problems = []

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
        sys.exit(
            f"Problems in {profile_path.name}:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


# --- Loading ---

def get_profile() -> dict:
    """Load, check, and cache the hardware profile.

    The profile TOML is read from the engine package directory.
    Detection, parsing, and checking happen once per process.
    """
    global _profile
    if _profile is not None:
        return _profile

    hw = detect_hardware()
    profile_path = PROFILES_DIR / f"{hw}.toml"

    if not profile_path.exists():
        sys.exit(
            f"No profile found for hardware '{hw}' at {profile_path}\n"
            f"Available profiles: "
            f"{[p.stem for p in PROFILES_DIR.glob('*.toml')]}"
        )

    profile = tomllib.loads(profile_path.read_text())
    _validate(profile, profile_path)

    _profile = profile
    logger.info("Loaded hardware profile: %s", hw)
    return _profile


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

    sys.exit("No chat models defined in profile.")


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
