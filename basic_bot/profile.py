"""Hardware profile loader.

Detects hardware once, loads the matching profile TOML once,
caches the result for the lifetime of the process. Every module
that needs profile data imports from here.

Profiles ship with the engine package in basic_bot/profiles/.
Each profile is a TOML file describing the models, launch args,
sampling parameters, and UI metadata for a specific hardware
target.
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
        "Bountiful requires an NVIDIA GPU (Linux) or Apple Silicon (macOS)."
    )


def _detect_nvidia() -> str:
    """Identify the NVIDIA GPU and return a profile name."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit("nvidia-smi failed. Check your GPU drivers.")

    line = result.stdout.strip()
    logger.info("Detected GPU: %s", line)

    if "3060" in line:
        return "nvidia_3060"
    # Future: "5090", "4090", etc.

    # Unknown NVIDIA — fall back to the most constrained profile
    logger.warning(
        "No exact profile for this GPU. Using nvidia_3060 as fallback."
    )
    return "nvidia_3060"


def _detect_apple() -> str:
    """Identify Apple Silicon and RAM, return a profile name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            ram_bytes = int(result.stdout.strip())
            ram_gb = ram_bytes // (1024 ** 3)
            logger.info("Detected Apple Silicon with %d GB RAM", ram_gb)

            if ram_gb >= 48:
                return "apple_m_48gb"
            return "apple_m_32gb"
    except (ValueError, OSError):
        pass

    logger.warning("Could not detect RAM size. Using apple_m_32gb.")
    return "apple_m_32gb"


def get_profile() -> dict:
    """Load and cache the hardware profile.

    The profile TOML is read from the engine package directory.
    Detection and parsing happen once per process.
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

    _profile = tomllib.loads(profile_path.read_text())
    logger.info("Loaded hardware profile: %s", hw)
    return _profile


# --- Helpers ---

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
    """Chat models whose GGUF files exist on disk.

    The profile is the menu. The filesystem is the filter.
    Only models that have been downloaded appear in the UI.
    """
    return {
        model_id: config
        for model_id, config in get_chat_models().items()
        if (MODELS_DIR / config["file"]).exists()
    }


def get_downloadable_models() -> list[dict]:
    """All models across all roles, with download status.

    Returns a list of dicts with role, model_id, file, url,
    download flag, and whether the file exists on disk.
    """
    profile = get_profile()
    models = []

    # Embedding
    emb = profile["embedding"]
    models.append({
        "role": "embedding",
        "file": emb["file"],
        "url": emb["url"],
        "download": emb.get("download", True),
        "exists": (MODELS_DIR / emb["file"]).exists(),
    })

    # Summary
    summ = profile["summary"]
    models.append({
        "role": "summary",
        "file": summ["file"],
        "url": summ["url"],
        "download": summ.get("download", True),
        "exists": (MODELS_DIR / summ["file"]).exists(),
    })

    # Chat models
    for model_id, config in get_chat_models().items():
        models.append({
            "role": "chat",
            "model_id": model_id,
            "file": config["file"],
            "url": config["url"],
            "download": config.get("download", False),
            "default": config.get("default", False),
            "exists": (MODELS_DIR / config["file"]).exists(),
        })

    return models
