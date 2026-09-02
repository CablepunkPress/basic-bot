"""Build Orchestrator.

Called by build.py in 'bountiful' repo.

Entry point for `python -m basic_bot.`.

Detects hardware, loads the matching profile, compiles llama.cpp
with the right flags, downloads the models the profile requires,
and enables default chat models.
"""

import sys
from pathlib import Path

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.profile import (
    detect_hardware,
    enable_default_models,
    get_downloadable_models,
    get_profile,
    MODELS_DIR,
)


def _download(name: str, url: str, dest: Path) -> None:
    """Download a model file if not already present."""
    import urllib.request

    if dest.exists():
        print(f"    {name}: already downloaded")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial")

    def progress(count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(count * block_size, total)
            percent = done * 100 // total
            mb = done // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            print(f"\r    {name}: {percent}% ({mb}/{total_mb} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, partial, reporthook=progress)
    except Exception as e:
        if partial.exists():
            partial.unlink()
        sys.exit(f"\nERROR: download failed: {e}\nURL: {url}")

    partial.rename(dest)
    print(f"\n    {name}: saved to {dest}")


def _download_models() -> None:
    """Download all models marked for download in the profile."""
    for model in get_downloadable_models():
        if model["download"] and not model["exists"]:
            label = model.get("model_id", model["role"])
            _download(label, model["url"], MODELS_DIR / model["file"])


def _build_flags() -> list[str]:
    """Determine cmake build flags from hardware detection."""
    hw = detect_hardware()

    if hw.startswith("nvidia"):
        return ["-DGGML_CUDA=ON"]
    if hw.startswith("apple"):
        return []  # Metal auto-detected by cmake

    return []  # CPU-only fallback


def main() -> None:
    print("  checking build tools")
    check_prerequisites()

    print("  detecting hardware")
    hw = detect_hardware()
    profile = get_profile()
    print(f"    hardware: {hw}")

    print("  llama.cpp")
    flags = _build_flags()
    build(flags)

    print("  models")
    _download_models()

    print("  enabling default models")
    enable_default_models()


if __name__ == "__main__":
    main()
