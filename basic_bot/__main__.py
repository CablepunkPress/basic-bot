"""Build orchestrator.

Called by build.py in the bountiful repo, as `python -m basic_bot`.

Detects hardware, checks the build tools for it, compiles the pinned
llama.cpp with the right backend, and downloads the models the
hardware profile requires.
"""

import sys
from pathlib import Path

from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.profile import (
    detect_hardware,
    get_downloadable_models,
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
