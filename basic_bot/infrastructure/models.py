"""Download and manage inference models.

Models are stored at ~/.bountiful/models/, shared by all agents
on the machine.
"""

import sys
import urllib.request
from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"

MODEL_NAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
MODEL_FILE = MODELS_DIR / MODEL_NAME
MODEL_URL = (
    "https://huggingface.co/CablepunkPress/Qwen3-Embedding-0.6B-GGUF"
    f"/resolve/main/{MODEL_NAME}"
)


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def download() -> None:
    """Download the embedding model if not already present."""
    if MODEL_FILE.exists():
        print(f"    already done — {MODEL_FILE} exists")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = MODEL_FILE.with_suffix(".partial")

    def report(count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(count * block_size, total)
            percent = done * 100 // total
            mb = done // (1024 * 1024)
            print(f"\r    {percent}% ({mb} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, partial, reporthook=report)
    except Exception as e:
        if partial.exists():
            partial.unlink()
        _fail(f"model download failed: {e}\nURL: {MODEL_URL}")

    partial.rename(MODEL_FILE)
    print(f"\n    saved to {MODEL_FILE}")
