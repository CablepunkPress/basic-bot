"""Download and manage models.

Models are stored at ~/.bountiful/models/, shared by all agents
on the machine. Three roles: embedding (CPU), summary (GPU),
and chat (GPU). Each role has one model, selected per hardware
profile.

Hardware profile: NVIDIA RTX 3060 12GB + 32GB DDR4
"""

import sys
import urllib.request
from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"


# --- Embedding model ---
# Qwen3-Embedding-0.6B Q8_0, self-hosted with EOS token fix
# CPU-only, ~639MB

EMBEDDING_MODEL_NAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBEDDING_MODEL_FILE = MODELS_DIR / EMBEDDING_MODEL_NAME
EMBEDDING_MODEL_URL = (
    "https://huggingface.co/CablepunkPress/Qwen3-Embedding-0.6B-GGUF"
    f"/resolve/main/{EMBEDDING_MODEL_NAME}"
)

SUMMARY_MODEL_NAME = "Qwen3-8B-Q8_0.gguf"
SUMMARY_MODEL_FILE = MODELS_DIR / SUMMARY_MODEL_NAME
SUMMARY_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF"
    f"/resolve/main/{SUMMARY_MODEL_NAME}"
)

CHAT_MODEL_NAME = "Qwen3-8B-Q8_0.gguf"
CHAT_MODEL_FILE = MODELS_DIR / CHAT_MODEL_NAME
CHAT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF"
    f"/resolve/main/{CHAT_MODEL_NAME}"
)


# --- Download utilities ---

def _progress(label: str):
    """Return a reporthook that prints download progress."""
    def report(count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(count * block_size, total)
            percent = done * 100 // total
            mb = done // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            print(f"\r    {label}: {percent}% ({mb}/{total_mb} MB)", end="", flush=True)
    return report


def _download(name: str, url: str, dest: Path) -> None:
    """Download a model file if not already present."""
    if dest.exists():
        print(f"    {name}: already downloaded")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial")

    try:
        urllib.request.urlretrieve(url, partial, reporthook=_progress(name))
    except Exception as e:
        if partial.exists():
            partial.unlink()
        sys.exit(f"\nERROR: download failed: {e}\nURL: {url}")

    partial.rename(dest)
    print(f"\n    {name}: saved to {dest}")


def download_embedding_model() -> None:
    """Download the embedding model if not already present."""
    _download("embedding", EMBEDDING_MODEL_URL, EMBEDDING_MODEL_FILE)


def download_summary_model() -> None:
    """Download the summary model if not already present."""
    _download("summary", SUMMARY_MODEL_URL, SUMMARY_MODEL_FILE)


def download_chat_model() -> None:
    """Download the chat model if not already present."""
    _download("chat", CHAT_MODEL_URL, CHAT_MODEL_FILE)
