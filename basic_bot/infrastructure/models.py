"""Download and manage models.

Models are stored at ~/.bountiful/models/, shared by all agents
on the machine. Two categories: one embedding model (small, CPU)
and inference models (large, GPU).
"""

import sys
import urllib.request
from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"

# --- Embedding model (built automatically by build.py) ---

EMBEDDING_MODEL_NAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBEDDING_MODEL_FILE = MODELS_DIR / EMBEDDING_MODEL_NAME
EMBEDDING_MODEL_URL = (
    "https://huggingface.co/CablepunkPress/Qwen3-Embedding-0.6B-GGUF"
    f"/resolve/main/{EMBEDDING_MODEL_NAME}"
)

# --- Inference model catalog (downloaded on demand) ---

INFERENCE_MODELS: dict[str, dict] = {
    "qwen3-8b-q4_k_m": {
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        "launch_args": [
            "--n-gpu-layers", "-1",
            "--flash-attn", "on",
            "-np", "1",
        ],
    },
    "qwen3-8b-q8_0": {
        "filename": "Qwen3-8B-Q8_0.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf",
        "launch_args": [
            "--n-gpu-layers", "-1",
            "--flash-attn", "on",
            "-np", "1",
        ],
    },
        "qwen3.5-9b-q5_k_m": {
        "filename": "Qwen3.5-9B-Q5_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q5_K_M.gguf",
        "launch_args": [
            "--n-gpu-layers", "-1",
            "--flash-attn", "on",
            "-np", "1",
        ],
    },
}


def get_inference_model_path(model_id: str) -> Path:
    """Return the expected file path for an inference model."""
    if model_id not in INFERENCE_MODELS:
        raise ValueError(
            f"Unknown inference model: {model_id!r}\n"
            f"Known models: {sorted(INFERENCE_MODELS)}"
        )
    return MODELS_DIR / INFERENCE_MODELS[model_id]["filename"]


def get_inference_launch_args(model_id: str) -> list[str]:
    """Return the llama-server launch flags for a model."""
    if model_id not in INFERENCE_MODELS:
        raise ValueError(f"Unknown inference model: {model_id!r}")
    return list(INFERENCE_MODELS[model_id]["launch_args"])


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def download_embedding_model() -> None:
    """Download the embedding model if not already present."""
    if EMBEDDING_MODEL_FILE.exists():
        print(f"    already done — {EMBEDDING_MODEL_FILE} exists")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partial = EMBEDDING_MODEL_FILE.with_suffix(".partial")

    def report(count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(count * block_size, total)
            percent = done * 100 // total
            mb = done // (1024 * 1024)
            print(f"\r    {percent}% ({mb} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(EMBEDDING_MODEL_URL, partial, reporthook=report)
    except Exception as e:
        if partial.exists():
            partial.unlink()
        _fail(f"model download failed: {e}\nURL: {EMBEDDING_MODEL_URL}")

    partial.rename(EMBEDDING_MODEL_FILE)
    print(f"\n    saved to {EMBEDDING_MODEL_FILE}")
