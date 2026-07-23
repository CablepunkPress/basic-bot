"""One-time local setup for Basic Bot.

Creates the virtual environment, installs Python dependencies, builds
llama.cpp from source (CPU), downloads the embedding model, and stores
the Anthropic API key in the OS-native encrypted keyring (KWallet,
GNOME Keyring, or macOS Keychain — whichever the system provides).

Idempotent — each step is skipped if already done, so re-running after
a failure picks up where it left off. Run with the system Python:

    python build.py
"""

import getpass
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"

HOME = Path.home() / ".basic-bot"
LLAMA_DIR = HOME / "llama.cpp"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"

MODELS_DIR = HOME / "models"
MODEL_NAME = "embeddinggemma-300M-Q8_0.gguf"
MODEL_FILE = MODELS_DIR / MODEL_NAME
MODEL_URL = (
    "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF"
    f"/resolve/main/{MODEL_NAME}"
)

KEYRING_SERVICE = "basic-bot"
KEYRING_USERNAME = "anthropic_api_key"

TOTAL_STEPS = 5


def step(n: int, message: str) -> None:
    print(f"\n[{n}/{TOTAL_STEPS}] {message}")


def skip(message: str) -> None:
    print(f"    already done — {message}")


def fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    step(1, "Checking prerequisites")

    missing = []
    if shutil.which("git") is None:
        missing.append("git")
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(c) for c in ("c++", "g++", "clang++", "cc")):
        missing.append("a C++ compiler (g++ or clang++)")

    if missing:
        fail(
            "Missing required tools: " + ", ".join(missing) + "\n"
            "Install them with your system package manager and re-run.\n"
            "  Debian/Ubuntu:  sudo apt install git cmake build-essential\n"
            "  Fedora:         sudo dnf install git cmake gcc-c++\n"
            "  Arch:           sudo pacman -S git cmake gcc\n"
            "  macOS:          xcode-select --install && brew install cmake"
        )
    print("    git, cmake, and a C++ compiler found")


def create_venv() -> None:
    step(2, "Creating virtual environment and installing dependencies")

    if not venv_python().exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        print(f"    created {VENV}")
    else:
        skip(f"{VENV} exists")

    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-e", ".[local]"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        fail("pip install failed — see output above")
    print("    dependencies installed")


def build_llama_cpp() -> None:
    step(3, "Building llama.cpp (CPU)")

    if SERVER_BIN.exists():
        skip(f"{SERVER_BIN} exists")
        return

    HOME.mkdir(parents=True, exist_ok=True)

    if not (LLAMA_DIR / "CMakeLists.txt").exists():
        result = subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR)],
        )
        if result.returncode != 0:
            fail("git clone of llama.cpp failed — see output above")

    print("    compiling llama-server — this takes a few minutes...")
    configure = subprocess.run(
        ["cmake", "-B", "build"],
        cwd=LLAMA_DIR,
    )
    if configure.returncode != 0:
        fail("cmake configure failed — see output above")

    build = subprocess.run(
        [
            "cmake", "--build", "build",
            "--config", "Release",
            "--target", "llama-server",
            "-j", str(os.cpu_count() or 4),
        ],
        cwd=LLAMA_DIR,
    )
    if build.returncode != 0 or not SERVER_BIN.exists():
        fail("llama.cpp build failed — see output above")
    print(f"    built {SERVER_BIN}")


def download_model() -> None:
    step(4, f"Downloading embedding model ({MODEL_NAME}, ~334 MB)")

    if MODEL_FILE.exists():
        skip(f"{MODEL_FILE} exists")
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
        fail(f"model download failed: {e}\nURL: {MODEL_URL}")

    partial.rename(MODEL_FILE)
    print(f"\n    saved to {MODEL_FILE}")


def store_api_key() -> None:
    step(5, "Anthropic API key (stored in your system keyring)")

    import keyring
    import keyring.errors

    try:
        existing = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except keyring.errors.KeyringError:
        existing = None

    if existing:
        replace = input(
            "    A key is already stored. Replace it? [y/N] "
        ).strip().lower()
        if replace != "y":
            skip("keeping existing key")
            return

    print("    Get a key at https://console.anthropic.com/")
    print("    (input is hidden — paste and press Enter)")
    key = getpass.getpass("    Paste your Anthropic API key: ").strip()
    if not key:
        fail("no key entered — re-run build.py to try again")

    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    except keyring.errors.KeyringError as e:
        fail(
            f"Could not store the key in your system keyring: {e}\n"
            "Basic Bot needs a working OS keyring — KWallet or GNOME "
            "Keyring on Linux, Keychain on macOS. Make sure one is "
            "installed and unlocked, then re-run build.py."
        )

    # Verify it round-trips through a real backend rather than trusting
    # the write silently succeeded.
    check = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if check != key:
        fail(
            "Key was written but could not be read back — your "
            "system keyring may not be working correctly."
        )

    backend = type(keyring.get_keyring()).__module__
    print(f"    stored in system keyring ({backend})")


def main() -> None:
    print("Basic Bot — local setup")
    check_prerequisites()
    create_venv()
    build_llama_cpp()
    download_model()
    store_api_key()
    print("\nSetup complete. Start Basic Bot with:\n\n    python run.py\n")


if __name__ == "__main__":
    main()
