"""Build llama.cpp from source.

Clones the repo into ~/.bountiful/llama.cpp and compiles llama-server
with cmake. Checks for required build tools before starting.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
LLAMA_DIR = BOUNTIFUL_HOME / "llama.cpp"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def check_prerequisites() -> None:
    """Verify git, cmake, and a C++ compiler are available."""
    missing = []
    if shutil.which("git") is None:
        missing.append("git")
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(c) for c in ("c++", "g++", "clang++", "cc")):
        missing.append("a C++ compiler (g++ or clang++)")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n"
            "Install them with your system package manager and re-run.\n"
            "  Arch:           sudo pacman -S git cmake gcc\n"
            "  Debian/Ubuntu:  sudo apt install git cmake build-essential\n"
            "  Fedora:         sudo dnf install git cmake gcc-c++\n"
            "  macOS:          xcode-select --install && brew install cmake"
        )
    print("    git, cmake, and a C++ compiler found")


def build() -> None:
    """Clone and compile llama-server if not already built."""
    if SERVER_BIN.exists():
        print(f"    already done — {SERVER_BIN} exists")
        return

    BOUNTIFUL_HOME.mkdir(parents=True, exist_ok=True)

    if not (LLAMA_DIR / "CMakeLists.txt").exists():
        result = subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_REPO, str(LLAMA_DIR)],
        )
        if result.returncode != 0:
            _fail("git clone of llama.cpp failed — see output above")

    print("    compiling llama-server — this takes a few minutes...")
    configure = subprocess.run(["cmake", "-B", "build"], cwd=LLAMA_DIR)
    if configure.returncode != 0:
        _fail("cmake configure failed — see output above")

    result = subprocess.run(
        [
            "cmake", "--build", "build",
            "--config", "Release",
            "--target", "llama-server",
            "-j", str(os.cpu_count() or 4),
        ],
        cwd=LLAMA_DIR,
    )
    if result.returncode != 0 or not SERVER_BIN.exists():
        _fail("llama.cpp build failed — see output above")
    print(f"    built {SERVER_BIN}")
