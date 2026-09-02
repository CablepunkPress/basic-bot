"""Build llama.cpp from source.

Clones the repo into ~/.bountiful/llama.cpp and compiles llama-server
with cmake. Checks for required build tools before starting.

Build flags (e.g. -DGGML_CUDA=ON) are passed by __main__.py based
on hardware detection.
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


def build(flags: list[str] | None = None) -> None:
    """Clone and compile llama-server if not already built.

    Args:
        flags: Additional cmake flags, e.g. ["-DGGML_CUDA=ON"].
               Determined by hardware detection in __main__.py.
    """
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

    cmake_flags = flags or []
    flag_str = " ".join(cmake_flags) if cmake_flags else "CPU-only"
    print(f"    compiling llama-server ({flag_str}) — this takes a few minutes...")

    configure = subprocess.run(
        ["cmake", "-B", "build"] + cmake_flags,
        cwd=LLAMA_DIR,
    )
    if configure.returncode != 0:
        _fail("cmake configure failed — see output above")

    # CUDA builds segfault with too many parallel jobs
    jobs = "4" if any("CUDA" in f for f in cmake_flags) else str(os.cpu_count() or 4)

    result = subprocess.run(
        [
            "cmake", "--build", "build",
            "--config", "Release",
            "--target", "llama-server",
            "-j", jobs,
        ],
        cwd=LLAMA_DIR,
    )
    if result.returncode != 0 or not SERVER_BIN.exists():
        _fail("llama.cpp build failed — see output above")
    print(f"    built {SERVER_BIN}")
