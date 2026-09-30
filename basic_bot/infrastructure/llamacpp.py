"""Build llama.cpp from source.

Clones a pinned release of llama.cpp into ~/.bountiful/llama.cpp and
compiles llama-server with cmake. Checks for required build tools
before starting.

The release is pinned by LLAMA_VERSION. After a successful build, the
version is recorded beside the source, and a later run rebuilds only
if the recorded version differs from the pin. Changing the pin is how
llama.cpp is upgraded on purpose.

Build flags (e.g. -DGGML_CUDA=ON) are passed by __main__.py based
on hardware detection.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_VERSION = "v0.5.0"

BOUNTIFUL_HOME = Path.home() / ".bountiful"
LLAMA_DIR = BOUNTIFUL_HOME / "llama.cpp"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"
VERSION_FILE = LLAMA_DIR / ".bountiful-version"


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def _built_version() -> str | None:
    """The llama.cpp version recorded by the last successful build."""
    if not VERSION_FILE.exists():
        return None
    return VERSION_FILE.read_text().strip() or None


def check_prerequisites() -> None:
    """Verify cmake and a C++ compiler are available."""
    missing = []
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(c) for c in ("c++", "g++", "clang++", "cc")):
        missing.append("a C++ compiler (g++ or clang++)")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n"
            "Install them with your system package manager and re-run.\n"
            "  Arch/CachyOs:   sudo pacman -S cmake gcc\n"
            "  Debian/Ubuntu:  sudo apt install cmake build-essential\n"
            "  Fedora:         sudo dnf install cmake gcc-c++\n"
            "  macOS:          xcode-select --install && brew install cmake"
        )
    print("    cmake and a C++ compiler found")


def build(flags: list[str] | None = None) -> None:
    """Clone and compile the pinned llama-server, if not already built.

    Args:
        flags: Additional cmake flags, e.g. ["-DGGML_CUDA=ON"].
               Determined by hardware detection in __main__.py.
    """
    built = _built_version()
    if SERVER_BIN.exists() and built == LLAMA_VERSION:
        print(f"    already done — llama.cpp {LLAMA_VERSION}")
        return

    if LLAMA_DIR.exists():
        print(
            f"    llama.cpp {built or 'unpinned build'} → {LLAMA_VERSION}, "
            "rebuilding"
        )
        shutil.rmtree(LLAMA_DIR)

    BOUNTIFUL_HOME.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--branch", LLAMA_VERSION,
            LLAMA_REPO, str(LLAMA_DIR),
        ],
    )
    if result.returncode != 0:
        _fail(f"git clone of llama.cpp {LLAMA_VERSION} failed — see output above")

    cmake_flags = flags or []
    flag_str = " ".join(cmake_flags) if cmake_flags else "CPU-only"
    print(
        f"    compiling llama-server {LLAMA_VERSION} ({flag_str}) "
        "— this takes a few minutes..."
    )

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

    VERSION_FILE.write_text(LLAMA_VERSION + "\n")
    print(f"    built {SERVER_BIN} ({LLAMA_VERSION})")
