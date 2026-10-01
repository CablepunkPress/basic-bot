"""Build llama.cpp from source.

Clones a pinned release of llama.cpp into ~/.bountiful/llama.cpp and
compiles llama-server with cmake. Checks for required build tools
before starting.

The release is pinned by LLAMA_VERSION. After a successful build, the
version is recorded beside the source, and a later run rebuilds only
if the recorded version differs from the pin. Changing the pin is how
llama.cpp is upgraded on purpose.

Build flags and the backend name (Metal or CUDA) are passed by
__main__.py based on hardware detection.
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


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def _has_xcode_tools() -> bool:
    """True if Apple's Command Line Tools are installed.

    macOS ships placeholder compiler commands that exist even without
    the tools, so checking for the commands themselves isn't enough.
    """
    result = subprocess.run(["xcode-select", "-p"], capture_output=True)
    return result.returncode == 0


def _check_macos() -> None:
    """Apple Silicon: Homebrew, Apple's Command Line Tools, and cmake."""
    if shutil.which("brew") is None:
        _fail(
            "Homebrew is not installed.\n"
            "Follow 'Prepare your Mac' in the README, then re-run."
        )

    missing, commands = [], []
    if not _has_xcode_tools():
        missing.append("Apple's Command Line Tools")
        commands.append("xcode-select --install")
    if shutil.which("cmake") is None:
        missing.append("cmake")
        commands.append("brew install cmake")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n"
            "Install them and re-run:\n"
            + "\n".join(f"  {c}" for c in commands)
        )
    print("    Homebrew, Command Line Tools, and cmake found")


def _check_nvidia_linux() -> None:
    """NVIDIA on Linux: cmake, a C++ compiler, and the CUDA toolkit."""
    missing = []
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(c) for c in ("c++", "g++", "clang++")):
        missing.append("a C++ compiler")

    if shutil.which("nvcc") is None:
        # Arch's cuda package installs here and joins the PATH at next login
        if Path("/opt/cuda/bin/nvcc").exists():
            _fail(
                "The CUDA toolkit is installed but not on your PATH yet.\n"
                "Log out and back in, then re-run."
            )
        missing.append("the CUDA toolkit")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n"
            "Install them with your system's package manager and re-run.\n"
            "  Arch/CachyOS:   sudo pacman -S cmake gcc cuda"
        )
    print("    cmake, a C++ compiler, and the CUDA toolkit found")


def check_prerequisites(hardware: str) -> None:
    """Verify the build tools for the detected hardware are installed."""
    if hardware.startswith("apple"):
        _check_macos()
    else:
        _check_nvidia_linux()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(flags: list[str], backend: str) -> None:
    """Clone and compile the pinned llama-server, if not already built.

    Args:
        flags: Additional cmake flags, e.g. ["-DGGML_CUDA=ON"].
        backend: The GPU backend being built, for messages ("Metal", "CUDA").
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

    print(
        f"    compiling llama-server {LLAMA_VERSION} with {backend} "
        "— this takes a few minutes..."
    )

    configure = subprocess.run(
        ["cmake", "-B", "build", *flags],
        cwd=LLAMA_DIR,
    )
    if configure.returncode != 0:
        _fail("cmake configure failed — see output above")

    # CUDA builds segfault with too many parallel jobs
    jobs = "4" if backend == "CUDA" else str(os.cpu_count() or 4)

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
    print(f"    built {SERVER_BIN} ({LLAMA_VERSION}, {backend})")
