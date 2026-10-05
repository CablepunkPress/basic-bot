"""Build llama.cpp from source.

Clones a pinned release of llama.cpp into ~/.bountiful/llama.cpp and
compiles llama-server with cmake. Checks for required build tools
before starting.

The release is pinned twice: by LLAMA_VERSION, the tag that is cloned,
and by LLAMA_COMMIT, the commit that tag must point to. A tag is only
a name and can be moved; a commit identifies the exact source. After
cloning, the commit is checked before anything is compiled.

After a successful build, the pin and the cmake flags are recorded
beside the source, and a later run rebuilds only if either differs.
Changing the pin is how llama.cpp is upgraded on purpose.

Build flags and the backend name (Metal or CUDA) are passed by
__main__.py based on hardware detection.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_VERSION = "v0.5.0"
# The commit the tag points to. A tag can be moved; a commit can't.
# Check with:
#   git ls-remote https://github.com/ggml-org/llama.cpp.git 'refs/tags/v0.5.0*'
# For an annotated tag, use the line ending in ^{}.
LLAMA_COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"

BOUNTIFUL_HOME = Path.home() / ".bountiful"
LLAMA_DIR = BOUNTIFUL_HOME / "llama.cpp"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
SERVER_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"
VERSION_FILE = LLAMA_DIR / ".bountiful-version"

RERUN = "Then run 'python3 build.py' again."


def _fail(message: str) -> None:
    sys.exit(f"\nERROR: {message}")


def _record(flags: list[str]) -> str:
    """What the version file records: tag, commit, and cmake flags."""
    return " ".join([LLAMA_VERSION, LLAMA_COMMIT, *flags])


def _built_version() -> str | None:
    """What the last successful build recorded."""
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
            "Homebrew is not installed.\n\n"
            "Follow 'Prepare your Mac' in the README.\n"
            f"{RERUN}"
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
            "Missing required tools: " + ", ".join(missing) + "\n\n"
            "Install them:\n"
            + "\n".join(f"  {c}" for c in commands) + "\n\n"
            f"{RERUN}"
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
                "The CUDA toolkit is installed but not on your PATH yet.\n\n"
                "Log out and back in.\n"
                f"{RERUN}"
            )
        missing.append("the CUDA toolkit")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n\n"
            "Install them with your system's package manager:\n"
            "  Arch/CachyOS:  sudo pacman -S cmake gcc cuda\n\n"
            f"{RERUN}"
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

def _verify_commit() -> None:
    """Stop if the cloned tag doesn't point to the pinned commit."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=LLAMA_DIR, capture_output=True, text=True,
    )
    actual = head.stdout.strip()
    if head.returncode != 0 or actual != LLAMA_COMMIT:
        shutil.rmtree(LLAMA_DIR)
        _fail(
            f"llama.cpp {LLAMA_VERSION} is not the expected commit.\n"
            f"  expected: {LLAMA_COMMIT}\n"
            f"  received: {actual or 'unknown'}\n\n"
            f"The tag may have been moved. Check the release before "
            f"changing the pin."
        )
    print(f"    source verified: commit {LLAMA_COMMIT[:7]}")


def build(flags: list[str], backend: str) -> None:
    """Clone and compile the pinned llama-server, if not already built.

    Args:
        flags: Additional cmake flags, e.g. ["-DGGML_CUDA=ON"].
        backend: The GPU backend being built, for messages ("Metal", "CUDA").
    """
    record = _record(flags)
    built = _built_version()
    if SERVER_BIN.exists() and built == record:
        print(f"    already built — llama.cpp {LLAMA_VERSION}")
        return

    if LLAMA_DIR.exists():
        reason = "first pinned build" if built is None else "pin or build flags changed"
        print(f"    rebuilding llama.cpp {LLAMA_VERSION}: {reason}")
        shutil.rmtree(LLAMA_DIR)

    BOUNTIFUL_HOME.mkdir(parents=True, exist_ok=True)

    print(f"    downloading llama.cpp {LLAMA_VERSION} source from GitHub")
    result = subprocess.run(
        [
            "git", "-c", "advice.detachedHead=false",
            "clone", "--depth", "1",
            "--branch", LLAMA_VERSION,
            LLAMA_REPO, str(LLAMA_DIR),
        ],
    )
    if result.returncode != 0:
        _fail(
            f"downloading llama.cpp {LLAMA_VERSION} failed — "
            f"see output above.\n\n"
            f"Check your connection. {RERUN}"
        )

    _verify_commit()

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

    VERSION_FILE.write_text(record + "\n")
    print(f"    built {SERVER_BIN} ({LLAMA_VERSION}, {backend})")
