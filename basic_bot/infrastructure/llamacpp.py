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

Build flags and the backend name (Metal or Vulkan) are passed by
__main__.py, from the operating system.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from basic_bot.paths import BOUNTIFUL_HOME, LLAMA_DIR, SERVER_BIN

LLAMA_VERSION = "v0.7.0"
# The commit the tag points to. A tag can be moved; a commit can't.
# Check with:
#   git ls-remote https://github.com/ggml-org/llama.cpp.git 'refs/tags/v0.7.0*'
# For an annotated tag, use the line ending in ^{}.
LLAMA_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
VERSION_FILE = LLAMA_DIR / ".bountiful-version"

RERUN = "Then run 'python3 build.py' again."


def _fail(message: str) -> NoReturn:
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


# Headers the Vulkan build needs, where distributions install them.
# SPIR-V's are a separate package that the Vulkan one doesn't always
# pull in, per llama.cpp's build documentation (docs/build.md).
VULKAN_HEADER = Path("/usr/include/vulkan/vulkan.h")
SPIRV_HEADER = Path("/usr/include/spirv/unified1/spirv.hpp")


def _check_vulkan_linux() -> None:
    """Vulkan on Linux: cmake, a C++ compiler, glslc, and the headers.

    The Vulkan loader and the graphics driver are needed too, but
    desktop distributions install those already. If one is missing,
    detection finds no graphics card and says so.
    """
    missing = []
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(c) for c in ("c++", "g++", "clang++")):
        missing.append("a C++ compiler")
    if not VULKAN_HEADER.exists():
        missing.append("the Vulkan headers")
    if shutil.which("glslc") is None:
        missing.append("glslc, the shader compiler")
    if not SPIRV_HEADER.exists():
        missing.append("the SPIR-V headers")

    if missing:
        _fail(
            "Missing required tools: " + ", ".join(missing) + "\n\n"
            "Install them with your system's package manager:\n"
            "  Arch/CachyOS:   sudo pacman -S cmake gcc vulkan-headers "
            "shaderc spirv-headers\n"
            "  Debian/Ubuntu:  sudo apt install cmake g++ libvulkan-dev "
            "glslc spirv-headers\n\n"
            f"{RERUN}"
        )
    print("    cmake, a C++ compiler, glslc, and the Vulkan headers found")


def check_prerequisites(backend: str) -> None:
    """Verify the build tools for this machine's backend are installed."""
    if backend == "metal":
        _check_macos()
    else:
        _check_vulkan_linux()


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
        flags: Additional cmake flags, e.g. ["-DGGML_VULKAN=ON"].
        backend: The GPU backend being built, for messages ("Metal", "Vulkan").
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

    VERSION_FILE.write_text(record + "\n")
    print(f"    built {SERVER_BIN} ({LLAMA_VERSION}, {backend})")
