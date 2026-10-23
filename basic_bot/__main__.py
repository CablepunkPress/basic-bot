"""The engine's build: llama.cpp, the hardware profile, and the models.

Called by build.py in the bountiful repo, as `python -m basic_bot`,
once the user has agreed to install. It asks nothing more, except
before a large new download on a machine that's already set up.

Takes the GPU backend from the operating system, checks the build
tools for it, and compiles the pinned llama.cpp with that backend.
Then detects the machine's memory, which on Linux needs the
llama-server just built, chooses and records the hardware profile,
and downloads the models it requires.

The build keeps the models this machine uses: the embedding model,
the summary model, and the default chat model with its drafter, plus
any optional chat model already accepted with 'bountiful models' or
already on disk. It never offers or lists the others; that's
'bountiful models'. An interrupted download keeps its partial file and
resumes on the next run.
"""

import json
import subprocess
import sys
from typing import NoReturn

from basic_bot import registry
from basic_bot.infrastructure.llamacpp import build, check_prerequisites
from basic_bot.profile import ProfileError, choose_profile, platform_backend
from basic_bot.setup.downloads import install, load_verified, profile_models
from basic_bot.setup.models import kept_models, load_choices

RERUN = "Run 'python3 build.py' again"


def _fail(message: str) -> NoReturn:
    sys.exit(f"\nERROR: {message}")


# --- Agents' tool dependencies ---

def _install_tool_dependencies() -> None:
    """Install what every recorded agent's tool groups declare.

    Tool groups live in each agent's tools/ folder, but their
    dependencies live in this venv, which a rebuild starts empty.
    pip skips anything already installed. Nothing is printed when no
    agent has any, as on a first install.
    """
    try:
        found = registry.entries()
    except registry.RegistryError as e:
        # Only tools are affected, so the rest of the build goes on
        print(f"\nAgents' tools\n    skipping them: {e}")
        return

    deps: set[str] = set()
    problems: list[str] = []
    for entry in found:
        if not entry.present:
            continue
        for manifest_path in sorted((entry.folder / "tools").glob("*/tool.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError) as e:
                problems.append(f"skipping {manifest_path}: {e}")
                continue
            deps.update(manifest.get("dependencies", []))

    if not deps and not problems:
        return

    print("\nAgents' tools")
    for problem in problems:
        print(f"    {problem}")
    if not deps:
        return

    print(f"    installing their dependencies: {', '.join(sorted(deps))}")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", *sorted(deps),
        ],
    )
    if result.returncode != 0:
        _fail(
            "installing tool dependencies failed — see output above.\n\n"
            f"Check your connection. {RERUN}."
        )


# --- Build ---

# Flags for every backend. By default, llama.cpp's build downloads
# llama-server's built-in web interface from Hugging Face, falling back
# to a moving "latest". Bountiful talks to llama-server's API and has
# its own interface, so the download is turned off. The build then
# warns "no assets available - building without an embedded UI",
# which is expected.
COMMON_FLAGS = ["-DLLAMA_USE_PREBUILT_UI=OFF"]


def _build_config(backend: str) -> tuple[list[str], str]:
    """cmake flags and backend name for this machine's backend.

    Metal is llama.cpp's default on macOS, so it needs no backend flag.
    """
    if backend == "metal":
        return COMMON_FLAGS, "Metal"
    return ["-DGGML_VULKAN=ON", *COMMON_FLAGS], "Vulkan"


def main() -> None:
    _install_tool_dependencies()

    try:
        backend = platform_backend()
    except ProfileError as e:
        _fail(str(e))
    flags, backend_name = _build_config(backend)

    print("\nllama.cpp, the inference server")
    check_prerequisites(backend)
    build(flags, backend_name)

    print("\nHardware profile")
    try:
        machine, name = choose_profile()
        models = profile_models()
    except ProfileError as e:
        _fail(str(e))
    print(f"    {backend_name}, {machine.memory_gb} GB for models: {name}")

    # A machine with verified models is already set up, so a large new
    # download there, such as a new default after an update, asks first.
    already_set_up = bool(load_verified())

    print("\nAI models")
    install(kept_models(models, load_choices()), RERUN, confirm=already_set_up)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Partial downloads are kept. {RERUN} to resume.")
        sys.exit(130)
