"""Optional chat models: what 'bountiful models' runs.

The build installs only what every agent needs, the embedding model,
the summary model, and the default chat model, so a first install is
as short and simple as it can be. Every other chat model in the
profile is optional, and is offered here and nowhere else.

Answers are recorded in .choices.json beside the models, since every
agent on the machine shares them. The build keeps accepted models
current when an update changes their pins, and never offers new ones.
An optional model already on disk counts as accepted, as when an old
default chat model becomes optional.

Model files the profile doesn't use, or that belong to a declined
model, are listed here and offered for deletion in one question,
defaulting to no. The models folder belongs to Bountiful; models to
experiment with belong somewhere else.
"""

import json
import sys
from pathlib import Path

from basic_bot.paths import MODELS_DIR
from basic_bot.profile import ProfileError, profile_name
from basic_bot.setup.downloads import (
    Model,
    ask,
    fail,
    gb,
    install,
    load_verified,
    profile_models,
    save_verified,
)

# The user's answers about optional models: model ids accepted and
# declined.
CHOICES_FILE = MODELS_DIR / ".choices.json"

RERUN = "Run 'bountiful models' again"


def _short(path: Path) -> str:
    """A path with the home folder shown as ~."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


# --- Choices ---

def load_choices() -> dict[str, list[str]]:
    try:
        data = json.loads(CHOICES_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    return {key: list(data.get(key, [])) for key in ("accepted", "declined")}


def _record_choice(choices: dict, model_id: str, accepted: bool) -> None:
    for key in ("accepted", "declined"):
        if model_id in choices[key]:
            choices[key].remove(model_id)
    choices["accepted" if accepted else "declined"].append(model_id)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHOICES_FILE.write_text(json.dumps(choices, indent=2) + "\n")


def kept_models(models: list[Model], choices: dict) -> list[Model]:
    """The models this machine keeps: every mandatory one, and each
    optional one that was accepted or is already on disk.

    An optional model on disk with no recorded answer is recorded as
    accepted, so the build keeps it current from then on.
    """
    kept = []
    for model in models:
        if model.mandatory:
            kept.append(model)
        elif not model.pinned or model.model_id in choices["declined"]:
            continue
        elif model.model_id in choices["accepted"]:
            kept.append(model)
        elif model.on_disk:
            _record_choice(choices, model.model_id, True)
            kept.append(model)
    return kept


# --- Leftover files ---

def _unused_files(models: list[Model], choices: dict) -> list[Path]:
    """Model files nothing uses: not in the profile, or a declined model's.

    Unfinished downloads count, since a partial file can be large.
    """
    named = {f["file"] for m in models for f in m.files}
    keep = {f["file"] for m in kept_models(models, choices) for f in m.files}
    declined = {
        f["file"] for m in models
        if m.model_id in choices["declined"] for f in m.files
    }

    unused = []
    for path in sorted(MODELS_DIR.glob("*.gguf")) + sorted(MODELS_DIR.glob("*.gguf.partial")):
        name = path.name.removesuffix(".partial")
        if name in keep:
            continue
        if name not in named or name in declined:
            unused.append(path)
    return unused


def _offer_cleanup(models: list[Model], choices: dict) -> None:
    """List model files nothing uses, and offer to delete them."""
    unused = _unused_files(models, choices)
    if not unused:
        return

    total = sum(p.stat().st_size for p in unused)
    print("\nModel files no longer used\n")
    width = max(len(p.name) for p in unused)
    for path in unused:
        print(f"    {path.name:<{width}}  {gb(path.stat().st_size):>8}")
    print()
    if not ask(f"Delete them, freeing {gb(total)}?", False, indent="  "):
        print(f"  kept. They're in {_short(MODELS_DIR)}.")
        return

    verified = load_verified()
    for path in unused:
        path.unlink()
        verified.pop(path.name, None)
    save_verified(verified)
    print("  deleted")


# --- bountiful models ---

def run() -> None:
    """Offer each optional chat model, download the accepted ones, and
    offer to delete files nothing uses."""
    if not sys.stdin.isatty():
        fail("'bountiful models' asks questions, so run it in a terminal.")

    try:
        name = profile_name()
        models = profile_models()
    except ProfileError as e:
        fail(str(e))

    choices = load_choices()
    kept_models(models, choices)   # records optional models already here

    default = next(m for m in models if m.role == "chat" and m.mandatory)
    optional = [m for m in models if m.role == "chat" and not m.mandatory]

    print(f"Optional chat models, for this machine's profile ({name})\n")
    print(f"  {default.name}, the default chat model, is always installed.")

    if not optional:
        print("  This profile has no optional chat models.")
    else:
        print("  These can be added, and any agent can switch to them:\n")
        for model in optional:
            if not model.pinned:
                print(f"  {model.name} isn't available yet.")
                continue
            wanted_before = model.on_disk or model.model_id in choices["accepted"]
            if model.on_disk:
                question = f"Keep {model.name} ({gb(model.size)}, installed)?"
            else:
                question = f"Download {model.name} ({gb(model.size)})?"
            _record_choice(choices, model.model_id, ask(question, wanted_before, indent="  "))

    wanted = [m for m in optional if m.model_id in choices["accepted"]]
    if wanted:
        print("\nAI models")
        install(wanted, RERUN)

    _offer_cleanup(models, choices)
    print()


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Partial downloads are kept. {RERUN} to resume.")
        sys.exit(130)
