"""Model file paths.

This module is transitional. Model definitions are moving to
hardware profiles (basic_bot/profiles/*.toml). Once all consumers
read from the profile, this module will be removed.
"""

from pathlib import Path

BOUNTIFUL_HOME = Path.home() / ".bountiful"
MODELS_DIR = BOUNTIFUL_HOME / "models"
