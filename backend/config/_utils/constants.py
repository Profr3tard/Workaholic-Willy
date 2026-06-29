"""
Doc string here: Error and Constants
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised on any configuration load / validation failure."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
_PROFILE_ENV_VAR = "WORKAHOLIC-WILLY"
_ADAPTATION_OVERLAY_ENV_VAR = "WORKAHOLIC-WILLY_ADAPTATION_OVERLAY"

# Required camera files. These are not auto-discovered because the
# loader maps each one to a specific section in the AppConfig tree.
_CAMERA_FILES = ("cam.yaml", "stereomatcher.yaml", "eye_to_hand.yaml")
_CAMERA_KEYS = ("cameras", "stereomatcher", "eye_to_hand")

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\:-([^}]*))?\}")