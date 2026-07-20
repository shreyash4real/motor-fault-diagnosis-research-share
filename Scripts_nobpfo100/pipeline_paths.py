"""Small path helper shared by the legacy-named pipeline scripts.

Environment variables make the scripts portable without hiding any data path in
source. ``run_all_nobpfo100.py`` sets these values for a full pipeline run.
"""

from __future__ import annotations

import os
from pathlib import Path


def path_value(name: str, default: str | Path) -> str:
    """Return an expanded path from an environment variable or a documented default."""
    return str(Path(os.environ.get(name, str(default))).expanduser())
