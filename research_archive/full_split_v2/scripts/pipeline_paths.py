"""Path compatibility layer for the archived full-split v2 scripts.

The original research was run from ``C:\\Project Work``.  This module maps
those historical locations to explicit, portable environment variables without
changing feature extraction or model logic.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


ARCHIVE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ARCHIVE_ROOT / "reproduced_outputs"


def _path_from_env(name: str, default: Path) -> Path:
    """Return an environment override or a documented repository default."""
    return Path(os.environ.get(name, str(default))).expanduser()


def legacy_path(historical_path: str) -> str:
    """Map an original ``C:\\Project Work`` path to a portable location.

    Set the input roots before running preprocessing or training, for example:

    ``MFDS_ARCHIVE_RAW_ROOT=/data/Motor-2``
    ``MFDS_ARCHIVE_DENOISED_ROOT=/data/Denoised/Motor-2``
    ``MFDS_ARCHIVE_OUTPUT_ROOT=/work/full-split-v2``
    """
    windows = PureWindowsPath(historical_path)
    parts = [part.lower() for part in windows.parts]
    project_index = parts.index("project work") if "project work" in parts else 0
    suffix = list(windows.parts[project_index + 1 :])
    lower = [part.lower() for part in suffix]

    if lower[:3] == ["dataset", "electric", "motor-2"]:
        root = _path_from_env("MFDS_ARCHIVE_RAW_ROOT", ARCHIVE_ROOT / "data" / "Motor-2")
        return str(root.joinpath(*suffix[3:]))
    if lower[:3] == ["denoised", "electric", "motor-2"]:
        root = _path_from_env("MFDS_ARCHIVE_DENOISED_ROOT", ARCHIVE_ROOT / "data" / "Denoised" / "Motor-2")
        return str(root.joinpath(*suffix[3:]))
    if lower[:3] == ["dataset", "vibration", "motor-2"]:
        root = _path_from_env("MFDS_ARCHIVE_VIBRATION_RAW_ROOT", ARCHIVE_ROOT / "data" / "vibration" / "Motor-2")
        return str(root.joinpath(*suffix[3:]))
    if lower[:3] == ["denoised", "vibration", "motor-2"]:
        root = _path_from_env("MFDS_ARCHIVE_VIBRATION_DENOISED_ROOT", ARCHIVE_ROOT / "data" / "Denoised" / "vibration" / "Motor-2")
        return str(root.joinpath(*suffix[3:]))
    if lower[:2] == ["outputs", "4class"]:
        root = _path_from_env("MFDS_ARCHIVE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
        return str(root.joinpath(*suffix[2:]))
    if lower[:2] == ["outputs", "vibration"]:
        root = _path_from_env("MFDS_ARCHIVE_VIBRATION_OUTPUT_ROOT", ARCHIVE_ROOT / "reproduced_vibration_outputs")
        return str(root.joinpath(*suffix[2:]))
    if lower[:2] == ["outputs", "main"]:
        default = ARCHIVE_ROOT / "results" / "validated_manifest.csv"
        root = _path_from_env("MFDS_ARCHIVE_VALIDATED_MANIFEST", default)
        return str(root if len(suffix) == 3 else root.parent.joinpath(*suffix[2:]))
    return str(ARCHIVE_ROOT.joinpath(*suffix))
