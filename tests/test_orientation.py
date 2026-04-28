"""OSD service tests. Requires Tesseract on PATH and (optional) sample images."""

import os
from pathlib import Path
from typing import Optional

import pytest

from app.services.orientation import ORIENTATION_MAP, detect_orientation

_EXPECTED = {
    "SCR-20260427-rspu.png": "upside_down",
    "SCR-20260427-rvlq TLC.png": "rotated_left",
    "SCR-20260427-rvlq.png": "rotated_left",
    "SCR-20260427-rvst.png": "upright",
    "SCR-20260427-rwad.png": "rotated_right",
}


def _samples_dir() -> Optional[Path]:
    env = os.getenv("SAMPLES_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    default = Path("/Users/conradosk/Desktop/Tesseract OCR 2/samples")
    return default if default.is_dir() else None


def test_mapping_covers_tesseract_rotations() -> None:
    for deg in (0, 90, 180, 270):
        assert ORIENTATION_MAP[deg] in (
            "upright",
            "rotated_left",
            "upside_down",
            "rotated_right",
        )


@pytest.mark.skipif(
    _samples_dir() is None,
    reason="SAMPLES_DIR not set or default samples path missing; skipping integration tests",
)
def test_detect_matches_expected_against_samples() -> None:
    root = _samples_dir()
    assert root is not None
    for path in sorted(root.glob("*.png")):
        name = path.name
        expected = _EXPECTED.get(name)
        if expected is None:
            continue
        data = path.read_bytes()
        result = detect_orientation(data)
        assert result.orientation == expected, (
            f"{name}: got {result.orientation!r} expected {expected!r}"
        )
