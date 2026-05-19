"""OSD service tests. Requires Tesseract on PATH and (optional) sample images."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np
import pytest

from app.services import orientation
from app.services.orientation import ORIENTATION_MAP, _OcrResult, detect_orientation

_EXPECTED = {
    "SCR-20260427-rspu.png": "upside_down",
    "SCR-20260427-rvlq TLC.png": "rotated_left",
    "SCR-20260427-rvlq.png": "rotated_left",
    "SCR-20260427-rvst.png": "upright",
    "SCR-20260427-rwad.png": "rotated_right",
}

# Samples bundled with the repo, relative to the project root.
_REPO_SAMPLES: dict[str, str] = {
    "559637c1-cf2a-4bc3-bdb9-fdd06e56a1a4.png": "rotated_left",
    # wrong.png stores pixels upside-down with EXIF Orientation=3; printsrc is a
    # screenshot of the same page (upright pixels, no EXIF). Both should read upright.
    "printsrc.png": "upright",
    "wrong.png": "upright",
    # EXIF Orientation=8; OSD hints 90° but OCR confirms 270° (rotated_right).
    "wrong_exif8.png": "rotated_right",
}

_REPO_SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "rotation"

_VALID_METHODS = frozenset(("osd", "osd_preprocessed", "ocr_2way", "ocr_4way"))


def _samples_dir() -> Optional[Path]:
    env = os.getenv("SAMPLES_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    default = Path("/Users/conradosk/Desktop/Tesseract OCR 2/samples")
    return default if default.is_dir() else None


def _assert_contract(result) -> None:
    assert isinstance(result.confidence, float)
    assert result.method in _VALID_METHODS


def _png_bytes() -> bytes:
    image = np.full((16, 16), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_mapping_covers_tesseract_rotations() -> None:
    for deg in (0, 90, 180, 270):
        assert ORIENTATION_MAP[deg] in (
            "upright",
            "rotated_left",
            "upside_down",
            "rotated_right",
        )


def test_preprocessed_path_keeps_raw_osd_confidence(monkeypatch) -> None:
    calls = [
        (270, 1.25, "raw-osd"),
        (180, 4.75, "preprocessed-osd"),
    ]
    monkeypatch.setattr(orientation, "_run_osd", lambda _image: calls.pop(0))

    result = detect_orientation(_png_bytes())

    assert result.method == "osd_preprocessed"
    assert result.orientation == "upside_down"
    assert result.confidence == 1.25


def test_lateral_osd_hint_verified_by_ocr_2way(monkeypatch) -> None:
    # Confident 90° OSD must not short-circuit; 2-way OCR picks 270°.
    calls = [
        (90, 7.0, "raw-osd"),
        (90, 5.0, "preprocessed-osd"),
    ]
    monkeypatch.setattr(orientation, "_run_osd", lambda _image: calls.pop(0))
    monkeypatch.setattr(
        orientation,
        "_ocr_run",
        lambda *candidates, image: _OcrResult(rotate_degrees=270, margin=0.6),
    )

    result = detect_orientation(_png_bytes())

    assert result.method == "ocr_2way"
    assert result.orientation == "rotated_right"
    assert result.rotate_degrees == 270


def test_ocr_2way_stops_when_margin_is_strong(monkeypatch) -> None:
    # Both OSD passes weak -> enter OCR stages.
    # 2-way OCR returns a strong margin -> stop at stage 3.
    calls = [
        (270, 3.5, "raw-osd"),
        (180, 2.0, "preprocessed-osd"),
    ]
    monkeypatch.setattr(orientation, "_run_osd", lambda _image: calls.pop(0))
    monkeypatch.setattr(
        orientation,
        "_ocr_run",
        lambda *candidates, image: _OcrResult(rotate_degrees=90, margin=0.8),
    )

    result = detect_orientation(_png_bytes())

    assert result.method == "ocr_2way"
    assert result.orientation == "rotated_left"
    assert result.confidence == 3.5


def test_ocr_4way_used_when_2way_margin_is_weak(monkeypatch) -> None:
    # Both OSD passes weak, 2-way OCR margin too low -> escalate to 4-way.
    calls = [
        (270, 2.5, "raw-osd"),
        (180, 1.0, "preprocessed-osd"),
    ]
    captured: dict[str, object] = {}
    monkeypatch.setattr(orientation, "_run_osd", lambda _image: calls.pop(0))

    ocr_calls: list[tuple[int, ...]] = []

    def _fake_ocr(*candidates: int, image: np.ndarray) -> _OcrResult:
        ocr_calls.append(candidates)
        # First call (2-way): weak margin; second call (4-way): winner returned.
        if len(candidates) == 2:
            return _OcrResult(rotate_degrees=candidates[0], margin=0.1)
        captured["candidates"] = set(candidates)
        return _OcrResult(rotate_degrees=90, margin=0.7)

    monkeypatch.setattr(orientation, "_ocr_run", _fake_ocr)

    result = detect_orientation(_png_bytes())

    assert result.method == "ocr_4way"
    assert result.orientation == "rotated_left"
    assert result.confidence == 2.5
    assert captured.get("candidates") == {0, 90, 180, 270}


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
        _assert_contract(result)


@pytest.mark.skipif(
    not _REPO_SAMPLES_DIR.is_dir(),
    reason="samples/rotation directory not present",
)
def test_detect_repo_rotation_samples() -> None:
    """Integration test against samples bundled in the repo."""
    for name, expected in _REPO_SAMPLES.items():
        path = _REPO_SAMPLES_DIR / name
        if not path.exists():
            pytest.skip(f"sample file missing: {path}")
        data = path.read_bytes()
        result = detect_orientation(data)
        assert result.orientation == expected, (
            f"{name}: got {result.orientation!r} expected {expected!r}"
        )
        _assert_contract(result)
