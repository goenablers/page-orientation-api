"""OSD service tests. Requires Tesseract on PATH and sample images in samples/rotation."""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pytest

from app.services import orientation
from app.services.orientation import ORIENTATION_MAP, _OcrResult, detect_orientation

_ROTATION_SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "rotation"

# Filename prefixes (UR, RL, RR, UD) encode the expected orientation.
_PREFIX_EXPECTED = {
    "UR": "upright",
    "RL": "rotated_left",
    "RR": "rotated_right",
    "UD": "upside_down",
}

# Pages that do not follow the prefix convention.
_SPECIAL_EXPECTED = {
    "printsrc.png": "upright",
    "wrong.png": "upright",
    "wrong_exif8.png": "rotated_right",
}

_VALID_METHODS = frozenset(("osd", "osd_preprocessed", "ocr_2way", "ocr_4way"))


def _expected_orientation(filename: str) -> str | None:
    if filename in _SPECIAL_EXPECTED:
        return _SPECIAL_EXPECTED[filename]
    return _PREFIX_EXPECTED.get(filename[:2].upper())


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

    def _fake_ocr(*candidates: int, image: np.ndarray) -> _OcrResult:
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
    not _ROTATION_SAMPLES_DIR.is_dir(),
    reason="samples/rotation directory not present",
)
def test_detect_rotation_samples() -> None:
    """Integration test against samples bundled in the repo."""
    tested = 0
    for path in sorted(_ROTATION_SAMPLES_DIR.glob("*.png")):
        expected = _expected_orientation(path.name)
        if expected is None:
            continue
        result = detect_orientation(path.read_bytes())
        assert result.orientation == expected, (
            f"{path.name}: got {result.orientation!r} expected {expected!r}"
        )
        _assert_contract(result)
        tested += 1
    assert tested > 0, "no labeled rotation samples found in samples/rotation"
