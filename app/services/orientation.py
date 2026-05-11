"""Tesseract OSD-based page rotation detection (service layer)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import cv2  # type: ignore
import numpy as np
import pytesseract  # type: ignore

# Maps Tesseract "Rotate" (clockwise deg to make image upright) to API labels.
ORIENTATION_MAP: dict[int, str] = {
    0: "upright",
    90: "rotated_left",
    180: "upside_down",
    270: "rotated_right",
}

_RE_KV = re.compile(r"^([A-Za-z ()]+):\s*(.+?)\s*$")


def _parse_osd(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _RE_KV.match(line.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        try:
            out[key] = int(val)
        except ValueError:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


@dataclass(frozen=True)
class OrientationResult:
    orientation: str
    confidence: Optional[float]
    rotate_degrees: int
    raw_osd: Optional[str] = None


def detect_orientation(image_bytes: bytes, *, include_debug: bool = False) -> OrientationResult:
    """
    Detect how the page is rotated using Tesseract OSD (--psm 0).

    Uses OpenCV imdecode (grayscale) for consistent results with the original CLI.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode image")

    try:
        raw = pytesseract.image_to_osd(image, config="--psm 0")
    except pytesseract.pytesseract.TesseractError as exc:
        if "Too few characters" in str(exc):
            return OrientationResult(
                orientation="upright",
                confidence=None,
                rotate_degrees=0,
                raw_osd=str(exc) if include_debug else None,
            )
        raise

    parsed = _parse_osd(raw)

    rotate_raw = parsed.get("Rotate", 0)
    rotate = int(rotate_raw) if rotate_raw is not None else 0
    rotate = rotate % 360
    confidence = parsed.get("Orientation confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
    else:
        confidence = float(confidence) if confidence is not None else None

    return OrientationResult(
        orientation=ORIENTATION_MAP.get(rotate, "upright"),
        confidence=confidence,
        rotate_degrees=rotate,
        raw_osd=raw if include_debug else None,
    )
