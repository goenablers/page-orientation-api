"""Tesseract OSD-based page rotation detection (service layer)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import cv2  # type: ignore
import numpy as np
import pytesseract  # type: ignore
from pytesseract import Output  # type: ignore

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


_LOW_CONFIDENCE_RETRY_THRESHOLD = 5.0


@dataclass(frozen=True)
class _OsdAttempt:
    name: str
    rotate_degrees: int
    confidence: Optional[float]
    raw_osd: str


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _run_osd(image: np.ndarray) -> _OsdAttempt:
    raw = pytesseract.image_to_osd(image, config="--psm 0")
    parsed = _parse_osd(raw)

    rotate_raw = parsed.get("Rotate", 0)
    rotate = int(rotate_raw) if rotate_raw is not None else 0
    rotate = rotate % 360

    confidence = _to_float(parsed.get("Orientation confidence"))
    return _OsdAttempt(
        name="osd",
        rotate_degrees=rotate,
        confidence=confidence,
        raw_osd=raw,
    )


def _binarize(image: np.ndarray) -> np.ndarray:
    # OSD is often more stable on clean binary text than on gray scans.
    # Otsu handles both light and dark backgrounds fairly well.
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th


def _upsample(image: np.ndarray, *, scale: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _rotate_90(image: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate image by 90° increments clockwise.
    k: number of 90° clockwise rotations.
    """
    k = k % 4
    if k == 0:
        return image
    if k == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _run_osd_with_prerotate(image: np.ndarray, *, prerotate_degrees: int, name: str) -> _OsdAttempt:
    """
    Run OSD on a pre-rotated image and translate the returned 'Rotate' degrees
    back to the original image frame.
    """
    k = (prerotate_degrees // 90) % 4
    rotated = _rotate_90(image, k)
    a = _run_osd(rotated)
    implied_rotate = (prerotate_degrees + a.rotate_degrees) % 360
    return _OsdAttempt(
        name=name,
        rotate_degrees=implied_rotate,
        confidence=a.confidence,
        raw_osd=a.raw_osd,
    )


def _split_halves(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    h = image.shape[0]
    mid = h // 2
    top = image[:mid, :]
    bottom = image[mid:, :]
    return [("top_half", top), ("bottom_half", bottom)]


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Estimate skew angle from text-ish pixels and rotate to deskew.
    This is a best-effort fallback used only for low-confidence cases.
    """
    th = _binarize(image)
    # Prefer dark text on white background; invert so text becomes white pixels.
    inv = 255 - th
    coords = cv2.findNonZero(inv)
    if coords is None:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # minAreaRect angle is in [-90, 0); convert to a small deskew angle.
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.1:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _best_attempt(attempts: list[_OsdAttempt]) -> _OsdAttempt:
    def key(a: _OsdAttempt) -> float:
        # None confidence sorts last.
        return a.confidence if a.confidence is not None else float("-inf")

    return max(attempts, key=key)


def _ocr_rotation_score(image: np.ndarray) -> Optional[tuple[int, float, float, int]]:
    """
    Fallback when OSD confidence is persistently low:
    score rotations by running light OCR and picking the rotation that yields
    the highest average word confidence with enough detected text.

    Returns: (best_rotate_degrees, score, avg_word_conf_0_100, word_count)
    """
    best: Optional[tuple[int, float, float, int]] = None
    for deg in (0, 90, 180, 270):
        rotated = _rotate_90(image, (deg // 90) % 4)
        data = pytesseract.image_to_data(rotated, config="--psm 6", output_type=Output.DICT)
        confs: list[float] = []
        texts = data.get("text") or []
        raw_confs = data.get("conf") or []
        for t, c in zip(texts, raw_confs):
            t = (t or "").strip()
            if not t:
                continue
            try:
                cf = float(c)
            except (TypeError, ValueError):
                continue
            if cf < 0:
                continue
            confs.append(cf)

        if not confs:
            continue
        avg = float(sum(confs) / len(confs))  # 0..100
        n = len(confs)
        # Heuristic: prefer rotations with more readable words and higher confidence.
        score = avg * (1.0 + (n ** 0.5))
        cand = (deg, score, avg, n)
        if best is None or cand[1] > best[1]:
            best = cand
    return best


def detect_orientation(image_bytes: bytes, *, include_debug: bool = False) -> OrientationResult:
    """
    Detect how the page is rotated using Tesseract OSD (--psm 0).

    Uses OpenCV imdecode (grayscale) for consistent results with the original CLI.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode image")

    base = _run_osd(image)
    attempts: list[_OsdAttempt] = [_OsdAttempt("original", base.rotate_degrees, base.confidence, base.raw_osd)]

    # If OSD is unsure, retry with a few cheap variants and pick the highest-confidence result.
    if (base.confidence or 0.0) < _LOW_CONFIDENCE_RETRY_THRESHOLD:
        try:
            for name, part in _split_halves(image):
                a = _run_osd(part)
                attempts.append(_OsdAttempt(name, a.rotate_degrees, a.confidence, a.raw_osd))
        except Exception:
            # Keep going; these are best-effort fallbacks.
            pass

        try:
            deskewed = _deskew(image)
            a = _run_osd(deskewed)
            attempts.append(_OsdAttempt("deskew", a.rotate_degrees, a.confidence, a.raw_osd))
        except Exception:
            pass

        try:
            cleaned = _upsample(_binarize(image), scale=2.0)
            a = _run_osd(cleaned)
            attempts.append(_OsdAttempt("binarize_upsample", a.rotate_degrees, a.confidence, a.raw_osd))
        except Exception:
            pass

        # Final fallback: try 90° pre-rotations. This helps in cases where OSD
        # is confused between 180° and 0°/90° due to page layout symmetry.
        try:
            for deg in (90, 180, 270):
                attempts.append(
                    _run_osd_with_prerotate(
                        image,
                        prerotate_degrees=deg,
                        name=f"prerotate_{deg}",
                    )
                )
        except Exception:
            pass

        # If still low confidence after OSD retries, use OCR-based scoring.
        best_so_far = _best_attempt(attempts)
        if (best_so_far.confidence or 0.0) < _LOW_CONFIDENCE_RETRY_THRESHOLD:
            try:
                cleaned = _upsample(_binarize(image), scale=2.0)
                scored = _ocr_rotation_score(cleaned)
                if scored is not None:
                    deg, score, avg100, n = scored
                    attempts.append(
                        _OsdAttempt(
                            name=f"ocr_sweep(avg={avg100:.1f},n={n})",
                            rotate_degrees=deg,
                            # Normalize 0..100 OCR confidence to roughly 0..10 scale.
                            confidence=avg100 / 10.0,
                            raw_osd=f"OCR score={score:.2f} avg_conf={avg100:.1f} n_words={n}",
                        )
                    )
            except Exception:
                pass

    best = _best_attempt(attempts)

    return OrientationResult(
        orientation=ORIENTATION_MAP.get(best.rotate_degrees, "upright"),
        confidence=best.confidence,
        rotate_degrees=best.rotate_degrees,
        raw_osd=(
            (
                f"[chosen={best.name}; tried="
                + ", ".join(
                    f"{a.name}:{a.rotate_degrees}/{a.confidence if a.confidence is not None else 'None'}"
                    for a in attempts
                )
                + "]\n"
                + best.raw_osd
            )
            if include_debug
            else None
        ),
    )
