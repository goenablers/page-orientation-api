"""Tesseract OSD-based page rotation detection (service layer)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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

# Tesseract's "Rotate" value already IS the CW degrees to apply to reach upright,
# so we use it directly when rotating candidate images for OCR verification.

# ── Stage-local stop thresholds ──────────────────────────────────────────────
#
# Stage 1 — raw OSD
#   Stop if raw OSD confidence >= OSD_ACCEPT.
#
# Stage 2 — preprocessed OSD
#   Stop if preprocessed OSD confidence >= OSD_PRE_ACCEPT.
#
# Stage 3 — 2-way OCR
#   Run OCR on the two candidates that belong to the axis family hinted by the
#   best OSD pass so far.  Stop if the OCR margin ratio >= OCR_2WAY_MARGIN.
#   The margin ratio = (winner_score - loser_score) / (winner_score + loser_score).
#   A value of 0.5 means the winner holds >= 75 % of the combined score.
#
# Stage 4 — 4-way OCR  (final fallback, always stops)
#   Score all four candidate rotations and return the best.
#
_OSD_ACCEPT = 6.0
_OSD_PRE_ACCEPT = 4.0
_OCR_2WAY_MARGIN = 0.5

# Candidate pairs for the 2-way OCR stage, keyed by the OSD-hinted Rotate value.
_AMBIGUOUS_PAIRS: dict[int, tuple[int, int]] = {
    90: (90, 270),   # left vs right
    270: (90, 270),
    0: (0, 180),     # upright vs upside-down
    180: (0, 180),
}

_ALL_ROTATIONS = (0, 90, 180, 270)

# Minimum Tesseract per-token confidence to count a word.
_OCR_MIN_CONF = 50
# Minimum word length to earn the longer-word bonus.
_OCR_MIN_WORD_LEN = 3
# Junk-token penalty pattern.
_JUNK_RE = re.compile(r"[|/\[\]=\\{}]")

_RE_KV = re.compile(r"^([A-Za-z ()]+):\s*(.+?)\s*$")

# Languages used for OCR verification — keep tight to your document set.
_OCR_LANG = "por+eng+spa+fra+ita+deu"


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
    # Public confidence: always the native score from the first raw OSD pass.
    # Use `method` to understand how the final answer was reached.
    confidence: float
    method: str
    rotate_degrees: int
    raw_osd: str | None = None


@dataclass(frozen=True)
class _OcrResult:
    rotate_degrees: int
    margin: float   # (winner - loser) / (winner + loser), in [0, 1]


def _run_osd(image: np.ndarray) -> tuple[int, float, str]:
    """Run Tesseract OSD on a grayscale image.

    Returns (rotate_deg, orientation_confidence, raw_osd_text).
    """
    try:
        raw = pytesseract.image_to_osd(image, config="--psm 0")
    except pytesseract.pytesseract.TesseractError as exc:
        if "Too few characters" in str(exc):
            return 0, 0.0, str(exc)
        raise

    parsed = _parse_osd(raw)
    rotate_raw = parsed.get("Rotate", 0)
    rotate = int(rotate_raw) % 360 if rotate_raw is not None else 0

    conf_raw = parsed.get("Orientation confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0

    return rotate, confidence, raw


def _preprocess_for_osd(image: np.ndarray) -> np.ndarray:
    """Return a text-friendly variant for a second OSD pass.

    Crops 5 % margins, binarises with Otsu, and upscales small images.
    """
    h, w = image.shape[:2]
    my = max(4, h // 20)
    mx = max(4, w // 20)
    cropped = image[my: h - my, mx: w - mx]
    if cropped.size == 0:
        cropped = image

    _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ch, cw = binary.shape[:2]
    if min(ch, cw) < 600:
        scale = 600 / min(ch, cw)
        binary = cv2.resize(binary, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)

    return binary


def _rotate_image(image: np.ndarray, rotate_deg: int) -> np.ndarray:
    """Rotate image clockwise by rotate_deg degrees."""
    if rotate_deg == 0:
        return image
    if rotate_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotate_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _ocr_quality_score(image: np.ndarray) -> float:
    """Score OCR readability on an image assumed to be upright.

    Higher is better.  Rewards confident alphanumeric tokens of reasonable
    length; penalises junk punctuation.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            lang=_OCR_LANG,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:  # noqa: BLE001
        return 0.0

    score = 0.0
    for conf, text in zip(data["conf"], data["text"]):
        try:
            c = int(conf)
        except (TypeError, ValueError):
            continue
        if c < _OCR_MIN_CONF or not text:
            continue
        if not re.search(r"[A-Za-z0-9]", text):
            continue
        token_score = float(c)
        if len(text) >= _OCR_MIN_WORD_LEN:
            token_score *= 1.5
        if _JUNK_RE.search(text):
            token_score *= 0.3
        score += token_score
    return score


def _ocr_run(*candidates: int, image: np.ndarray) -> _OcrResult:
    """Score all candidates, return the winner and the margin ratio.

    margin = (best_score - second_best_score) / (best_score + second_best_score)
    For a 2-candidate comparison this is the standard normalised gap.
    For 4 candidates it compares the winner against its nearest rival.
    """
    scores = {deg: _ocr_quality_score(_rotate_image(image, deg))
              for deg in candidates}

    sorted_degs = sorted(scores, key=lambda d: scores[d], reverse=True)
    best = sorted_degs[0]
    best_score = scores[best]
    second_score = scores[sorted_degs[1]] if len(sorted_degs) > 1 else 0.0

    total = best_score + second_score
    margin = (best_score - second_score) / total if total > 0 else 0.0

    return _OcrResult(rotate_degrees=best, margin=round(margin, 4))


def detect_orientation(image_bytes: bytes, *, include_debug: bool = False) -> OrientationResult:
    """
    Progressive cascade — each stage evaluates its own confidence and decides
    whether to stop or hand off to the next stage.

    Stage 1 — raw OSD
      Stop if raw OSD confidence >= 6.0.

    Stage 2 — preprocessed OSD  (binarised + margin-cropped)
      Stop if preprocessed OSD confidence >= 4.0.

    Stage 3 — 2-way OCR
      Use the axis family hinted by the best OSD pass so far.
      Stop if the OCR winner margin >= 0.5  (winner holds >= 75 % of score).

    Stage 4 — 4-way OCR  (final fallback, always stops)
      Score all four rotations and return the best.

    Public `confidence` is always the first raw OSD pass value.
    Use `method` to see which stage produced the final answer.

    Maximum Tesseract calls: 1 OSD + 1 OSD + 2 OCR + 4 OCR = 8.
    No loops; no retries beyond this fixed cascade.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode image")

    # ── Stage 1: raw OSD ────────────────────────────────────────────────────
    rotate, raw_confidence, raw_osd = _run_osd(image)

    if raw_confidence >= _OSD_ACCEPT:
        return OrientationResult(
            orientation=ORIENTATION_MAP.get(rotate, "upright"),
            confidence=raw_confidence,
            method="osd",
            rotate_degrees=rotate,
            raw_osd=raw_osd if include_debug else None,
        )

    # ── Stage 2: preprocessed OSD ───────────────────────────────────────────
    preprocessed = _preprocess_for_osd(image)
    rotate2, confidence2, raw_osd2 = _run_osd(preprocessed)

    if confidence2 >= _OSD_PRE_ACCEPT:
        return OrientationResult(
            orientation=ORIENTATION_MAP.get(rotate2, "upright"),
            confidence=raw_confidence,
            method="osd_preprocessed",
            rotate_degrees=rotate2,
            raw_osd=raw_osd2 if include_debug else None,
        )

    # Pick the better OSD result as the axis hint for stage 3.
    if confidence2 > raw_confidence:
        osd_hint, best_raw = rotate2, raw_osd2
    else:
        osd_hint, best_raw = rotate, raw_osd

    # ── Stage 3: 2-way OCR ──────────────────────────────────────────────────
    pair = _AMBIGUOUS_PAIRS.get(osd_hint, (0, 180))
    ocr2 = _ocr_run(*pair, image=image)

    if ocr2.margin >= _OCR_2WAY_MARGIN:
        return OrientationResult(
            orientation=ORIENTATION_MAP.get(ocr2.rotate_degrees, "upright"),
            confidence=raw_confidence,
            method="ocr_2way",
            rotate_degrees=ocr2.rotate_degrees,
            raw_osd=best_raw if include_debug else None,
        )

    # ── Stage 4: 4-way OCR (final fallback) ─────────────────────────────────
    ocr4 = _ocr_run(*_ALL_ROTATIONS, image=image)

    return OrientationResult(
        orientation=ORIENTATION_MAP.get(ocr4.rotate_degrees, "upright"),
        confidence=raw_confidence,
        method="ocr_4way",
        rotate_degrees=ocr4.rotate_degrees,
        raw_osd=best_raw if include_debug else None,
    )
