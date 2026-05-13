"""OpenCV-based page blankness classification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import cv2  # type: ignore
import numpy as np
import pytesseract  # type: ignore


@dataclass(frozen=True)
class BlanknessMetrics:
    background_level: float
    strong_threshold: int
    weak_threshold: int
    strong_ink_ratio: float
    weak_ink_ratio: float
    edge_density: float
    component_count: int
    large_component_count: int
    largest_component_ratio: float
    intensity_std: float


@dataclass(frozen=True)
class BlanknessResult:
    classification: str
    confidence: float
    metrics: BlanknessMetrics

    def debug_payload(self) -> dict[str, Any]:
        out = asdict(self.metrics)
        out["blankness_score"] = self.confidence
        return out


def _decode_grayscale(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _crop_page_region(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_y = min(max(4, height // 50), max(height // 4, 1))
    margin_x = min(max(4, width // 50), max(width // 4, 1))
    cropped = image[margin_y : height - margin_y, margin_x : width - margin_x]
    if cropped.size == 0:
        return image
    return cropped


def _connected_component_stats(mask: np.ndarray) -> tuple[int, int, float]:
    area = int(mask.size)
    if area == 0 or not np.any(mask):
        return 0, 0, 0.0

    mask_u8 = mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    min_area = max(8, area // 50000)
    large_area = max(min_area * 8, area // 5000)

    component_count = 0
    large_component_count = 0
    largest_component = 0
    for label_idx in range(1, num_labels):
        component_area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if component_area < min_area:
            continue
        component_count += 1
        if component_area >= large_area:
            large_component_count += 1
        largest_component = max(largest_component, component_area)

    largest_component_ratio = largest_component / float(area)
    return component_count, large_component_count, largest_component_ratio


_OCR_MIN_CONF = 50
# Small-label path: a compact cluster of up to this many tokens is accepted
# as a label/page-number annotation.
_OCR_MAX_TOKENS_LABEL = 20
_OCR_MAX_SPREAD_LABEL = 0.25  # max bounding-box fraction for a label
# Full-page path: a real document page will have many confident tokens that
# together cover a meaningful fraction of the page.
_OCR_MIN_TOKENS_PAGE = 30
_OCR_MIN_SPREAD_PAGE = 0.10  # token bounding box must span at least 10 % of page
_OCR_MIN_AVG_CONF_PAGE = 70  # require genuinely confident text, not OCR noise


def _has_readable_text(image: np.ndarray) -> bool:
    """
    Return True when OCR finds readable alphanumeric content.

    Two separate rules cover the two cases we care about:

    Label rule (original):
      A spatially compact cluster of up to 20 confident tokens indicates a
      small annotation like a page number or a brief label.

    Full-page rule (new):
      A real document page produces many confident tokens spread across a
      substantial portion of the image.  Where the structural pass fails
      (e.g. thin text on a sparse page) this rule catches the document.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        h_img, w_img = image.shape[:2]
        page_area = float(max(h_img * w_img, 1))

        xs_left: list[int] = []
        ys_top: list[int] = []
        xs_right: list[int] = []
        ys_bottom: list[int] = []
        confs: list[int] = []

        for conf, text, x, y, w, h in zip(
            data["conf"],
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
        ):
            try:
                if int(conf) >= _OCR_MIN_CONF and re.search(r"[A-Za-z0-9]", text):
                    xs_left.append(int(x))
                    ys_top.append(int(y))
                    xs_right.append(int(x) + int(w))
                    ys_bottom.append(int(y) + int(h))
                    confs.append(int(conf))
            except (ValueError, TypeError):
                pass

        count = len(xs_left)
        if count == 0:
            return False

        spread = (
            (max(xs_right) - min(xs_left))
            * (max(ys_bottom) - min(ys_top))
            / page_area
        )

        # Label rule: small compact cluster.
        if count <= _OCR_MAX_TOKENS_LABEL and spread <= _OCR_MAX_SPREAD_LABEL:
            return True

        # Full-page rule: many confident tokens spread across the page.
        avg_conf = sum(confs) / len(confs)
        if (
            count >= _OCR_MIN_TOKENS_PAGE
            and spread >= _OCR_MIN_SPREAD_PAGE
            and avg_conf >= _OCR_MIN_AVG_CONF_PAGE
        ):
            return True

        return False

    except Exception:  # noqa: BLE001
        return False


def detect_blankness(image_bytes: bytes) -> BlanknessResult:
    """
    Classify a page as blank or content.

    The decision is based on structural signals rather than raw ink coverage.
    Scattered marks, smudges, faint stamps, or unreadable stray writing produce
    many small components but no large grouped regions; those pages are blank.
    Real content pages have substantial component area and high pixel variance.

    Key thresholds are intentionally conservative and easy to adjust against
    a labeled sample set.
    """

    image = _decode_grayscale(image_bytes)
    cropped = _crop_page_region(image)
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

    background_level = float(np.percentile(blurred, 95))
    strong_threshold = int(np.clip(background_level - 32, 96, 245))
    weak_threshold = int(np.clip(background_level - 18, 128, 250))

    strong_mask = blurred < strong_threshold
    weak_mask = blurred < weak_threshold

    # Merge fragmented letter strokes into word-level blobs before measuring
    # connected components.  A 3×3 closing pass is enough to join the gaps
    # that thin or anti-aliased text creates, without distorting coarse shapes.
    _morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged_mask = cv2.morphologyEx(
        strong_mask.astype(np.uint8), cv2.MORPH_CLOSE, _morph_kernel
    ).astype(bool)

    component_count, large_component_count, largest_component_ratio = (
        _connected_component_stats(merged_mask)
    )

    strong_ink_ratio = float(np.mean(strong_mask))
    weak_ink_ratio = float(np.mean(weak_mask))
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
    edge_density = float(np.mean(np.abs(laplacian) > 12.0))
    intensity_std = float(np.std(blurred))

    metrics = BlanknessMetrics(
        background_level=background_level,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        strong_ink_ratio=strong_ink_ratio,
        weak_ink_ratio=weak_ink_ratio,
        edge_density=edge_density,
        component_count=component_count,
        large_component_count=large_component_count,
        largest_component_ratio=largest_component_ratio,
        intensity_std=intensity_std,
    )

    # A page counts as content only when it shows real structural density:
    # grouped regions large enough to be letters/words, and overall pixel
    # variance consistent with a page of text or a form.
    has_substantial_structure = (
        largest_component_ratio >= 0.010
        or large_component_count >= 40
        or intensity_std >= 30.0
    )

    if not has_substantial_structure:
        # OCR fallback: a page can look structurally blank but still carry a
        # small label, page number, or brief annotation. If readable
        # alphanumeric text is found, treat the page as content.
        if _has_readable_text(cropped):
            confidence = min(
                0.99,
                0.55
                + min(0.20, largest_component_ratio * 2.5)
                + min(0.15, large_component_count / 300.0)
                + min(0.10, intensity_std / 600.0),
            )
            return BlanknessResult("content", confidence, metrics)

        confidence = min(
            0.99,
            0.70
            + max(0.0, (0.010 - largest_component_ratio) * 2.0)
            + max(0.0, (30.0 - intensity_std) / 60.0),
        )
        return BlanknessResult("blank", confidence, metrics)

    confidence = min(
        0.99,
        0.55
        + min(0.20, largest_component_ratio * 2.5)
        + min(0.15, large_component_count / 300.0)
        + min(0.10, intensity_std / 600.0),
    )
    return BlanknessResult("content", confidence, metrics)
