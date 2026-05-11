"""Tests for the blank/near-blank/content classifier."""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np

from app.services.blankness import detect_blankness


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _blank_page() -> np.ndarray:
    return np.full((1400, 1000), 255, dtype=np.uint8)


def _marked_blank_page() -> np.ndarray:
    """White page with scattered stray marks but no readable structure."""
    image = _blank_page()
    cv2.line(image, (120, 240), (230, 255), 150, 2)
    cv2.circle(image, (700, 1050), 10, 170, -1)
    cv2.putText(
        image,
        "xx",
        (350, 780),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        205,
        1,
        cv2.LINE_AA,
    )
    return image


def _content_page() -> np.ndarray:
    image = _blank_page()
    rows = (
        "Invoice 10024",
        "Customer: Example Co.",
        "Amount due: 1290.45",
        "Please remit payment within 30 days.",
        "Thank you for your business.",
    )
    start_y = 220
    for idx, row in enumerate(rows):
        cv2.putText(
            image,
            row,
            (90, start_y + idx * 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            20,
            3,
            cv2.LINE_AA,
        )
    return image


def test_detect_blankness_blank_page() -> None:
    result = detect_blankness(_encode_png(_blank_page()))

    assert result.classification == "blank"
    assert result.metrics.largest_component_ratio < 0.010
    assert result.metrics.intensity_std < 30.0


def test_detect_blankness_marked_blank_page() -> None:
    """Stray marks with no text structure should still be classified as blank."""
    result = detect_blankness(_encode_png(_marked_blank_page()))

    assert result.classification == "blank"
    assert result.metrics.large_component_count < 40


def _small_label_page() -> np.ndarray:
    """White page with only a small label like 'Pg 1' — structurally near-blank but readable."""
    image = _blank_page()
    cv2.putText(
        image,
        "Pg 1",
        (70, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        0,
        2,
        cv2.LINE_AA,
    )
    return image


def test_detect_blankness_content_page() -> None:
    result = detect_blankness(_encode_png(_content_page()))

    assert result.classification == "content"
    assert result.metrics.large_component_count >= 40 or result.metrics.intensity_std >= 30.0


def test_detect_blankness_small_label_page() -> None:
    """A page with only a small readable label should be classified as content."""
    result = detect_blankness(_encode_png(_small_label_page()))

    assert result.classification == "content"
