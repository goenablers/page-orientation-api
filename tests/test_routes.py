"""Route-level tests for blank-first detection flow."""

from __future__ import annotations

import asyncio

from starlette.requests import Request

from app.api import routes
from app.services.blankness import BlanknessMetrics, BlanknessResult
from app.services.orientation import OrientationResult


def _blankness_result(classification: str) -> BlanknessResult:
    return BlanknessResult(
        classification=classification,
        confidence=0.91,
        metrics=BlanknessMetrics(
            background_level=252.0,
            strong_threshold=220,
            weak_threshold=236,
            strong_ink_ratio=0.0001,
            weak_ink_ratio=0.0008,
            edge_density=0.0005,
            component_count=1,
            large_component_count=0,
            largest_component_ratio=0.0,
            intensity_std=1.2,
        ),
    )


def _request(body: bytes = b"page-bytes", query_string: bytes = b"") -> Request:
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/detect",
            "headers": [],
            "query_string": query_string,
        },
        receive,
    )


def test_detect_skips_orientation_for_blank(monkeypatch) -> None:
    for classification in ("blank",):
        monkeypatch.setattr(
            routes,
            "detect_blankness",
            lambda _image_bytes, classification=classification: _blankness_result(
                classification
            ),
        )

        def _unexpected_orientation(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
            raise AssertionError("Orientation detection should have been skipped")

        monkeypatch.setattr(routes, "detect_orientation", _unexpected_orientation)

        result = asyncio.run(routes.detect(_request(), file=None))

        assert result == {
            "blank": True,
            "blankness_score": 0.91,
            "orientation": None,
            "orientation_confidence": None,
        }


def test_detect_runs_orientation_for_content(monkeypatch) -> None:
    monkeypatch.setattr(routes, "detect_blankness", lambda _image_bytes: _blankness_result("content"))
    monkeypatch.setattr(
        routes,
        "detect_orientation",
        lambda _image_bytes, *, include_debug: OrientationResult(
            orientation="upright",
            confidence=12.5,
            rotate_degrees=0,
            raw_osd="Rotate: 0" if include_debug else None,
        ),
    )

    result = asyncio.run(routes.detect(_request(query_string=b"debug=1"), file=None))

    assert result == {
        "blank": False,
        "blankness_score": 0.91,
        "orientation": "upright",
        "orientation_confidence": 12.5,
        "blankness_debug": {
            "background_level": 252.0,
            "strong_threshold": 220,
            "weak_threshold": 236,
            "strong_ink_ratio": 0.0001,
            "weak_ink_ratio": 0.0008,
            "edge_density": 0.0005,
            "component_count": 1,
            "large_component_count": 0,
            "largest_component_ratio": 0.0,
            "intensity_std": 1.2,
            "blankness_score": 0.91,
        },
        "rotate_degrees": 0,
        "raw_osd": "Rotate: 0",
    }
