"""Route-level tests for the /blankness and /orientation endpoints."""

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


def _orientation_result(include_debug: bool = False) -> OrientationResult:
    return OrientationResult(
        orientation="upright",
        confidence=12.5,
        method="osd",
        rotate_degrees=0,
        raw_osd="Rotate: 0" if include_debug else None,
    )


def _request(body: bytes = b"page-bytes", query_string: bytes = b"", path: str = "/") -> Request:
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": query_string,
        },
        receive,
    )


# ── /blankness ────────────────────────────────────────────────────────────────

def test_blankness_blank(monkeypatch) -> None:
    monkeypatch.setattr(
        routes, "detect_blankness", lambda _bytes: _blankness_result("blank")
    )
    result = asyncio.run(routes.blankness(_request(path="/blankness"), file=None))
    assert result == {"blank": True, "blankness_score": 0.91}


def test_blankness_content(monkeypatch) -> None:
    monkeypatch.setattr(
        routes, "detect_blankness", lambda _bytes: _blankness_result("content")
    )
    result = asyncio.run(routes.blankness(_request(path="/blankness"), file=None))
    assert result == {"blank": False, "blankness_score": 0.91}


def test_blankness_debug(monkeypatch) -> None:
    monkeypatch.setattr(
        routes, "detect_blankness", lambda _bytes: _blankness_result("blank")
    )
    result = asyncio.run(
        routes.blankness(_request(path="/blankness", query_string=b"debug=1"), file=None)
    )
    assert result["blank"] is True
    assert result["blankness_score"] == 0.91
    assert result["blankness_debug"] == {
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
    }


def test_blankness_no_image_returns_400() -> None:
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.blankness(_request(body=b"", path="/blankness"), file=None))
    assert exc_info.value.status_code == 400


# ── /orientation ──────────────────────────────────────────────────────────────

def test_orientation_returns_result(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "detect_orientation",
        lambda _bytes, *, include_debug: _orientation_result(include_debug),
    )
    result = asyncio.run(routes.orientation(_request(path="/orientation"), file=None))
    assert result == {
        "orientation": "upright",
        "orientation_confidence": 12.5,
    }


def test_orientation_debug(monkeypatch) -> None:
    monkeypatch.setattr(
        routes,
        "detect_orientation",
        lambda _bytes, *, include_debug: _orientation_result(include_debug),
    )
    result = asyncio.run(
        routes.orientation(
            _request(path="/orientation", query_string=b"debug=1"), file=None
        )
    )
    assert result == {
        "orientation": "upright",
        "orientation_confidence": 12.5,
        "orientation_method": "osd",
        "rotate_degrees": 0,
        "raw_osd": "Rotate: 0",
    }


def test_orientation_no_image_returns_400() -> None:
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.orientation(_request(body=b"", path="/orientation"), file=None))
    assert exc_info.value.status_code == 400
