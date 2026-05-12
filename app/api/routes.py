"""REST API for page orientation detection."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.blankness import BlanknessResult, detect_blankness
from app.services.orientation import detect_orientation, OrientationResult

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/detect")
async def detect(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> dict:
    if file is not None:
        image_bytes = await file.read()
    else:
        image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Use multipart form field 'file' or raw PNG body.",
        )

    debug = request.query_params.get("debug", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )

    def _run_detection() -> tuple[BlanknessResult, Optional[OrientationResult]]:
        blankness = detect_blankness(image_bytes)
        if blankness.classification != "content":
            return blankness, None
        orientation = detect_orientation(image_bytes, include_debug=debug)
        return blankness, orientation

    try:
        blankness_result, orientation_result = await run_in_threadpool(_run_detection)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    out: dict = {
        "blank": blankness_result.classification == "blank",
        "blankness_score": blankness_result.confidence,
        "orientation": (
            orientation_result.orientation if orientation_result is not None else None
        ),
        "orientation_confidence": (
            orientation_result.confidence if orientation_result is not None else None
        ),
        "orientation_method": (
            orientation_result.method if orientation_result is not None else None
        ),
    }
    if debug:
        out["blankness_debug"] = blankness_result.debug_payload()
        if orientation_result is not None:
            out["rotate_degrees"] = orientation_result.rotate_degrees
            if orientation_result.raw_osd is not None:
                out["raw_osd"] = orientation_result.raw_osd
    return out
