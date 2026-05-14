"""REST API for page blankness and orientation detection."""

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


async def _read_image(request: Request, file: Optional[UploadFile]) -> bytes:
    if file is not None:
        image_bytes = await file.read()
    else:
        image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Use multipart form field 'file' or raw PNG body.",
        )
    return image_bytes


def _is_debug(request: Request) -> bool:
    return request.query_params.get("debug", "").lower() in ("1", "true", "yes", "y")


@router.post("/blankness")
async def blankness(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> dict:
    image_bytes = await _read_image(request, file)
    debug = _is_debug(request)

    try:
        result: BlanknessResult = await run_in_threadpool(detect_blankness, image_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    out: dict = {
        "blank": result.classification == "blank",
        "blankness_score": result.confidence,
    }
    if debug:
        out["blankness_debug"] = result.debug_payload()
    return out


@router.post("/orientation")
async def orientation(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
) -> dict:
    image_bytes = await _read_image(request, file)
    debug = _is_debug(request)

    try:
        result: OrientationResult = await run_in_threadpool(
            detect_orientation, image_bytes, include_debug=debug
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    out: dict = {
        "orientation": result.orientation,
        "orientation_confidence": result.confidence,
    }
    if debug:
        out["orientation_method"] = result.method
        out["rotate_degrees"] = result.rotate_degrees
        if result.raw_osd is not None:
            out["raw_osd"] = result.raw_osd
    return out
