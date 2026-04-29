"""REST API for page orientation detection."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.orientation import detect_orientation, OrientationResult

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile | None = File(default=None),
) -> dict:
    if file is not None:
        image_bytes = await file.read()
    else:
        # If the client sent multipart but omitted the 'file' field,
        # FastAPI has already consumed the request stream while parsing form data.
        # In that scenario, reading request.body() raises "Stream consumed".
        content_type = (request.headers.get("content-type") or "").lower()
        if content_type.startswith("multipart/"):
            image_bytes = b""
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
    def _run_osd() -> OrientationResult:
        return detect_orientation(image_bytes, include_debug=debug)

    try:
        result = await run_in_threadpool(_run_osd)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    out: dict = {
        "orientation": result.orientation,
        "confidence": result.confidence,
    }
    if debug:
        out["rotate_degrees"] = result.rotate_degrees
        if result.raw_osd is not None:
            out["raw_osd"] = result.raw_osd
    return out
