"""FastAPI application entry (used by gunicorn: app.main:app)."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Page Orientation API",
    description="Detect PNG page orientation using Tesseract OSD (no image rotation in response).",
    version="1.0.0",
)

app.include_router(router)
