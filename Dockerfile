# Single Dockerfile: pass APP_ENV at build time (staging|production) via fly.toml [build.args]
FROM python:3.11-slim
ARG APP_ENV=production
ENV APP_ENV=${APP_ENV}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tesseract + system libs for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Fly captures stdout/stderr; enable access + error logs.
# Fly sets $PORT. Uvicorn worker for ASGI.
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:${PORT:-8080} --timeout ${GUNICORN_TIMEOUT:-180} --access-logfile - --error-logfile - app.main:app"]
