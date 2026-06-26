# page-orientation-api

REST API (FastAPI) with two independent endpoints for PNG page analysis:

- **`/blankness`** — classify a page as `blank` or `content`
- **`/orientation`** — detect page rotation with **Tesseract OSD** (`--psm 0`);
  returns one of: `upright`, `rotated_left`, `rotated_right`, `upside_down`

No image is rotated in the response.

## Stack (all free / open-source)

- **FastAPI** (MIT) · **Uvicorn** (BSD) · **Gunicorn** (MIT)
- **Tesseract** (Apache-2.0) + **pytesseract** (Apache-2.0)
- **OpenCV** (Apache-2.0) · **NumPy** (BSD)

## Local run

Requires [Tesseract](https://github.com/tesseract-ocr/tesseract) on your PATH.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

- `GET /healthz` — health check
- `POST /blankness` — `multipart/form-data` field `file` (PNG), or raw bytes body
  - Response: `blank` (`true`/`false`), `blankness_score`
  - Optional `?debug=1` adds `blankness_debug`
- `POST /orientation` — `multipart/form-data` field `file` (PNG), or raw bytes body
  - Response: `orientation`, `orientation_confidence`
  - Optional `?debug=1` adds `orientation_method`, `rotate_degrees`, and `raw_osd`

## Tests

Sample PNGs live in `samples/blankness/` and `samples/rotation/`. Unit tests use synthetic images; integration tests use those folders when present.

```bash
pytest tests/ -q
```

## Git branches → Fly.io

| Branch   | Environment | Fly org          | App name                  | Config                 |
|----------|-------------|------------------|---------------------------|------------------------|
| `release` | Staging     | `stg-goenablers` | `page-orientation-api-stg` | [fly.staging.toml](fly.staging.toml) |
| `master`  | Production  | `prd-goenablers` | `page-orientation-api-prd`   | [fly.production.toml](fly.production.toml) |

GitHub Actions: [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — set secrets `FLY_API_TOKEN_STG` and `FLY_API_TOKEN_PRD` (per-app deploy tokens from `fly tokens create deploy`).

## One-time Fly.io setup (example)

```bash
fly apps create page-orientation-api-stg --org stg-goenablers
fly apps create page-orientation-api-prd --org prd-goenablers   # requires prd org to exist
fly tokens create deploy -a page-orientation-api-stg
fly tokens create deploy -a page-orientation-api-prd
# Add the tokens to GitHub repo secrets (see table above)
```

**Staging** is available at: `https://page-orientation-api-stg.fly.dev` after deploy.

Pushing the repo to **GitHub** and configuring secrets is described in [docs/SETUP_GITHUB_AND_PROD.md](docs/SETUP_GITHUB_AND_PROD.md) (if `gh` is not installed on your machine, use a manual `git remote` as documented there).

## Docker

```bash
docker build --build-arg APP_ENV=production -t page-orientation-api .
docker run -e PORT=8080 -p 8080:8080 page-orientation-api
```

## Project layout

- `app/main.py` — FastAPI app
- `app/api/routes.py` — HTTP routes
- `app/services/blankness.py` — blank vs content page classification
- `app/services/orientation.py` — Tesseract OSD / rotation label logic
- `app/core/config.py` — settings (`APP_ENV`, `PORT`)
