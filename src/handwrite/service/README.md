# HandWrite-as-Service

A small FastAPI HTTP wrapper around the HandWrite Python API. It is shipped as
an **optional** module: FastAPI and uvicorn are not declared as hard
dependencies, so the rest of the package keeps working without them. Install
them only when you want to run the service.

## Quickstart

```bash
pip install handwrite fastapi "uvicorn[standard]"

# Run with the helper:
python -c "from handwrite.service import run_server; run_server(host='0.0.0.0', port=8000)"

# Or directly with uvicorn using the factory pattern:
uvicorn handwrite.service.app:create_app --factory --host 0.0.0.0 --port 8000
```

The service exposes:

| Method | Path              | Description                                  |
| ------ | ----------------- | -------------------------------------------- |
| GET    | `/health`         | Liveness probe                                |
| GET    | `/v1/styles`      | List of built-in handwriting styles           |
| POST   | `/v1/inspect`     | Coverage / realism report for some text       |
| POST   | `/v1/generate`    | PNG bytes for a generated page                |
| POST   | `/v1/note-session`| Full classroom-note session manifest          |
| POST   | `/v1/digitize`    | Recognise a handwritten image (bytes body)    |
| GET    | `/v1/usage`       | Per-API-key usage snapshot                    |

All non-public endpoints require an `X-API-Key` header. Usage is metered
per key and the daily character budget is enforced (default 50,000 chars
per UTC day; set `HANDWRITE_MAX_CHARS_PER_DAY` to override).

## curl examples

```bash
# Health
curl -s http://localhost:8000/health

# List styles
curl -s http://localhost:8000/v1/styles

# Inspect text
curl -s -X POST http://localhost:8000/v1/inspect \
    -H "X-API-Key: demo-key" \
    -H "Content-Type: application/json" \
    -d '{"text":"今天上课复习牛顿第二定律。","style":"行书流畅"}'

# Generate a page (returns PNG bytes)
curl -s -X POST http://localhost:8000/v1/generate \
    -H "X-API-Key: demo-key" \
    -H "Content-Type: application/json" \
    -d '{"text":"hello","style":"行书流畅"}' \
    --output page.png

# Note session manifest
curl -s -X POST http://localhost:8000/v1/note-session \
    -H "X-API-Key: demo-key" \
    -H "Content-Type: application/json" \
    -d '{"text":"hello"}'

# Usage report
curl -s http://localhost:8000/v1/usage -H "X-API-Key: demo-key"
```

## Docker

```bash
# From the repo root
docker compose -f src/handwrite/service/docker-compose.yml up --build
```

The compose file performs a multi-stage `python:3.11-slim` build and runs the
service as the non-root `handwrite` user on port 8000.

## Programmatic billing meter

You can use `BillingMeter` standalone — it does not require FastAPI:

```python
from handwrite.service import BillingMeter

meter = BillingMeter()
meter.record("client-123", action="generate", chars=42)
meter.check_quota("client-123", max_chars_per_day=1000)  # -> True
meter.usage("client-123")  # -> dict snapshot
```
