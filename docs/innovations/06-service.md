# Innovation #6 — HandWrite-as-Service

The `handwrite.service` module turns the local HandWrite generator into a multi-tenant HTTP service. It is delivered as an **optional** capability: FastAPI and uvicorn are not promoted into `pyproject.toml`, so the rest of the package keeps importing cleanly even when the web stack is missing.

## Endpoints

| Method | Path              | Purpose                                            |
| ------ | ----------------- | -------------------------------------------------- |
| GET    | `/health`         | Liveness probe                                     |
| GET    | `/v1/styles`      | Built-in handwriting styles catalog                |
| POST   | `/v1/inspect`     | Coverage / realism pre-check for note text         |
| POST   | `/v1/generate`    | Returns rendered PNG bytes                         |
| POST   | `/v1/note-session`| Full classroom-note manifest                       |
| POST   | `/v1/digitize`    | Recognises an uploaded handwritten image           |
| GET    | `/v1/usage`       | Per-key usage snapshot for billing reconciliation  |

All write endpoints require an `X-API-Key` header. The `BillingMeter` records each call (action + character count), keys usage by UTC day.

## Deployment story

The bundled `Dockerfile` is a multi-stage `python:3.11-slim` build. `docker-compose.yml` wires this image to port 8000 with environment-driven configuration.

## Testing & isolation

The handlers are split out of the FastAPI app so they can be unit-tested without importing FastAPI. The integration tests use `pytest.importorskip("fastapi")`.
