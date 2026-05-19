"""FastAPI app factory for the HandWrite-as-Service module.

FastAPI and uvicorn are optional dependencies. The factory functions in this
module gracefully degrade when those packages are missing so the rest of the
``handwrite.service`` package is still importable.
"""

from __future__ import annotations

from typing import Any, Optional

from .billing import BillingMeter
from .endpoints import (
    InvalidRequest,
    RateLimitExceeded,
    digitize_handler,
    generate_handler,
    health_handler,
    inspect_handler,
    note_session_handler,
    styles_handler,
)


__all__ = ["create_app", "fastapi_available", "FASTAPI_AVAILABLE", "DEFAULT_DAILY_CHAR_QUOTA"]


DEFAULT_DAILY_CHAR_QUOTA = 50_000


try:  # pragma: no cover - exercised when fastapi is installed
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, Response

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when fastapi missing
    FastAPI = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False


def fastapi_available() -> bool:
    """Return ``True`` when FastAPI/uvicorn are importable."""

    return bool(FASTAPI_AVAILABLE)


def create_app(
    *,
    meter: Optional[BillingMeter] = None,
    max_chars_per_day: int = DEFAULT_DAILY_CHAR_QUOTA,
) -> Optional["FastAPI"]:
    """Return a configured :class:`FastAPI` app, or ``None`` if FastAPI is missing.

    Parameters
    ----------
    meter:
        Optional pre-existing :class:`BillingMeter`. A fresh one is created
        otherwise.
    max_chars_per_day:
        Daily character quota per API key. ``0`` disables enforcement.
    """

    if not FASTAPI_AVAILABLE:
        return None

    billing = meter or BillingMeter()
    app = FastAPI(title="HandWrite Service", version="0.1.0")
    app.state.meter = billing
    app.state.max_chars_per_day = int(max_chars_per_day)

    @app.exception_handler(InvalidRequest)
    async def _on_invalid(request, exc):  # type: ignore[override]
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RateLimitExceeded)
    async def _on_rate_limit(request, exc):  # type: ignore[override]
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.get("/health")
    async def _health() -> Any:
        return health_handler()

    @app.get("/v1/styles")
    async def _styles() -> Any:
        return styles_handler()

    @app.post("/v1/inspect")
    async def _inspect(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Any:
        payload = await _safe_json(request)
        return inspect_handler(
            payload,
            api_key=x_api_key,
            meter=billing,
            max_chars_per_day=app.state.max_chars_per_day,
        )

    @app.post("/v1/generate")
    async def _generate(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Any:
        payload = await _safe_json(request)
        png_bytes, headers = generate_handler(
            payload,
            api_key=x_api_key,
            meter=billing,
            max_chars_per_day=app.state.max_chars_per_day,
        )
        return Response(content=png_bytes, media_type="image/png", headers=headers)

    @app.post("/v1/note-session")
    async def _note_session(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Any:
        payload = await _safe_json(request)
        return note_session_handler(
            payload,
            api_key=x_api_key,
            meter=billing,
            max_chars_per_day=app.state.max_chars_per_day,
        )

    @app.post("/v1/digitize")
    async def _digitize(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Any:
        body = await request.body()
        return digitize_handler(
            body,
            api_key=x_api_key,
            meter=billing,
            max_chars_per_day=app.state.max_chars_per_day,
        )

    @app.get("/v1/usage")
    async def _usage(
        x_api_key: Optional[str] = Header(default=None),
    ) -> Any:
        if not x_api_key:
            raise HTTPException(status_code=400, detail="missing X-API-Key header")
        return billing.usage(x_api_key)

    return app


async def _safe_json(request: "Request") -> dict:
    try:
        data = await request.json()
    except Exception as exc:  # pragma: no cover - depends on http client
        raise InvalidRequest(f"invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidRequest("request body must be a JSON object")
    return data


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app via uvicorn. Raises ``RuntimeError`` if unavailable."""

    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install fastapi uvicorn` "
            "to run the HandWrite service."
        )
    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on uvicorn
        raise RuntimeError("uvicorn is required to run the service") from exc

    app = create_app()
    uvicorn.run(app, host=host, port=port)
