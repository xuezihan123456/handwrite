"""HandWrite-as-Service: optional FastAPI multi-tenant HTTP layer."""

from __future__ import annotations

from .app import (
    DEFAULT_DAILY_CHAR_QUOTA,
    FASTAPI_AVAILABLE,
    create_app,
    fastapi_available,
    run_server,
)
from .billing import BillingMeter, UsageEvent
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


__all__ = [
    "BillingMeter",
    "UsageEvent",
    "create_app",
    "fastapi_available",
    "run_server",
    "FASTAPI_AVAILABLE",
    "DEFAULT_DAILY_CHAR_QUOTA",
    "InvalidRequest",
    "RateLimitExceeded",
    "health_handler",
    "styles_handler",
    "inspect_handler",
    "generate_handler",
    "note_session_handler",
    "digitize_handler",
]
