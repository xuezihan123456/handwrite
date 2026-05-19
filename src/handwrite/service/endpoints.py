"""Pure-Python route handlers for the HandWrite service.

These handlers do not depend on FastAPI; they accept simple inputs and return
plain dictionaries (or bytes). The thin FastAPI layer in :mod:`app` wraps
them with HTTP request/response semantics. Keeping the logic here means the
core service can be unit-tested without the web stack installed.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .billing import BillingMeter


__all__ = [
    "health_handler",
    "styles_handler",
    "inspect_handler",
    "generate_handler",
    "note_session_handler",
    "digitize_handler",
    "RateLimitExceeded",
    "InvalidRequest",
]


class RateLimitExceeded(Exception):
    """Raised when a request would exceed the per-key daily quota."""


class InvalidRequest(Exception):
    """Raised when request payloads are malformed."""


def _require_api_key(api_key: Optional[str]) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise InvalidRequest("missing X-API-Key header")
    return api_key.strip()


def _require_text(payload: Dict[str, Any], field: str = "text") -> str:
    if not isinstance(payload, dict):
        raise InvalidRequest("request body must be a JSON object")
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequest(f"field '{field}' must be a non-empty string")
    return value


def health_handler() -> Dict[str, Any]:
    """Return a basic liveness payload."""

    return {"status": "ok", "service": "handwrite", "version": "0.1.0"}


def styles_handler(style_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Return the catalog of built-in handwriting styles."""

    if style_names is None:
        try:
            from handwrite.styles import list_style_names

            style_names = list_style_names()
        except Exception:  # pragma: no cover - depends on package internals
            style_names = []
    return {"styles": list(style_names)}


def _enforce_quota(
    meter: BillingMeter,
    api_key: str,
    action: str,
    chars: int,
    max_chars_per_day: int,
) -> None:
    if max_chars_per_day > 0 and not meter.check_quota(api_key, max_chars_per_day):
        raise RateLimitExceeded(
            f"daily quota of {max_chars_per_day} chars exceeded for key"
        )
    meter.record(api_key, action, chars=chars)


def inspect_handler(
    payload: Dict[str, Any],
    api_key: Optional[str],
    meter: BillingMeter,
    *,
    max_chars_per_day: int = 0,
    inspect_fn=None,
) -> Dict[str, Any]:
    """Inspect a chunk of text and report coverage."""

    key = _require_api_key(api_key)
    text = _require_text(payload)
    _enforce_quota(meter, key, "inspect", len(text), max_chars_per_day)
    if inspect_fn is None:
        from handwrite import inspect_text as inspect_fn  # type: ignore
    report = inspect_fn(text, style=payload.get("style") or "行书流畅")
    return {"report": _strip_non_serialisable(report)}


def generate_handler(
    payload: Dict[str, Any],
    api_key: Optional[str],
    meter: BillingMeter,
    *,
    max_chars_per_day: int = 0,
    generate_fn=None,
) -> Tuple[bytes, Dict[str, str]]:
    """Generate a page image and return its PNG bytes plus headers."""

    key = _require_api_key(api_key)
    text = _require_text(payload)
    _enforce_quota(meter, key, "generate", len(text), max_chars_per_day)
    if generate_fn is None:
        from handwrite import generate as generate_fn  # type: ignore
    image = generate_fn(
        text,
        style=payload.get("style") or "行书流畅",
        paper=payload.get("paper") or "white",
        layout=payload.get("layout") or "natural",
        font_size=int(payload.get("font_size", 80)),
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    headers = {
        "content-type": "image/png",
        "x-handwrite-chars": str(len(text)),
    }
    return buffer.getvalue(), headers


def note_session_handler(
    payload: Dict[str, Any],
    api_key: Optional[str],
    meter: BillingMeter,
    *,
    max_chars_per_day: int = 0,
    build_fn=None,
) -> Dict[str, Any]:
    """Return a manifest describing a generated note session."""

    key = _require_api_key(api_key)
    text = _require_text(payload)
    _enforce_quota(meter, key, "note-session", len(text), max_chars_per_day)
    if build_fn is None:
        from handwrite import build_note_session as build_fn  # type: ignore
    session = build_fn(
        text,
        style=payload.get("style") or "行书流畅",
        paper=payload.get("paper") or "white",
        layout=payload.get("layout") or "natural",
        font_size=int(payload.get("font_size", 80)),
    )
    manifest = _build_session_manifest(session)
    return manifest


def digitize_handler(
    image_bytes: bytes,
    api_key: Optional[str],
    meter: BillingMeter,
    *,
    max_chars_per_day: int = 0,
    digitize_fn=None,
) -> Dict[str, Any]:
    """Recognise a handwritten image and return the digitised payload."""

    key = _require_api_key(api_key)
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise InvalidRequest("request body must contain image bytes")
    _enforce_quota(meter, key, "digitize", 0, max_chars_per_day)
    if digitize_fn is None:
        from handwrite import digitize as digitize_fn  # type: ignore
    try:
        from PIL import Image  # type: ignore

        image = Image.open(BytesIO(bytes(image_bytes)))
        image.load()
    except Exception as exc:  # pragma: no cover - depends on Pillow
        raise InvalidRequest(f"failed to decode image: {exc}") from exc
    result = digitize_fn(image)
    return _strip_non_serialisable(result)


def _build_session_manifest(session: Dict[str, Any]) -> Dict[str, Any]:
    pages = session.get("pages") or []
    page_count = int(session.get("page_count", len(pages)))
    return {
        "text": session.get("text", ""),
        "style": session.get("style"),
        "paper": session.get("paper"),
        "layout": session.get("layout"),
        "font_size": session.get("font_size"),
        "page_count": page_count,
        "status_text": session.get("status_text"),
        "report_markdown": session.get("report_markdown"),
        "prototype_pack_name": session.get("prototype_pack_name"),
        "prototype_source_kind": session.get("prototype_source_kind"),
    }


def _strip_non_serialisable(value: Any) -> Any:
    """Drop image / numpy / bytes payloads so we can JSON-encode the result."""

    if isinstance(value, dict):
        return {
            k: _strip_non_serialisable(v)
            for k, v in value.items()
            if not _is_binary(v)
        }
    if isinstance(value, list):
        return [_strip_non_serialisable(item) for item in value if not _is_binary(item)]
    if isinstance(value, tuple):
        return [_strip_non_serialisable(item) for item in value if not _is_binary(item)]
    if _is_binary(value):
        return None
    return value


def _is_binary(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    cls = type(value)
    qualified = f"{cls.__module__}.{cls.__name__}"
    return qualified.startswith("PIL.") or qualified.startswith("numpy.ndarray")
