"""In-memory usage metering and quota enforcement for the HandWrite service.

This module is **pure Python** with no FastAPI dependency, so it can be
imported and unit-tested in environments where the web stack is missing.

The :class:`BillingMeter` records per-API-key usage events with a UTC date
stamp, supports daily quota checks, and exposes a serialisable usage report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from threading import RLock
from typing import Callable, Dict, List, Optional


__all__ = ["BillingMeter", "UsageEvent"]


@dataclass(frozen=True)
class UsageEvent:
    """A single recorded usage event."""

    api_key: str
    action: str
    chars: int
    at: datetime
    day: date


@dataclass
class _KeyState:
    """Per-API-key running totals."""

    total_calls: int = 0
    total_chars: int = 0
    by_action: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_day: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events: List[UsageEvent] = field(default_factory=list)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BillingMeter:
    """Tracks per-API-key usage with daily quota enforcement.

    The meter is intentionally simple — everything lives in process memory,
    keyed by API-key string. It is thread-safe via an :class:`RLock`, but it
    is not multi-process safe; that is acceptable for the in-memory
    demonstration service.

    Parameters
    ----------
    clock:
        Optional callable returning the current UTC :class:`datetime`. Tests
        inject a controllable clock to exercise the daily-reset logic.
    """

    def __init__(self, clock: Optional[Callable[[], datetime]] = None) -> None:
        self._clock = clock or _utc_now
        self._lock = RLock()
        self._state: Dict[str, _KeyState] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _normalise_key(api_key: str) -> str:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        cleaned = api_key.strip()
        if not cleaned:
            raise ValueError("api_key must be a non-empty string")
        return cleaned

    @staticmethod
    def _day_key(at: datetime) -> str:
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        return at.astimezone(timezone.utc).date().isoformat()

    def _get_state(self, api_key: str) -> _KeyState:
        state = self._state.get(api_key)
        if state is None:
            state = _KeyState()
            self._state[api_key] = state
        return state

    # ---------------------------------------------------------------- public
    def record(self, api_key: str, action: str, chars: int = 0) -> UsageEvent:
        """Record a single usage event and return it.

        Parameters
        ----------
        api_key:
            The non-empty API key supplied by the client.
        action:
            A short identifier for the action (e.g. ``"generate"``).
        chars:
            Optional character count consumed by this call.
        """

        key = self._normalise_key(api_key)
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        if not isinstance(chars, int):
            raise TypeError("chars must be an integer")
        if chars < 0:
            raise ValueError("chars must be non-negative")

        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        day_key = self._day_key(now)
        event = UsageEvent(
            api_key=key,
            action=action.strip(),
            chars=chars,
            at=now,
            day=now.astimezone(timezone.utc).date(),
        )
        with self._lock:
            state = self._get_state(key)
            state.total_calls += 1
            state.total_chars += chars
            state.by_action[event.action] += 1
            state.by_day[day_key] += chars
            state.events.append(event)
        return event

    def usage(self, api_key: str) -> Dict[str, object]:
        """Return a serialisable usage snapshot for ``api_key``."""

        key = self._normalise_key(api_key)
        with self._lock:
            state = self._state.get(key)
            if state is None:
                today = self._day_key(self._clock())
                return {
                    "api_key": key,
                    "total_calls": 0,
                    "total_chars": 0,
                    "today": today,
                    "today_chars": 0,
                    "by_action": {},
                    "by_day": {},
                }
            today = self._day_key(self._clock())
            return {
                "api_key": key,
                "total_calls": state.total_calls,
                "total_chars": state.total_chars,
                "today": today,
                "today_chars": int(state.by_day.get(today, 0)),
                "by_action": dict(state.by_action),
                "by_day": dict(state.by_day),
            }

    def check_quota(self, api_key: str, max_chars_per_day: int) -> bool:
        """Return ``True`` when the key may still consume more characters today.

        A ``max_chars_per_day`` of ``0`` or negative is treated as "unlimited"
        and always returns ``True``.
        """

        if not isinstance(max_chars_per_day, int):
            raise TypeError("max_chars_per_day must be an integer")
        key = self._normalise_key(api_key)
        if max_chars_per_day <= 0:
            return True
        with self._lock:
            state = self._state.get(key)
            if state is None:
                return True
            today = self._day_key(self._clock())
            today_chars = int(state.by_day.get(today, 0))
            return today_chars < max_chars_per_day

    def reset(self, api_key: Optional[str] = None) -> None:
        """Reset all state for one key, or every key when ``api_key`` is None."""

        with self._lock:
            if api_key is None:
                self._state.clear()
                return
            key = self._normalise_key(api_key)
            self._state.pop(key, None)

    def keys(self) -> List[str]:
        """Return the API keys that have at least one recorded event."""

        with self._lock:
            return sorted(self._state.keys())
