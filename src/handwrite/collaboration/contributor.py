"""Contributor definition for collaborative handwriting documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Contributor:
    """A single contributor who writes part of a collaborative document.

    Attributes:
        name: Display name of the contributor (e.g. "Alice").
        style_id: Numeric handwriting style id mapped to a built-in style.
            See ``handwrite.styles.BUILTIN_STYLES`` for valid values (0-4).
        params: Optional per-contributor style parameters that may fine-tune
            rendering (e.g. ``{"slant": 5, "pressure": 0.8}``).  These are
            passed through to the underlying engine when available.
    """

    name: str
    style_id: int
    params: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Contributor name must be a non-empty string")
        if not isinstance(self.style_id, int):
            raise TypeError(f"style_id must be int, got {type(self.style_id).__name__}")
