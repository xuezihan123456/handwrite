"""StyleVector dataclass for natural-language style mapping.

This is a *distinct* StyleVector from ``handwrite.style_mixing.StyleVector``.
It carries the concrete numeric parameters that a downstream composer or
rendering engine can consume directly (rotation jitter, scale jitter, ink
density, baseline jitter, character spacing, line spacing) plus a
human-readable ``style_name``, a ``suggested_layout`` value matching one of
the composer layout constants (NEAT/NATURAL/CURSIVE), and a list of mood
tags surfaced from the parsed description.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar


# Layout constants mirror those in ``handwrite.composer`` so we do not need
# to import the composer module (and pull in PIL) just to produce a vector.
# They are kept in sync with ``composer.NEAT_LAYOUT`` / ``NATURAL_LAYOUT`` /
# ``CURSIVE_LAYOUT``.
NEAT_LAYOUT = "\u5de5\u6574"      # 工整
NATURAL_LAYOUT = "\u81ea\u7136"   # 自然
CURSIVE_LAYOUT = "\u6f47\u8349"   # 潇草

_VALID_LAYOUTS = (NEAT_LAYOUT, NATURAL_LAYOUT, CURSIVE_LAYOUT)


@dataclass
class StyleVector:
    """Concrete style parameters produced by the natural-language parser.

    All numeric fields are clamped to their valid ranges on construction.
    The dataclass is intentionally *mutable* so that the parser can build it
    up incrementally; callers that want immutability can simply not mutate
    it after parsing.
    """

    rotation_jitter: float = 0.0      # degrees of random per-char rotation
    scale_jitter: float = 0.0         # 0..1 fraction of size jitter
    ink_density: float = 1.0          # ink darkness multiplier 0.5..1.5
    baseline_jitter: float = 0.0      # 0..1 baseline wobble fraction
    char_spacing: float = 1.0         # multiplier on char gap 0.5..2.0
    line_spacing: float = 1.0         # multiplier on line gap 0.5..2.0
    style_name: str = "default"
    suggested_layout: str = NATURAL_LAYOUT
    mood_tags: list[str] = field(default_factory=list)

    _RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "rotation_jitter": (0.0, 15.0),
        "scale_jitter": (0.0, 1.0),
        "ink_density": (0.5, 1.5),
        "baseline_jitter": (0.0, 1.0),
        "char_spacing": (0.5, 2.0),
        "line_spacing": (0.5, 2.0),
    }

    def __post_init__(self) -> None:
        for fld in fields(self):
            if fld.name not in self._RANGES:
                continue
            value = float(getattr(self, fld.name))
            lo, hi = self._RANGES[fld.name]
            clamped = max(lo, min(hi, value))
            object.__setattr__(self, fld.name, clamped)

        if self.suggested_layout not in _VALID_LAYOUTS:
            self.suggested_layout = NATURAL_LAYOUT

        # mood_tags should always be a list of unique strings preserving order
        if self.mood_tags is None:
            self.mood_tags = []
        else:
            seen: set[str] = set()
            unique: list[str] = []
            for tag in self.mood_tags:
                if tag and tag not in seen:
                    seen.add(tag)
                    unique.append(tag)
            self.mood_tags = unique

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the full numeric + metadata payload as a plain dict."""
        return {
            "rotation_jitter": self.rotation_jitter,
            "scale_jitter": self.scale_jitter,
            "ink_density": self.ink_density,
            "baseline_jitter": self.baseline_jitter,
            "char_spacing": self.char_spacing,
            "line_spacing": self.line_spacing,
            "style_name": self.style_name,
            "suggested_layout": self.suggested_layout,
            "mood_tags": list(self.mood_tags),
        }

    # ------------------------------------------------------------------
    # Composer integration
    # ------------------------------------------------------------------

    def to_composer_kwargs(self) -> dict[str, Any]:
        """Return kwargs that map cleanly onto ``compose_page``.

        Only keys that ``compose_page`` accepts are included; the remaining
        numeric fields (jitter, density, spacing multipliers) are surfaced
        through ``style_params`` for downstream renderers that understand
        them.
        """
        return {
            "layout": self.suggested_layout,
            "style_params": {
                "rotation_jitter": self.rotation_jitter,
                "scale_jitter": self.scale_jitter,
                "ink_density": self.ink_density,
                "baseline_jitter": self.baseline_jitter,
                "char_spacing": self.char_spacing,
                "line_spacing": self.line_spacing,
                "style_name": self.style_name,
                "mood_tags": list(self.mood_tags),
            },
        }

    def apply_to_layout(self, base_layout: str | None = None) -> str:
        """Choose a concrete layout name for the composer.

        If ``base_layout`` is supplied and valid it takes precedence (lets
        callers override the parser's guess), otherwise the parser's
        ``suggested_layout`` is returned.
        """
        if base_layout in _VALID_LAYOUTS:
            return base_layout
        return self.suggested_layout


__all__ = [
    "StyleVector",
    "NEAT_LAYOUT",
    "NATURAL_LAYOUT",
    "CURSIVE_LAYOUT",
]
