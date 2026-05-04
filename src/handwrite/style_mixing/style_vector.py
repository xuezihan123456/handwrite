"""Style vector definition for handwriting style representation.

A style vector encodes five independent dimensions of handwriting appearance:
  - neatness (0-1): how tidy and regular the writing is
  - connectivity (0-1): how much characters are joined together
  - slant_angle (-15 to 15): pen tilt in degrees (negative = left, positive = right)
  - stroke_width (0.8-1.2): relative pen thickness multiplier
  - ink_density (0.7-1.3): ink darkness / saturation multiplier
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar


@dataclass(frozen=True)
class StyleVector:
    """Five-dimensional handwriting style descriptor.

    All values are clamped to their valid ranges on construction.
    """

    neatness: float = 0.7       # 0 = very messy, 1 = very neat
    connectivity: float = 0.3   # 0 = printed, 1 = fully cursive
    slant_angle: float = 0.0    # degrees: -15 .. +15
    stroke_width: float = 1.0   # multiplier: 0.8 .. 1.2
    ink_density: float = 1.0    # multiplier: 0.7 .. 1.3

    # -- Validation ranges ---------------------------------------------------

    _RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "neatness": (0.0, 1.0),
        "connectivity": (0.0, 1.0),
        "slant_angle": (-15.0, 15.0),
        "stroke_width": (0.8, 1.2),
        "ink_density": (0.7, 1.3),
    }

    def __post_init__(self) -> None:
        for fld in fields(self):
            value = getattr(self, fld.name)
            lo, hi = self._RANGES[fld.name]
            if not (lo <= value <= hi):
                raise ValueError(
                    f"{fld.name} must be in [{lo}, {hi}], got {value}"
                )

    # -- Convenience ---------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Return an ordered dictionary of all style dimensions."""
        return {fld.name: getattr(self, fld.name) for fld in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> StyleVector:
        """Create a StyleVector from a dictionary (ignores unknown keys)."""
        known = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def clamped(cls, **kwargs: float) -> StyleVector:
        """Create a StyleVector, clamping out-of-range values instead of raising."""
        clamped_kwargs: dict[str, float] = {}
        for fld in fields(cls):
            value = kwargs.get(fld.name, fld.default)
            lo, hi = cls._RANGES[fld.name]
            clamped_kwargs[fld.name] = max(lo, min(hi, float(value)))
        return cls(**clamped_kwargs)

    # -- Built-in presets ----------------------------------------------------

    @classmethod
    def neat(cls) -> StyleVector:
        """Preset: neat, print-like, upright writing."""
        return cls(neatness=0.9, connectivity=0.1, slant_angle=2.0,
                   stroke_width=1.0, ink_density=1.0)

    @classmethod
    def cursive(cls) -> StyleVector:
        """Preset: flowing, connected, slightly right-leaning writing."""
        return cls(neatness=0.5, connectivity=0.85, slant_angle=8.0,
                   stroke_width=1.05, ink_density=1.1)

    @classmethod
    def messy(cls) -> StyleVector:
        """Preset: messy, fast, light-pressure writing."""
        return cls(neatness=0.15, connectivity=0.6, slant_angle=-5.0,
                   stroke_width=0.9, ink_density=0.8)

    @classmethod
    def default(cls) -> StyleVector:
        """Preset: balanced, moderate style."""
        return cls()


# -- Distance / similarity utilities -----------------------------------------

def euclidean_distance(a: StyleVector, b: StyleVector) -> float:
    """Euclidean distance in *normalised* style space.

    Each dimension is normalised to [0, 1] before computing distance so that
    all five dimensions contribute roughly equally.
    """
    total = 0.0
    for fld in fields(StyleVector):
        va = getattr(a, fld.name)
        vb = getattr(b, fld.name)
        lo, hi = StyleVector._RANGES[fld.name]
        span = hi - lo if hi != lo else 1.0
        total += ((va - vb) / span) ** 2
    return math.sqrt(total)


def cosine_similarity(a: StyleVector, b: StyleVector) -> float:
    """Cosine similarity between two style vectors (treated as raw vectors)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for fld in fields(StyleVector):
        va = getattr(a, fld.name)
        vb = getattr(b, fld.name)
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


__all__ = [
    "StyleVector",
    "euclidean_distance",
    "cosine_similarity",
]
