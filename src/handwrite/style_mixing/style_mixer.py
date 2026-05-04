"""High-level style mixer.

Mixes two StyleVectors with a configurable ratio, optionally applying
dimension-specific mixing strategies (e.g. angular interpolation for slant).
"""

from __future__ import annotations

import math
from dataclasses import fields

from .style_vector import StyleVector
from . import interpolation_engine as interp


def _lerp_angle(a_deg: float, b_deg: float, t: float) -> float:
    """Linearly interpolate between two angles in degrees."""
    return a_deg + (b_deg - a_deg) * t


def mix_styles(
    style_a: StyleVector,
    style_b: StyleVector,
    ratio: float = 0.5,
    *,
    method: str = "linear",
) -> StyleVector:
    """Mix two handwriting styles.

    Args:
        style_a: The first (base) style.
        style_b: The second style.
        ratio: Blend weight for style_b in [0, 1].  0 = pure style_a, 1 = pure style_b.
        method: Interpolation method -- "linear", "smooth", or "bezier".

    Returns:
        A new mixed StyleVector.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")

    if method == "linear":
        return interp.lerp(style_a, style_b, ratio)
    if method == "smooth":
        return interp.slerp(style_a, style_b, ratio)
    if method == "bezier":
        return interp.bezier(style_a, style_b, ratio)
    raise ValueError(f"Unknown mixing method: {method!r}")


def mix_multi(
    styles: list[StyleVector],
    weights: list[float],
    *,
    method: str = "linear",
) -> StyleVector:
    """Mix multiple styles with explicit weights.

    Args:
        styles: Style vectors to blend.
        weights: Per-style weights (will be normalised).
        method: Interpolation method used for pairwise blending.

    Returns:
        A single blended StyleVector.
    """
    if len(styles) != len(weights):
        raise ValueError("styles and weights must have the same length")
    if not styles:
        raise ValueError("At least one style is required")
    if len(styles) == 1:
        return styles[0]
    return interp.weighted_blend(styles, weights)


def describe_mixture(
    style_a: StyleVector,
    style_b: StyleVector,
    ratio: float,
) -> dict[str, float]:
    """Return a human-readable breakdown of the mixture dimensions."""
    mixed = mix_styles(style_a, style_b, ratio)
    return {
        "ratio_b": round(ratio, 3),
        **{k: round(v, 4) for k, v in mixed.to_dict().items()},
    }


__all__ = [
    "mix_styles",
    "mix_multi",
    "describe_mixture",
]
