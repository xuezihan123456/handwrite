"""Style-space interpolation utilities.

Provides several interpolation methods for blending StyleVectors:
  - Linear interpolation (lerp)
  - Spherical linear interpolation (slerp)
  - Cubic Bezier interpolation
  - Multi-style weighted blend
"""

from __future__ import annotations

import math
from dataclasses import fields

from .style_vector import StyleVector


def _lerp_scalar(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _slerp_scalar(a: float, b: float, t: float) -> float:
    """Component-wise slerp approximation.

    For scalar values, true slerp on a unit sphere is not applicable; we use a
    smooth-step variant that eases in/out more naturally than linear.
    """
    # Smooth-step interpolation: 3t^2 - 2t^3
    smooth_t = t * t * (3 - 2 * t)
    return a + (b - a) * smooth_t


def _bezier_scalar(a: float, b: float, t: float, c1: float, c2: float) -> float:
    """Cubic Bezier interpolation for a single scalar."""
    u = 1 - t
    return u * u * u * a + 3 * u * u * t * c1 + 3 * u * t * t * c2 + t * t * t * b


def lerp(a: StyleVector, b: StyleVector, t: float) -> StyleVector:
    """Linear interpolation between two style vectors.

    Args:
        a: Start style (t=0).
        b: End style (t=1).
        t: Interpolation parameter in [0, 1].

    Returns:
        A new StyleVector with values linearly interpolated and clamped.
    """
    return StyleVector.clamped(
        **{
            fld.name: _lerp_scalar(getattr(a, fld.name), getattr(b, fld.name), t)
            for fld in fields(StyleVector)
        }
    )


def slerp(a: StyleVector, b: StyleVector, t: float) -> StyleVector:
    """Smooth-step interpolation (eased lerp).

    Uses 3t^2 - 2t^3 for a smoother transition than pure linear.
    """
    return StyleVector.clamped(
        **{
            fld.name: _slerp_scalar(getattr(a, fld.name), getattr(b, fld.name), t)
            for fld in fields(StyleVector)
        }
    )


def bezier(
    a: StyleVector,
    b: StyleVector,
    t: float,
    control1: StyleVector | None = None,
    control2: StyleVector | None = None,
) -> StyleVector:
    """Cubic Bezier interpolation with optional control points.

    If control points are None, they default to a and b respectively
    (which degenerates to a cubic ease-in-out curve).
    """
    if control1 is None:
        control1 = a
    if control2 is None:
        control2 = b

    return StyleVector.clamped(
        **{
            fld.name: _bezier_scalar(
                getattr(a, fld.name),
                getattr(b, fld.name),
                t,
                getattr(control1, fld.name),
                getattr(control2, fld.name),
            )
            for fld in fields(StyleVector)
        }
    )


def weighted_blend(styles: list[StyleVector], weights: list[float]) -> StyleVector:
    """Blend multiple style vectors using explicit weights.

    Args:
        styles: List of StyleVectors to blend.
        weights: Per-style weights (need not sum to 1; will be normalised).

    Returns:
        A single blended StyleVector.

    Raises:
        ValueError: If lengths mismatch or all weights are zero.
    """
    if len(styles) != len(weights):
        raise ValueError(
            f"styles ({len(styles)}) and weights ({len(weights)}) must have the same length"
        )
    if not styles:
        raise ValueError("At least one style is required")

    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights must be non-zero")

    norm_weights = [w / total for w in weights]

    blended: dict[str, float] = {}
    for fld in fields(StyleVector):
        blended[fld.name] = sum(
            getattr(s, fld.name) * w for s, w in zip(styles, norm_weights)
        )

    return StyleVector.clamped(**blended)


__all__ = [
    "lerp",
    "slerp",
    "bezier",
    "weighted_blend",
]
