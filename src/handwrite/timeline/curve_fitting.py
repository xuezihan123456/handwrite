"""Per-feature curve fitting for the handwriting timeline.

Given a sequence of (age, value) observations, :func:`fit_curve` returns a
small :class:`Curve` object that can predict the value at any age. Three
methods are supported, all implemented on top of ``numpy``:

* ``"linear"`` - degree 1 polynomial via :func:`numpy.polyfit`.
* ``"polynomial"`` - degree 2 polynomial via :func:`numpy.polyfit`.
* ``"smoothing"`` - piecewise-linear interpolation with clamped extrapolation.

The :class:`Curve` is JSON-serializable so that fitted models can be saved
and reloaded without pickling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

VALID_METHODS = ("linear", "polynomial", "smoothing")


@dataclass
class Curve:
    """A fitted age -> feature curve.

    Attributes:
        method: One of ``VALID_METHODS``.
        coefficients: Polynomial coefficients (highest degree first) for
            ``"linear"`` and ``"polynomial"`` methods. Empty for
            ``"smoothing"``.
        knots_age: Anchor ages used by the ``"smoothing"`` interpolator.
        knots_value: Anchor values aligned with :attr:`knots_age`.
        clamp: Optional ``(low, high)`` value clamp applied to predictions.
    """

    method: str
    coefficients: list[float] = field(default_factory=list)
    knots_age: list[float] = field(default_factory=list)
    knots_value: list[float] = field(default_factory=list)
    clamp: tuple[float, float] | None = None

    def predict(self, age: float) -> float:
        """Predict the feature value at a given age.

        Args:
            age: Target age (in years). May lie outside the fitting range -
                the predictor will extrapolate using the fitted method.

        Returns:
            The predicted feature value, optionally clamped.
        """
        age = float(age)
        if self.method in ("linear", "polynomial"):
            value = float(np.polyval(np.asarray(self.coefficients, dtype=float), age))
        elif self.method == "smoothing":
            value = _piecewise_linear(age, self.knots_age, self.knots_value)
        else:
            raise ValueError(f"Unknown curve method: {self.method!r}")

        if self.clamp is not None:
            low, high = self.clamp
            value = max(low, min(high, value))
        return float(value)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dict representation."""
        return {
            "method": self.method,
            "coefficients": list(self.coefficients),
            "knots_age": list(self.knots_age),
            "knots_value": list(self.knots_value),
            "clamp": list(self.clamp) if self.clamp is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Curve":
        """Reconstruct a :class:`Curve` from :meth:`to_dict` output."""
        clamp = data.get("clamp")
        return cls(
            method=str(data["method"]),
            coefficients=[float(c) for c in data.get("coefficients", [])],
            knots_age=[float(a) for a in data.get("knots_age", [])],
            knots_value=[float(v) for v in data.get("knots_value", [])],
            clamp=(float(clamp[0]), float(clamp[1])) if clamp else None,
        )


def fit_curve(
    ages: Sequence[float],
    values: Sequence[float],
    method: str = "polynomial",
    clamp: tuple[float, float] | None = None,
) -> Curve:
    """Fit a per-feature curve over (age, value) observations.

    Args:
        ages: Observation ages, length >= 2.
        values: Observed feature values aligned with :paramref:`ages`.
        method: Fitting strategy - ``"linear"``, ``"polynomial"`` (default,
            degree 2) or ``"smoothing"`` (piecewise-linear interpolation).
        clamp: Optional ``(low, high)`` clamp applied during prediction.

    Returns:
        A fitted :class:`Curve` ready for :meth:`Curve.predict`.

    Raises:
        ValueError: If inputs are inconsistent or the method is unknown.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown curve method {method!r}. Valid: {VALID_METHODS}"
        )

    ages_arr = np.asarray(ages, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if ages_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("ages and values must be 1-D sequences")
    if ages_arr.shape != values_arr.shape:
        raise ValueError("ages and values must have the same length")
    if ages_arr.size < 2:
        raise ValueError("At least two observations are required to fit a curve")

    # Sort by age to keep downstream operations well-defined.
    order = np.argsort(ages_arr)
    ages_arr = ages_arr[order]
    values_arr = values_arr[order]

    if method == "linear":
        coeffs = np.polyfit(ages_arr, values_arr, deg=1)
        return Curve(
            method="linear",
            coefficients=[float(c) for c in coeffs],
            clamp=clamp,
        )

    if method == "polynomial":
        # Polynomial of degree 2 is the default - keeps extrapolation tame
        # while still fitting non-linear age progressions like stroke width.
        degree = 2 if ages_arr.size >= 3 else 1
        coeffs = np.polyfit(ages_arr, values_arr, deg=degree)
        return Curve(
            method="polynomial",
            coefficients=[float(c) for c in coeffs],
            clamp=clamp,
        )

    # "smoothing"
    return Curve(
        method="smoothing",
        knots_age=[float(a) for a in ages_arr.tolist()],
        knots_value=[float(v) for v in values_arr.tolist()],
        clamp=clamp,
    )


def _piecewise_linear(
    age: float,
    knots_age: Sequence[float],
    knots_value: Sequence[float],
) -> float:
    """Evaluate a piecewise-linear interpolator with edge extrapolation."""
    if not knots_age:
        return 0.0
    if len(knots_age) == 1:
        return float(knots_value[0])

    ages = list(knots_age)
    values = list(knots_value)

    if age <= ages[0]:
        # Extrapolate using the first segment slope.
        slope = (values[1] - values[0]) / (ages[1] - ages[0]) if ages[1] != ages[0] else 0.0
        return float(values[0] + slope * (age - ages[0]))
    if age >= ages[-1]:
        slope = (
            (values[-1] - values[-2]) / (ages[-1] - ages[-2])
            if ages[-1] != ages[-2]
            else 0.0
        )
        return float(values[-1] + slope * (age - ages[-1]))

    for idx in range(len(ages) - 1):
        a0, a1 = ages[idx], ages[idx + 1]
        if a0 <= age <= a1:
            v0, v1 = values[idx], values[idx + 1]
            if a1 == a0:
                return float(v0)
            t = (age - a0) / (a1 - a0)
            return float(v0 + t * (v1 - v0))

    return float(values[-1])


__all__ = ["Curve", "fit_curve", "VALID_METHODS"]
