"""3D pen-tip dynamics simulator.

Given a 2D trajectory (list of ``(x, y)`` points), the simulator produces
a ``PenStroke3D`` that augments every sample with timing, pressure, tilt
(X/Y), barrel rotation and instantaneous velocity. Pressure is modulated
inversely with velocity (faster strokes press lighter), and tilt evolves
along smooth low-frequency curves with a small amount of paper-surface
friction jitter.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import math

import numpy as np

from .samples import PenSample3D, PenStroke3D


# Sentinel constants for tilt / rotation bounds (radians).
_MAX_TILT = math.pi / 3.0  # ~60 degrees from vertical
_TWO_PI = 2.0 * math.pi


class Pen3DSimulator:
    """Convert a 2D point sequence into a rich 3D pen stroke.

    Args:
        sample_rate: Sampling rate in Hz used to assign timestamps.
        base_pressure: Baseline pressure when the pen is moving at
            ``reference_velocity`` (in [0, 1]).
        reference_velocity: Velocity (pixels/second) at which the
            baseline pressure applies.
        pressure_velocity_gain: How strongly pressure drops as velocity
            grows above the reference velocity (positive, larger = more
            sensitive).
        tilt_amplitude: Maximum tilt magnitude in radians used for the
            low-frequency tilt curves (clamped by ``_MAX_TILT``).
        rotation_drift: Per-second rotation drift (radians/second) added
            on top of a base rotation. Models barrel slowly turning while
            the user holds the pen.
        friction_jitter: Standard deviation of the tiny random ``z``
            displacement that models paper-surface friction. Although
            ``z`` is not stored explicitly (the pen lives on paper), the
            jitter is folded into pressure as a small additive noise so
            the effect is observable.
        seed: Optional RNG seed for deterministic output.
    """

    def __init__(
        self,
        *,
        sample_rate: float = 120.0,
        base_pressure: float = 0.6,
        reference_velocity: float = 200.0,
        pressure_velocity_gain: float = 0.6,
        tilt_amplitude: float = math.pi / 6.0,
        rotation_drift: float = 0.4,
        friction_jitter: float = 0.02,
        seed: int | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= base_pressure <= 1.0:
            raise ValueError("base_pressure must be in [0, 1]")
        if reference_velocity <= 0:
            raise ValueError("reference_velocity must be positive")
        if pressure_velocity_gain < 0:
            raise ValueError("pressure_velocity_gain must be non-negative")
        if friction_jitter < 0:
            raise ValueError("friction_jitter must be non-negative")

        self.sample_rate = float(sample_rate)
        self.base_pressure = float(base_pressure)
        self.reference_velocity = float(reference_velocity)
        self.pressure_velocity_gain = float(pressure_velocity_gain)
        self.tilt_amplitude = float(min(tilt_amplitude, _MAX_TILT))
        self.rotation_drift = float(rotation_drift)
        self.friction_jitter = float(friction_jitter)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate(
        self,
        stroke_2d: Sequence[tuple[float, float]],
        *,
        metadata: dict | None = None,
    ) -> PenStroke3D:
        """Produce a ``PenStroke3D`` from a sequence of ``(x, y)`` points.

        Args:
            stroke_2d: Ordered iterable of 2D points (``(x, y)``).
            metadata: Optional metadata dictionary attached to the stroke.

        Returns:
            ``PenStroke3D`` with one sample per input point.

        Raises:
            ValueError: If ``stroke_2d`` is empty.
        """
        points = [(float(x), float(y)) for x, y in stroke_2d]
        if not points:
            raise ValueError("stroke_2d must contain at least one point")

        n = len(points)
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / self.sample_rate
        timestamps = np.arange(n, dtype=np.float64) * dt

        # Velocity (pixels/second): finite differences with forward/backward
        # at the endpoints and central differences elsewhere.
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        velocities = self._compute_velocity(xs, ys, dt)

        # Pressure: inversely related to velocity, clipped to [0, 1].
        pressure_jitter = rng.normal(0.0, self.friction_jitter, size=n)
        velocity_ratio = velocities / self.reference_velocity
        pressures = self.base_pressure / (
            1.0 + self.pressure_velocity_gain * np.maximum(velocity_ratio - 1.0, 0.0)
        )
        pressures = np.clip(pressures + pressure_jitter, 0.0, 1.0)

        # Tilt curves: low-frequency sinusoids along arc-length, plus
        # small Gaussian noise for natural variability.
        s_norm = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
        tilt_x = self.tilt_amplitude * np.sin(2.0 * math.pi * s_norm)
        tilt_y = self.tilt_amplitude * np.cos(math.pi * s_norm)
        tilt_x += rng.normal(0.0, self.tilt_amplitude * 0.05, size=n)
        tilt_y += rng.normal(0.0, self.tilt_amplitude * 0.05, size=n)
        tilt_x = np.clip(tilt_x, -_MAX_TILT, _MAX_TILT)
        tilt_y = np.clip(tilt_y, -_MAX_TILT, _MAX_TILT)

        # Rotation: monotonic drift wrapped into [0, 2*pi).
        rotation_base = float(rng.uniform(0.0, _TWO_PI))
        rotations = (rotation_base + self.rotation_drift * timestamps) % _TWO_PI

        samples: list[PenSample3D] = []
        for i, (x, y) in enumerate(points):
            samples.append(
                PenSample3D(
                    x=float(x),
                    y=float(y),
                    t=float(timestamps[i]),
                    pressure=float(pressures[i]),
                    tilt_x=float(tilt_x[i]),
                    tilt_y=float(tilt_y[i]),
                    rotation=float(rotations[i]),
                    velocity=float(velocities[i]),
                )
            )

        meta: dict = dict(metadata or {})
        meta.setdefault("sample_rate", self.sample_rate)
        meta.setdefault("source", "Pen3DSimulator")
        return PenStroke3D(samples=samples, metadata=meta)

    def simulate_batch(
        self, strokes_2d: Iterable[Sequence[tuple[float, float]]]
    ) -> list[PenStroke3D]:
        """Convenience helper: simulate every stroke in an iterable."""
        return [self.simulate(s) for s in strokes_2d]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_velocity(
        xs: np.ndarray, ys: np.ndarray, dt: float
    ) -> np.ndarray:
        """Compute per-sample velocity magnitude using gradient differences."""
        n = xs.shape[0]
        if n == 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np.zeros(1, dtype=np.float64)

        dx = np.gradient(xs)
        dy = np.gradient(ys)
        speeds = np.sqrt(dx * dx + dy * dy) / max(dt, 1e-9)
        return speeds


__all__ = ["Pen3DSimulator"]
