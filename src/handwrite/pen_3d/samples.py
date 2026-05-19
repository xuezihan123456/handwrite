"""Dataclasses describing 3D pen-tip samples and stroke containers.

A ``PenSample3D`` captures the full Wacom-style state of the pen at one
moment in time: planar position ``(x, y)``, timestamp ``t``, normalized
``pressure`` in ``[0, 1]``, pen tilt around the X / Y axes (radians),
barrel ``rotation`` (radians), and instantaneous ``velocity`` in pixels
per second.

A ``PenStroke3D`` bundles an ordered list of samples plus a free-form
metadata dictionary (stroke index, tool hint, author, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PenSample3D:
    """A single 3D pen-tip sample.

    Attributes:
        x: Horizontal position in canvas pixels.
        y: Vertical position in canvas pixels.
        t: Timestamp in seconds since the start of the stroke.
        pressure: Normalized pressure in ``[0.0, 1.0]``.
        tilt_x: Pen tilt around the X axis in radians (``[-pi/2, pi/2]``).
        tilt_y: Pen tilt around the Y axis in radians.
        rotation: Barrel rotation in radians (``[0, 2*pi]``).
        velocity: Instantaneous pen velocity in pixels/second.
    """

    x: float
    y: float
    t: float
    pressure: float
    tilt_x: float
    tilt_y: float
    rotation: float
    velocity: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a plain JSON-friendly dictionary."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "t": float(self.t),
            "pressure": float(self.pressure),
            "tilt_x": float(self.tilt_x),
            "tilt_y": float(self.tilt_y),
            "rotation": float(self.rotation),
            "velocity": float(self.velocity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PenSample3D":
        """Construct from a JSON-decoded dictionary."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            t=float(data["t"]),
            pressure=float(data["pressure"]),
            tilt_x=float(data["tilt_x"]),
            tilt_y=float(data["tilt_y"]),
            rotation=float(data["rotation"]),
            velocity=float(data["velocity"]),
        )


@dataclass(frozen=True)
class PenStroke3D:
    """Ordered collection of ``PenSample3D`` plus stroke-level metadata.

    The dataclass is frozen for immutability per project conventions; the
    sample list itself is a plain Python ``list`` so it can be inspected
    cheaply, but consumers must treat it as read-only.
    """

    samples: list[PenSample3D] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Total stroke duration in seconds (``t_last - t_first``)."""
        if len(self.samples) < 2:
            return 0.0
        return float(self.samples[-1].t - self.samples[0].t)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stroke to a plain JSON-friendly dictionary."""
        return {
            "samples": [s.to_dict() for s in self.samples],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PenStroke3D":
        """Construct from a JSON-decoded dictionary."""
        samples = [PenSample3D.from_dict(s) for s in data.get("samples", [])]
        metadata = dict(data.get("metadata", {}))
        return cls(samples=samples, metadata=metadata)


__all__ = ["PenSample3D", "PenStroke3D"]
