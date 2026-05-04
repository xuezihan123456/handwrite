"""Stroke dynamics simulation module.

Simulates handwriting physical properties: pressure-based stroke width
variation, ink density gradients, and speed-dependent lightness.
"""

from .dynamics_engine import DynamicsParams, apply_dynamics

__all__ = [
    "apply_dynamics",
    "DynamicsParams",
]
