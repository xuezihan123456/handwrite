"""Integrate all dynamics simulators into a single pipeline.

The ``apply_dynamics`` function is the main entry point: it takes a character
image and dynamics parameters, then applies pressure, ink, and speed effects
in sequence to produce a realistic handwriting image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from .ink_simulator import simulate_ink
from .pressure_simulator import simulate_pressure
from .speed_simulator import simulate_speed

PressureCurve = Literal["natural", "heavy_start", "uniform"]


@dataclass(frozen=True)
class DynamicsParams:
    """Parameters controlling the dynamics simulation.

    Attributes:
        enabled: Master switch to enable/disable all dynamics.
        pressure_strength: How strongly pressure affects stroke width (0.0-1.0).
        pressure_curve: Type of pressure curve.
        ink_density: Base ink density (0.0=light, 1.0=dark).
        ink_diffusion: Ink diffusion radius in pixels.
        speed_variation: How much speed varies across strokes (0.0-1.0).
        base_speed: Base writing speed (0.0=slow, 1.0=fast).
        output_size: Output image size (width, height).
    """

    enabled: bool = True
    pressure_strength: float = 0.5
    pressure_curve: PressureCurve = "natural"
    ink_density: float = 0.6
    ink_diffusion: int = 2
    speed_variation: float = 0.3
    base_speed: float = 0.5
    output_size: tuple[int, int] = (256, 256)


def apply_dynamics(
    char_image: Image.Image,
    params: DynamicsParams | None = None,
) -> Image.Image:
    """Apply handwriting dynamics simulation to a character image.

    Processes the image through three stages:
    1. Pressure simulation: varies stroke width based on simulated pen pressure.
    2. Ink simulation: adds ink density gradients and diffusion.
    3. Speed simulation: fast strokes become lighter and slightly blurred.

    Args:
        char_image: Input character image (grayscale or RGB).
        params: Dynamics parameters. Uses defaults if None.

    Returns:
        Processed PIL Image in grayscale mode ('L'), sized to ``output_size``.
    """
    if params is None:
        params = DynamicsParams()

    if not params.enabled:
        return char_image.convert("L").resize(
            params.output_size, Image.Resampling.LANCZOS
        )

    # Convert to grayscale numpy array
    gray = char_image.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # Resize if needed before processing
    if arr.shape != (params.output_size[1], params.output_size[0]):
        gray = gray.resize(params.output_size, Image.Resampling.LANCZOS)
        arr = np.array(gray, dtype=np.uint8)

    # Stage 1: Pressure simulation
    arr = simulate_pressure(
        arr,
        pressure_strength=params.pressure_strength,
        pressure_curve=params.pressure_curve,
    )

    # Stage 2: Ink simulation
    arr = simulate_ink(
        arr,
        ink_density=params.ink_density,
        diffusion_radius=params.ink_diffusion,
        pressure_curve=params.pressure_curve,
    )

    # Stage 3: Speed simulation
    arr = simulate_speed(
        arr,
        speed_variation=params.speed_variation,
        base_speed=params.base_speed,
    )

    return Image.fromarray(arr, mode="L")
