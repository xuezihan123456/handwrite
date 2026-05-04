"""Historical writing instrument style simulation.

Applies visual effects that mimic different writing instruments:
- Brush pen (毛笔): thick/thin variation based on stroke direction
- Fountain pen (钢笔): ink density variation, line variation
- Ballpoint pen (圆珠笔): consistent lines, slight ink skipping
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class HistoricalInstrument(Enum):
    """Historical writing instrument types."""

    BRUSH_PEN = "brush_pen"  # 毛笔
    FOUNTAIN_PEN = "fountain_pen"  # 钢笔
    BALLPOINT_PEN = "ballpoint_pen"  # 圆珠笔
    REED_PEN = "reed_pen"  # 芦苇笔 (ancient)


def apply_historical_style(
    image: Image.Image,
    instrument: HistoricalInstrument,
    ink_color: tuple[int, int, int] = (20, 20, 60),
) -> Image.Image:
    """Apply a historical writing instrument effect to a character image.

    Args:
        image: Source character image (grayscale or RGBA).
        instrument: The writing instrument to simulate.
        ink_color: RGB color for the ink (default: dark blue-black).

    Returns:
        A new RGBA Image with the instrument effect applied.
    """
    if instrument == HistoricalInstrument.BRUSH_PEN:
        return _apply_brush_pen(image, ink_color)
    elif instrument == HistoricalInstrument.FOUNTAIN_PEN:
        return _apply_fountain_pen(image, ink_color)
    elif instrument == HistoricalInstrument.BALLPOINT_PEN:
        return _apply_ballpoint_pen(image, ink_color)
    elif instrument == HistoricalInstrument.REED_PEN:
        return _apply_reed_pen(image, ink_color)
    else:
        raise ValueError(f"Unknown instrument: {instrument!r}")


def _apply_brush_pen(
    image: Image.Image,
    ink_color: tuple[int, int, int],
) -> Image.Image:
    """Simulate brush pen: thick/thin variation based on stroke direction.

    Brush pens produce thicker strokes on horizontal movements and thinner
    strokes on vertical movements. The effect is achieved by directional
    filtering of the ink mask.
    """
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    ink_mask = arr < 200

    if not np.any(ink_mask):
        return _create_ink_image(image.size, ink_color, np.zeros_like(arr))

    # Create directional thickness maps
    # Horizontal strokes (thick): detect horizontal edges
    h_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    v_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)

    h_response = _convolve2d(ink_mask.astype(np.float32), h_kernel)
    v_response = _convolve2d(ink_mask.astype(np.float32), v_kernel)

    # Thickness map: horizontal strokes get 1.4x thickness, vertical get 0.7x
    base_thickness = ink_mask.astype(np.float32)
    h_thickness = np.abs(h_response) > 0.5
    v_thickness = np.abs(v_response) > 0.5

    thickness_map = base_thickness.copy()
    thickness_map[h_thickness] *= 1.4
    thickness_map[v_thickness] *= 0.7

    # Apply dilation for thick strokes
    thick_ink = thickness_map > 0.3

    # Create output with ink color
    result = _create_ink_image(image.size, ink_color, thick_ink.astype(np.float32))

    # Add ink bleeding effect (slight blur for wet brush)
    result = result.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Add pressure-dependent opacity variation
    result = _add_pressure_variation(result, thick_ink)

    return result


def _apply_fountain_pen(
    image: Image.Image,
    ink_color: tuple[int, int, int],
) -> Image.Image:
    """Simulate fountain pen: ink density variation and slight line variation.

    Fountain pens have moderate line variation and can show ink pooling
    at stroke starts and ends.
    """
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    ink_mask = arr < 200

    if not np.any(ink_mask):
        return _create_ink_image(image.size, ink_color, np.zeros_like(arr))

    # Create ink density variation
    density = np.ones_like(arr, dtype=np.float32)
    ink_float = ink_mask.astype(np.float32)

    # Detect stroke endpoints (corners and edges)
    # Pool ink at stroke starts/ends by increasing density there

    # Erode to find stroke interior, endpoints are where erosion removes pixels
    eroded = _simple_erosion(ink_float)
    endpoints = ink_float - eroded
    endpoint_mask = endpoints > 0.5

    # Increase density at endpoints (ink pooling)
    density[endpoint_mask] = 1.3

    # Add slight ink skipping for older pen effect
    skip_mask = _generate_ink_skip(ink_float, skip_probability=0.02)
    ink_float[skip_mask] = 0.0

    # Apply density to create opacity variation
    alpha = ink_float * density
    alpha = np.clip(alpha, 0, 1)

    result = _create_ink_image(image.size, ink_color, alpha)

    # Slight line width variation
    result = result.filter(ImageFilter.GaussianBlur(radius=0.4))

    return result


def _apply_ballpoint_pen(
    image: Image.Image,
    ink_color: tuple[int, int, int],
) -> Image.Image:
    """Simulate ballpoint pen: consistent lines with ink skipping.

    Ballpoint pens produce relatively uniform lines but can skip
    at stroke starts, producing a dotted or broken line effect.
    """
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    ink_mask = arr < 200

    if not np.any(ink_mask):
        return _create_ink_image(image.size, ink_color, np.zeros_like(arr))

    ink_float = ink_mask.astype(np.float32)

    # Generate ink skipping pattern (characteristic of ballpoint pens)
    skip_mask = _generate_ink_skip(ink_float, skip_probability=0.05)
    ink_float[skip_mask] = 0.0

    # Add slight ink build-up at high-pressure points
    density = np.ones_like(ink_float)
    # Thicken slightly where ink accumulates
    dilated = _simple_dilation(ink_float)
    buildup = (dilated - ink_float) > 0.5
    density[buildup] = 1.1

    alpha = ink_float * density
    alpha = np.clip(alpha, 0, 1)

    return _create_ink_image(image.size, ink_color, alpha)


def _apply_reed_pen(
    image: Image.Image,
    ink_color: tuple[int, int, int],
) -> Image.Image:
    """Simulate ancient reed pen: broad strokes with ink splatter.

    Reed pens produce broad, uneven strokes with occasional ink splatters.
    """
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    ink_mask = arr < 200

    if not np.any(ink_mask):
        return _create_ink_image(image.size, ink_color, np.zeros_like(arr))

    ink_float = ink_mask.astype(np.float32)

    # Add broader strokes (dilate slightly)
    broad = _simple_dilation(ink_float, iterations=2)

    # Add ink splatter effect
    splatter = _generate_splatter(broad.shape, splatter_count=5)
    combined = np.clip(broad + splatter, 0, 1)

    # Add uneven ink density
    noise = np.random.RandomState(42).uniform(0.7, 1.3, combined.shape).astype(np.float32)
    alpha = combined * noise
    alpha = np.clip(alpha, 0, 1)

    result = _create_ink_image(image.size, ink_color, alpha)
    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _create_ink_image(
    size: tuple[int, int],
    ink_color: tuple[int, int, int],
    alpha_mask: np.ndarray,
) -> Image.Image:
    """Create an RGBA image with the given ink color and alpha mask."""
    width, height = size
    r, g, b = ink_color

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = r
    rgba[:, :, 1] = g
    rgba[:, :, 2] = b
    rgba[:, :, 3] = (alpha_mask * 255).astype(np.uint8)

    return Image.fromarray(rgba, "RGBA")


def _convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution without scipy dependency."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="edge")
    result = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            result[i, j] = np.sum(region * kernel)

    return result


def _simple_erosion(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple binary erosion using a 3x3 kernel."""
    result = binary.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        eroded = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + 3, j : j + 3]
                eroded[i, j] = np.min(region)
        result = eroded
    return result


def _simple_dilation(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple binary dilation using a 3x3 kernel."""
    result = binary.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        dilated = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded[i : i + 3, j : j + 3]
                dilated[i, j] = np.max(region)
        result = dilated
    return result


def _generate_ink_skip(
    ink_mask: np.ndarray,
    skip_probability: float = 0.03,
) -> np.ndarray:
    """Generate a mask of pixels where ink skips (goes transparent)."""
    rng = np.random.RandomState(123)
    random_field = rng.random(ink_mask.shape).astype(np.float32)

    # Only skip within ink regions
    skip_mask = (random_field < skip_probability) & (ink_mask > 0.5)

    # Also skip near edges (start of strokes)
    eroded = _simple_erosion(ink_mask)
    edge = ink_mask - eroded
    edge_skip = (rng.random(ink_mask.shape) < skip_probability * 2) & (edge > 0.5)

    return skip_mask | edge_skip


def _generate_splatter(
    shape: tuple[int, int],
    splatter_count: int = 5,
) -> np.ndarray:
    """Generate random ink splatter dots."""
    rng = np.random.RandomState(456)
    splatter = np.zeros(shape, dtype=np.float32)

    for _ in range(splatter_count):
        cx = rng.randint(0, shape[1])
        cy = rng.randint(0, shape[0])
        radius = rng.randint(1, 4)

        y_lo = max(0, cy - radius)
        y_hi = min(shape[0], cy + radius + 1)
        x_lo = max(0, cx - radius)
        x_hi = min(shape[1], cx + radius + 1)

        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist <= radius:
                    splatter[y, x] = 1.0

    return splatter


def _add_pressure_variation(
    image: Image.Image,
    ink_mask: np.ndarray,
) -> Image.Image:
    """Add opacity variation to simulate pressure changes."""
    arr = np.array(image.convert("RGBA"), dtype=np.float32)

    # Create a smooth pressure gradient
    h, w = ink_mask.shape[:2]
    gradient = np.linspace(0.8, 1.0, w, dtype=np.float32)
    pressure = np.tile(gradient, (h, 1))

    # Apply to alpha channel where there's ink
    ink_bool = ink_mask > 0.5
    if np.any(ink_bool):
        arr[ink_bool, 3] *= pressure[ink_bool]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


__all__ = ["HistoricalInstrument", "apply_historical_style"]
