"""Simulate writing speed effects.

Faster strokes produce thinner, lighter marks with slight blur.
Slower strokes are thicker and more saturated.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter

from .stroke_analyzer import analyze_stroke_structure


def simulate_speed(
    image: np.ndarray,
    threshold: int = 128,
    speed_variation: float = 0.3,
    base_speed: float = 0.5,
) -> np.ndarray:
    """Apply speed simulation to a character image.

    Faster strokes get slight Gaussian blur and become lighter/thinner.
    Slower strokes remain sharp and dark.

    Args:
        image: Grayscale numpy array (0=black ink, 255=white background).
        threshold: Binarization threshold.
        speed_variation: How much speed varies across the stroke (0.0-1.0).
        base_speed: Base writing speed (0.0=slow/heavy, 1.0=fast/light).

    Returns:
        Grayscale numpy array with speed effects applied.
    """
    analysis = analyze_stroke_structure(image, threshold)
    path = analysis["path"]
    dist = analysis["distance_map"]
    binary = image < threshold

    result = image.copy().astype(np.float64)

    if len(path) < 3:
        return image.copy()

    # Generate speed profile: speed varies along the stroke
    # Natural handwriting: accelerate in middle, slow at start/end
    n = len(path)
    t = np.linspace(0.0, 1.0, n)

    # Bell-shaped speed profile: slow at endpoints, fast in middle
    speed_profile = base_speed + speed_variation * (
        4.0 * t * (1.0 - t) - 0.5
    )
    speed_profile = np.clip(speed_profile, 0.0, 1.0)

    # Create a speed map for the whole image
    speed_map = np.full(image.shape, base_speed, dtype=np.float64)

    if len(path) > 1:
        from scipy.spatial import cKDTree

        path_arr = np.array(path, dtype=np.float64)
        ink_coords = np.argwhere(binary)

        if len(ink_coords) > 0:
            tree = cKDTree(path_arr)
            _, indices = tree.query(ink_coords, k=1)
            ink_speeds = speed_profile[indices]

            for idx, (r, c) in enumerate(ink_coords):
                speed_map[r, c] = ink_speeds[idx]

    # Apply speed effects:
    # 1. Fast strokes -> lighter (add white)
    # 2. Fast strokes -> thinner (erode slightly)
    # 3. Fast strokes -> slight blur

    # Lightening effect based on speed
    lightening = speed_map * speed_variation * 0.4  # max 40% lighter
    result[binary] = np.clip(
        result[binary] + lightening[binary] * 255, 0, 255
    )

    # Thinning effect: fast regions get slightly eroded
    fast_threshold = base_speed + speed_variation * 0.3
    fast_pixels = binary & (speed_map > fast_threshold)

    if fast_pixels.any():
        # Create a slightly eroded version for fast regions
        eroded = ndimage.binary_erosion(binary, iterations=1)
        thin_mask = binary & (~eroded) & (speed_map > fast_threshold)
        # Make thin pixels lighter (they're at the stroke edge)
        result[thin_mask] = np.clip(
            result[thin_mask] + speed_variation * 100, 0, 255
        )

    # Blur effect for fast strokes
    if speed_variation > 0.1:
        # Apply Gaussian blur to the whole image
        blur_sigma = speed_variation * 1.0
        blurred = ndimage.gaussian_filter(result, sigma=blur_sigma)

        # Blend: faster regions use more blur, slower regions keep sharp
        blend_factor = speed_map * speed_variation * 0.3
        result = result * (1.0 - blend_factor) + blurred * blend_factor

    return np.clip(result, 0, 255).astype(np.uint8)


def speed_profile(
    path: list[tuple[int, int]],
    base_speed: float = 0.5,
    speed_variation: float = 0.3,
) -> np.ndarray:
    """Generate a speed profile along a stroke path.

    Args:
        path: Ordered (row, col) coordinates.
        base_speed: Base speed (0.0-1.0).
        speed_variation: Speed variation amount (0.0-1.0).

    Returns:
        1-D numpy array of speed values in [0, 1].
    """
    n = len(path)
    if n == 0:
        return np.array([], dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)
    speed = base_speed + speed_variation * (4.0 * t * (1.0 - t) - 0.5)
    return np.clip(speed, 0.0, 1.0)
