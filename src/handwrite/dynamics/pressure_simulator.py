"""Simulate writing pressure to generate stroke width variation.

Uses the distance transform and stroke analysis to create a pressure-modulated
image where stroke width varies naturally: heavier at the start, lighter at the end.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from PIL import Image

from .stroke_analyzer import analyze_stroke_structure, get_pressure_profile


def simulate_pressure(
    image: np.ndarray,
    threshold: int = 128,
    pressure_strength: float = 0.5,
    pressure_curve: str = "natural",
) -> np.ndarray:
    """Apply pressure simulation to a character image.

    Args:
        image: Grayscale numpy array (0=black ink, 255=white background).
        threshold: Binarization threshold.
        pressure_strength: How strongly pressure affects stroke width (0.0-1.0).
        pressure_curve: Type of pressure curve ('natural', 'heavy_start', 'uniform').

    Returns:
        Grayscale numpy array with pressure-modulated stroke widths.
    """
    analysis = analyze_stroke_structure(image, threshold)
    dist = analysis["distance_map"]
    path = analysis["path"]
    stroke_zones = analysis["stroke_zones"]
    binary = image < threshold

    if len(path) == 0:
        return image.copy()

    pressure_profile = get_pressure_profile(path, dist, pressure_curve)

    # Build a pressure map from the skeleton pressure values
    pressure_map = np.full(image.shape, 0.5, dtype=np.float64)

    for i, (r, c) in enumerate(path):
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            pressure_map[r, c] = pressure_profile[i]

    # Propagate pressure values to nearby ink pixels using distance-weighted averaging
    # Use the distance transform to find pixels near the skeleton
    dist_to_skeleton = ndimage.distance_transform_edt(1 - (pressure_map > 0).astype(int) * binary)

    # For each ink pixel, find the nearest skeleton point's pressure
    # This is computationally expensive; use a dilation-based approach instead
    result = image.copy().astype(np.float64)

    # Scale pressure to width multiplier: high pressure = wider/darker stroke
    # We simulate this by morphological operations
    for pressure_level in np.linspace(0.0, 1.0, 8):
        # Pixels at this pressure level
        level_mask = np.zeros(image.shape, dtype=bool)
        for i, (r, c) in enumerate(path):
            if abs(pressure_profile[i] - pressure_level) < 0.1:
                if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
                    level_mask[r, c] = True

        if not level_mask.any():
            continue

        # Expand this skeleton region to cover nearby ink
        expanded = ndimage.binary_dilation(level_mask, iterations=int(dist.max()) + 1)
        ink_region = expanded & binary

        # Apply pressure effect: high pressure = darker, low pressure = lighter
        darkness = pressure_level * pressure_strength
        # Where pressure is high, keep ink darker; where low, make it lighter
        modulation = 1.0 - darkness * 0.6  # range [0.7, 1.0]
        result[ink_region] = np.clip(
            result[ink_region] * modulation + (1 - modulation) * 255, 0, 255
        )

    # Simpler fallback: use a global pressure gradient along the stroke path
    # Map each ink pixel to its nearest path point and assign pressure
    if len(path) > 1:
        path_arr = np.array(path, dtype=np.float64)
        ink_coords = np.argwhere(binary)

        if len(ink_coords) > 0:
            # For each ink pixel, find nearest skeleton point (approximate)
            from scipy.spatial import cKDTree

            tree = cKDTree(path_arr)
            _, indices = tree.query(ink_coords, k=1)

            # Get pressure at nearest skeleton point
            ink_pressures = pressure_profile[indices]

            # Modulate: high pressure -> darker (closer to 0), low pressure -> lighter (closer to 255)
            strength = pressure_strength
            for idx, (r, c) in enumerate(ink_coords):
                p = ink_pressures[idx]
                # Blend original ink with pressure-modulated value
                darkening = (1.0 - p) * strength * 0.5
                result[r, c] = np.clip(result[r, c] + darkening * 255, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def pressure_width_map(
    image: np.ndarray,
    threshold: int = 128,
    pressure_curve: str = "natural",
) -> np.ndarray:
    """Generate a pressure-based width multiplier map.

    Returns a float array where values > 1.0 mean wider strokes and
    values < 1.0 mean narrower strokes. Can be used by other simulators.

    Args:
        image: Grayscale numpy array.
        threshold: Binarization threshold.
        pressure_curve: Type of pressure curve.

    Returns:
        Float64 array same shape as image, values in [0.5, 1.5].
    """
    analysis = analyze_stroke_structure(image, threshold)
    dist = analysis["distance_map"]
    path = analysis["path"]
    binary = image < threshold

    width_map = np.ones(image.shape, dtype=np.float64)

    if len(path) < 2:
        return width_map

    pressure_profile = get_pressure_profile(path, dist, pressure_curve)

    path_arr = np.array(path, dtype=np.float64)
    ink_coords = np.argwhere(binary)

    if len(ink_coords) == 0:
        return width_map

    from scipy.spatial import cKDTree

    tree = cKDTree(path_arr)
    _, indices = tree.query(ink_coords, k=1)

    ink_pressures = pressure_profile[indices]

    # Map pressure to width: high pressure (1.0) -> 1.5x width, low (0.0) -> 0.7x width
    for idx, (r, c) in enumerate(ink_coords):
        width_map[r, c] = 0.7 + 0.8 * ink_pressures[idx]

    return width_map
