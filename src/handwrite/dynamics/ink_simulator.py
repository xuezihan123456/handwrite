"""Simulate ink diffusion and density variation.

Creates realistic ink density gradients: denser (darker) where the pen
first touches down and lighter as ink flows out over time.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from PIL import Image

from .stroke_analyzer import analyze_stroke_structure, get_pressure_profile


def simulate_ink(
    image: np.ndarray,
    threshold: int = 128,
    ink_density: float = 0.6,
    diffusion_radius: int = 2,
    pressure_curve: str = "natural",
) -> np.ndarray:
    """Apply ink density simulation to a character image.

    Args:
        image: Grayscale numpy array (0=black ink, 255=white background).
        threshold: Binarization threshold.
        ink_density: Base ink density (0.0=light, 1.0=dark).
        diffusion_radius: Radius of ink diffusion effect in pixels.
        pressure_curve: Type of pressure curve for density mapping.

    Returns:
        Grayscale numpy array with ink density gradients applied.
    """
    analysis = analyze_stroke_structure(image, threshold)
    dist = analysis["distance_map"]
    path = analysis["path"]
    binary = image < threshold

    result = image.copy().astype(np.float64)

    if len(path) < 2:
        return _apply_uniform_ink(result, binary, ink_density)

    # Generate density profile along stroke (similar to pressure but for ink)
    pressure_profile = get_pressure_profile(path, dist, pressure_curve)
    # Invert slightly: ink is densest at start, fading toward end
    density_profile = 1.0 - 0.3 * np.linspace(0.0, 1.0, len(path))
    density_profile = np.clip(density_profile * ink_density, 0.0, 1.0)

    # Map ink pixels to nearest skeleton point
    from scipy.spatial import cKDTree

    path_arr = np.array(path, dtype=np.float64)
    ink_coords = np.argwhere(binary)

    if len(ink_coords) == 0:
        return result.astype(np.uint8)

    tree = cKDTree(path_arr)
    _, indices = tree.query(ink_coords, k=1)
    ink_densities = density_profile[indices]

    # Apply ink diffusion: slightly blur the density map
    density_map = np.ones(image.shape, dtype=np.float64)
    for idx, (r, c) in enumerate(ink_coords):
        density_map[r, c] = ink_densities[idx]

    # Smooth the density map to simulate ink diffusion
    if diffusion_radius > 0:
        density_map = ndimage.gaussian_filter(density_map, sigma=diffusion_radius)

    # Apply density: darker where density is higher
    # Original ink pixels: modulate darkness by density
    for idx, (r, c) in enumerate(ink_coords):
        d = density_map[r, c]
        # Higher density -> darker (lower value)
        target = 255.0 * (1.0 - d)
        # Blend original with density target
        original = result[r, c]
        result[r, c] = original * (1.0 - d * 0.7) + target * d * 0.7

    # Add ink bleeding effect at stroke edges
    if diffusion_radius > 0:
        edge_mask = _detect_stroke_edges(binary, dist)
        bleed = ndimage.gaussian_filter(
            (binary.astype(np.float64) * density_map), sigma=diffusion_radius * 1.5
        )
        # Apply subtle bleed to edge pixels
        bleed_intensity = 0.3 * ink_density
        edge_ink = edge_mask & (~binary) & (bleed > 0.05)
        result[edge_ink] = np.clip(
            result[edge_ink] * (1.0 - bleed[edge_ink] * bleed_intensity),
            0, 255,
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_uniform_ink(
    image: np.ndarray, binary: np.ndarray, ink_density: float
) -> np.ndarray:
    """Apply uniform ink density when no skeleton path is found."""
    result = image.copy()
    modulation = 1.0 - ink_density * 0.4
    result[binary] = np.clip(result[binary] * modulation, 0, 255)
    return result.astype(np.uint8)


def _detect_stroke_edges(binary: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Detect edge pixels of strokes (pixels just outside the ink region)."""
    # Edge = background pixels adjacent to ink
    dilated = ndimage.binary_dilation(binary, iterations=1)
    return dilated & (~binary)


def ink_density_map(
    image: np.ndarray,
    threshold: int = 128,
    ink_density: float = 0.6,
    pressure_curve: str = "natural",
) -> np.ndarray:
    """Generate an ink density map for the character.

    Returns a float array in [0, 1] where 1.0 = maximum ink density.

    Args:
        image: Grayscale numpy array.
        threshold: Binarization threshold.
        ink_density: Base ink density scaling factor.
        pressure_curve: Pressure curve type.

    Returns:
        Float64 density map same shape as image.
    """
    analysis = analyze_stroke_structure(image, threshold)
    dist = analysis["distance_map"]
    path = analysis["path"]
    binary = image < threshold

    density = np.zeros(image.shape, dtype=np.float64)

    if len(path) < 2:
        density[binary] = ink_density
        return density

    pressure_profile = get_pressure_profile(path, dist, pressure_curve)
    density_profile = 1.0 - 0.3 * np.linspace(0.0, 1.0, len(path))
    density_profile *= ink_density

    from scipy.spatial import cKDTree

    path_arr = np.array(path, dtype=np.float64)
    ink_coords = np.argwhere(binary)

    if len(ink_coords) == 0:
        return density

    tree = cKDTree(path_arr)
    _, indices = tree.query(ink_coords, k=1)

    for idx, (r, c) in enumerate(ink_coords):
        density[r, c] = density_profile[indices[idx]]

    return density
