"""Analyze stroke structure of a character image.

Identifies start-stroke, mid-stroke, and end-stroke regions by computing
a distance transform skeleton and walking along connected components.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert grayscale image to binary mask (True = ink pixel)."""
    return image < threshold


def _distance_transform(binary: np.ndarray) -> np.ndarray:
    """Compute distance transform from ink pixels to background."""
    inverted = ~binary
    dist = ndimage.distance_transform_edt(inverted)
    return dist


def _thin_skeleton(binary: np.ndarray) -> np.ndarray:
    """Extract a 1-pixel-wide skeleton using morphological thinning."""
    # Simple iterative thinning (Zhang-Suen like approximation)
    # For production use, consider skimage.morphology.thin
    try:
        from skimage.morphology import thin as _skimage_thin
        return _skimage_thin(binary).astype(np.uint8)
    except ImportError:
        pass

    # Fallback: use distance transform ridge detection
    dist = _distance_transform(binary)
    if dist.max() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    # Find local maxima of distance transform as approximate skeleton
    kernel = np.ones((3, 3), dtype=bool)
    kernel[1, 1] = False
    local_max = ndimage.maximum_filter(dist, footprint=kernel)
    skeleton = (dist > 0) & (dist >= local_max)
    return skeleton.astype(np.uint8)


def _trace_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Trace skeleton pixels into an ordered path.

    Returns a list of (row, col) coordinates ordered along the stroke.
    Uses a simple neighbor-following heuristic.
    """
    coords = np.argwhere(skeleton > 0)
    if len(coords) == 0:
        return []

    # Build adjacency via 8-connectivity
    visited = set()
    path: list[tuple[int, int]] = []
    coord_set = {tuple(c) for c in coords}

    # Start from an endpoint (pixel with 1 neighbor) or arbitrary point
    start = tuple(coords[0])
    for c in coords:
        r, col = int(c[0]), int(c[1])
        neighbor_count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, col + dc) in coord_set:
                    neighbor_count += 1
        if neighbor_count == 1:
            start = (r, col)
            break

    # Walk the skeleton
    current = start
    while current not in visited:
        visited.add(current)
        path.append(current)
        r, c = current
        next_pixel = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r + dr, c + dc)
                if neighbor in coord_set and neighbor not in visited:
                    next_pixel = neighbor
                    break
            if next_pixel is not None:
                break
        if next_pixel is None:
            break
        current = next_pixel

    return path


def analyze_stroke_structure(
    image: np.ndarray,
    threshold: int = 128,
) -> dict[str, object]:
    """Analyze stroke structure of a character image.

    Args:
        image: Grayscale numpy array (0=black ink, 255=white background).
        threshold: Binarization threshold.

    Returns:
        Dictionary with:
        - ``skeleton``: binary skeleton array
        - ``distance_map``: distance transform array
        - ``path``: ordered list of (row, col) along the skeleton
        - ``stroke_zones``: array same shape as image with zone labels
          (0=background, 1=start, 2=mid, 3=end)
        - ``stroke_length``: number of skeleton pixels
    """
    binary = _binarize(image, threshold)
    skeleton = _thin_skeleton(binary)
    dist = _distance_transform(binary)
    path = _trace_skeleton_path(skeleton)

    # Build stroke zone map
    stroke_zones = np.zeros(image.shape, dtype=np.uint8)
    n = len(path)

    if n == 0:
        return {
            "skeleton": skeleton,
            "distance_map": dist,
            "path": path,
            "stroke_zones": stroke_zones,
            "stroke_length": 0,
        }

    # Label zones along the skeleton path
    start_end = max(1, n // 5)  # first 20% = start
    end_begin = n - max(1, n // 5)  # last 20% = end

    for i, (r, c) in enumerate(path):
        if i < start_end:
            zone = 1  # start stroke
        elif i >= end_begin:
            zone = 3  # end stroke
        else:
            zone = 2  # mid stroke
        stroke_zones[r, c] = zone

    # Dilate zone labels to fill the stroke width
    for zone_val in (1, 2, 3):
        zone_mask = (stroke_zones == zone_val).astype(np.uint8)
        if zone_mask.max() == 0:
            continue
        # Grow zone mask to cover nearby ink pixels
        grown = ndimage.binary_dilation(
            zone_mask, iterations=int(dist.max()) + 1
        )
        stroke_zones[grown & (stroke_zones == 0) & binary] = zone_val

    # Assign remaining ink pixels to mid-stroke
    stroke_zones[(stroke_zones == 0) & binary] = 2

    return {
        "skeleton": skeleton,
        "distance_map": dist,
        "path": path,
        "stroke_zones": stroke_zones,
        "stroke_length": n,
    }


def get_pressure_profile(
    path: list[tuple[int, int]],
    distance_map: np.ndarray,
    pressure_curve: str = "natural",
) -> np.ndarray:
    """Generate a pressure profile along a stroke path.

    Args:
        path: Ordered (row, col) coordinates along skeleton.
        distance_map: Distance transform of the character.
        pressure_curve: Type of pressure curve ('natural', 'heavy_start', 'uniform').

    Returns:
        1-D numpy array of pressure values in [0, 1] for each point on the path.
    """
    n = len(path)
    if n == 0:
        return np.array([], dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)

    if pressure_curve == "natural":
        # Natural handwriting: heavy start, lighter end, slight bump in middle
        pressure = 1.0 - 0.4 * t + 0.1 * np.sin(np.pi * t)
    elif pressure_curve == "heavy_start":
        # Strong start, tapering off
        pressure = 1.0 - 0.6 * t
    elif pressure_curve == "uniform":
        pressure = np.ones(n, dtype=np.float64)
    else:
        pressure = 1.0 - 0.3 * t

    # Modulate by local stroke width (wider strokes get slightly more pressure)
    for i, (r, c) in enumerate(path):
        if 0 <= r < distance_map.shape[0] and 0 <= c < distance_map.shape[1]:
            width_factor = distance_map[r, c] / max(distance_map.max(), 1.0)
            pressure[i] *= 0.7 + 0.3 * width_factor

    # Normalize to [0, 1]
    p_min, p_max = pressure.min(), pressure.max()
    if p_max > p_min:
        pressure = (pressure - p_min) / (p_max - p_min)
    else:
        pressure = np.ones(n, dtype=np.float64) * 0.5

    return pressure
