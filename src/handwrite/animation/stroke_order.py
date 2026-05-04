"""Stroke extraction and ordering from character images.

Uses morphological skeletonization to extract stroke center lines,
then traces paths and orders them by spatial position to approximate
Chinese stroke order (top-to-bottom, left-to-right).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def extract_strokes(
    char_image: Image.Image,
    threshold: int = 200,
    min_stroke_length: int = 5,
) -> list[list[tuple[int, int]]]:
    """Extract ordered stroke paths from a character image.

    Args:
        char_image: Grayscale character image (white background, dark ink).
        threshold: Binarization threshold (pixels darker than this are ink).
        min_stroke_length: Minimum number of points for a valid stroke.

    Returns:
        Ordered list of strokes, each stroke is a list of (x, y) points.
        Strokes are ordered top-to-bottom, left-to-right.
    """
    binary = _binarize(char_image, threshold)
    if cv2.countNonZero(binary) == 0:
        return []

    skeleton = _skeletonize(binary)
    if cv2.countNonZero(skeleton) == 0:
        return []

    paths = _trace_paths(skeleton, min_stroke_length)
    return _order_strokes(paths)


def _binarize(image: Image.Image, threshold: int) -> np.ndarray:
    """Convert grayscale image to binary ink mask (255 = ink, 0 = background)."""
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Compute 1-pixel-wide skeleton using Zhang-Suen thinning.

    Guarantees a single-pixel-wide skeleton, unlike morphological
    skeletonization which can produce multi-pixel-wide results for
    thick strokes, causing path tracing to fail.
    """
    img = (binary > 0).astype(np.uint8)
    if not np.any(img):
        return np.zeros_like(binary)

    # Pad with zeros to avoid boundary checks
    padded = np.pad(img, 1, mode="constant", constant_values=0)

    changed = True
    while changed:
        changed = False
        for sub in (1, 2):
            to_del = _zs_mark_deletable(padded, sub)
            if np.any(to_del):
                padded[1:-1, 1:-1][to_del] = 0
                changed = True

    result = padded[1:-1, 1:-1]
    return (result * 255).astype(np.uint8)


def _zs_mark_deletable(padded: np.ndarray, sub_iteration: int) -> np.ndarray:
    """Mark pixels deletable in one Zhang-Suen thinning sub-iteration.

    Args:
        padded: Binary (0/1) image padded with a 1-pixel zero border.
        sub_iteration: 1 or 2.

    Returns:
        Boolean array (same size as unpadded image) marking pixels to remove.
    """
    # Neighborhood: P2=north, P3=NE, P4=east, P5=SE,
    #               P6=south, P7=SW, P8=west, P9=NW
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    center = padded[1:-1, 1:-1]

    # Condition 1: center pixel is foreground
    c1 = center == 1

    # Condition 2: 2 <= number of foreground neighbours <= 6
    b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    c2 = (b >= 2) & (b <= 6)

    # Condition 3: exactly one 0->1 transition in P2,P3,...,P9,P2
    neighbors = np.stack([p2, p3, p4, p5, p6, p7, p8, p9], axis=-1)
    next_neighbors = np.roll(neighbors, -1, axis=-1)
    transitions = np.sum((neighbors == 0) & (next_neighbors == 1), axis=-1)
    c3 = transitions == 1

    # Conditions 4 & 5 differ between sub-iterations
    if sub_iteration == 1:
        c4 = (p2 * p4 * p6) == 0
        c5 = (p4 * p6 * p8) == 0
    else:
        c4 = (p2 * p4 * p8) == 0
        c5 = (p2 * p6 * p8) == 0

    return c1 & c2 & c3 & c4 & c5


def _trace_paths(
    skeleton: np.ndarray,
    min_length: int,
) -> list[list[tuple[int, int]]]:
    """Trace connected paths from the skeleton image.

    Finds endpoints and branch points, then traces paths between them.
    Returns paths as lists of (x, y) coordinate tuples.
    """
    # Get all skeleton pixel positions (row, col)
    rows, cols = np.where(skeleton > 0)
    if len(rows) == 0:
        return []

    pixel_set = set(zip(rows.tolist(), cols.tolist()))

    # Build neighbor counts and adjacency
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pixel_set:
        adj = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixel_set:
                    adj.append((nr, nc))
        neighbor_map[(r, c)] = adj

    # Find endpoints (degree 1) and branch points (degree >= 3)
    endpoints = [p for p, adj in neighbor_map.items() if len(adj) == 1]
    branch_points = [p for p, adj in neighbor_map.items() if len(adj) >= 3]
    special_points = set(endpoints) | set(branch_points)

    # Trace paths starting from endpoints, then branch points, then remaining
    visited: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    # Priority: endpoints first, then branch points
    start_points = list(endpoints)
    for bp in branch_points:
        if bp not in special_points:
            continue
        start_points.append(bp)

    # Add remaining unvisited pixels as potential starts
    for pixel in pixel_set:
        if pixel not in visited and pixel not in special_points:
            start_points.append(pixel)

    for start in start_points:
        if start in visited:
            continue

        for neighbor in neighbor_map.get(start, []):
            if neighbor in visited:
                continue

            path = _trace_single_path(start, neighbor, neighbor_map, visited)
            if len(path) >= min_length:
                paths.append(path)

    return paths


def _trace_single_path(
    start: tuple[int, int],
    next_pixel: tuple[int, int],
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]],
    visited: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Trace a single path from start through next_pixel until hitting a junction or end."""
    path = [start]
    visited.add(start)

    current = next_pixel
    prev = start

    while True:
        if current in visited:
            break

        path.append(current)
        visited.add(current)

        # Find next unvisited neighbor (excluding where we came from)
        neighbors = neighbor_map.get(current, [])
        next_candidates = [n for n in neighbors if n not in visited and n != prev]

        if len(next_candidates) == 0:
            # Dead end
            break
        if len(next_candidates) == 1:
            prev = current
            current = next_candidates[0]
        else:
            # Branch point - stop here (the branch point itself is included)
            break

    return path


def _order_strokes(
    paths: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Order strokes by spatial position: top-to-bottom, then left-to-right.

    Each stroke is represented by its starting point for ordering.
    Converts from (row, col) to (x, y) coordinate convention.
    """
    if not paths:
        return []

    # Sort by starting row (top first), then starting col (left first)
    def sort_key(path: list[tuple[int, int]]) -> tuple[int, int]:
        start_r, start_c = path[0]
        return (start_r, start_c)

    sorted_paths = sorted(paths, key=sort_key)

    # Convert from (row, col) to (x, y)
    result: list[list[tuple[int, int]]] = []
    for path in sorted_paths:
        converted = [(c, r) for r, c in path]
        result.append(converted)

    return result


__all__ = ["extract_strokes"]
