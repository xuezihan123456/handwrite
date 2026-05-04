"""Character segmentation from binarized handwriting images.

Strategies:
  1. Connected-component analysis for isolated characters
  2. Vertical projection profile for splitting touching regions
  3. Over-segmentation merging to handle sticky/connected strokes

Assumes white ink on black background (foreground = 255).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CharBox:
    """A bounding box for a single segmented character."""

    x: int
    y: int
    w: int
    h: int
    image: np.ndarray  # cropped character image (white ink on black)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2

    @property
    def center_y(self) -> int:
        return self.y + self.h // 2


class CharacterSegmenter:
    """Segment individual characters from a binarized handwriting image."""

    def __init__(
        self,
        *,
        min_char_height: int = 10,
        max_char_height: int = 800,
        min_char_width: int = 5,
        min_char_area: int = 50,
        merge_gap_ratio: float = 0.3,
        sticky_split_threshold: float = 2.5,
    ) -> None:
        """
        Parameters
        ----------
        min_char_height:
            Minimum height (px) for a valid character region.
        max_char_height:
            Maximum height (px); larger regions are ignored (likely noise).
        min_char_width:
            Minimum width (px) for a valid character region.
        min_char_area:
            Minimum pixel area for a valid character region.
        merge_gap_ratio:
            Ratio of median char width; gaps smaller than this are merged.
        sticky_split_threshold:
            Width/height ratio above which a region is considered sticky
            and will be split via vertical projection.
        """
        self.min_char_height = min_char_height
        self.max_char_height = max_char_height
        self.min_char_width = min_char_width
        self.min_char_area = min_char_area
        self.merge_gap_ratio = merge_gap_ratio
        self.sticky_split_threshold = sticky_split_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, binary: np.ndarray) -> list[CharBox]:
        """Segment characters from a binarized image.

        Parameters
        ----------
        binary:
            Binarized image, white ink (255) on black (0) background.

        Returns
        -------
        list[CharBox]
            Sorted left-to-right list of character bounding boxes with
            cropped images.
        """
        # Phase 1: connected-component analysis
        raw_boxes = self._connected_components(binary)

        if not raw_boxes:
            return []

        # Phase 2: split overly wide (sticky) regions
        expanded: list[tuple[int, int, int, int]] = []
        for box in raw_boxes:
            x, y, w, h = box
            aspect = w / max(h, 1)
            if aspect > self.sticky_split_threshold and w > self.min_char_width * 2:
                splits = self._split_sticky_region(binary, box)
                expanded.extend(splits)
            else:
                expanded.append(box)

        # Phase 3: merge small gaps between nearby boxes
        merged = self._merge_nearby(expanded)

        # Phase 4: build CharBox objects with cropped images
        result: list[CharBox] = []
        for x, y, w, h in merged:
            crop = binary[y:y + h, x:x + w].copy()
            result.append(CharBox(x=x, y=y, w=w, h=h, image=crop))

        # Sort left-to-right
        result.sort(key=lambda cb: cb.x)
        return result

    # ------------------------------------------------------------------
    # Connected-component analysis
    # ------------------------------------------------------------------

    def _connected_components(
        self, binary: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Find character regions via connected-component analysis."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )

        boxes: list[tuple[int, int, int, int]] = []
        for i in range(1, num_labels):  # skip background (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if h < self.min_char_height or h > self.max_char_height:
                continue
            if w < self.min_char_width:
                continue
            if area < self.min_char_area:
                continue

            boxes.append((x, y, w, h))

        return boxes

    # ------------------------------------------------------------------
    # Sticky region splitting
    # ------------------------------------------------------------------

    def _split_sticky_region(
        self, binary: np.ndarray, box: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Split a wide connected region using vertical projection."""
        x, y, w, h = box
        region = binary[y:y + h, x:x + w]

        # Vertical projection: count white pixels per column
        projection = np.sum(region > 0, axis=0).astype(np.int32)

        # Find valleys (low-count columns) as split points
        split_cols = self._find_projection_valleys(projection)

        if not split_cols:
            return [box]

        # Build sub-regions from split points
        boundaries = [0] + split_cols + [w]
        sub_boxes: list[tuple[int, int, int, int]] = []
        for i in range(len(boundaries) - 1):
            sx = boundaries[i]
            ex = boundaries[i + 1]
            sw = ex - sx
            if sw >= self.min_char_width:
                sub_boxes.append((x + sx, y, sw, h))

        return sub_boxes if sub_boxes else [box]

    @staticmethod
    def _find_projection_valleys(projection: np.ndarray) -> list[int]:
        """Find valley positions in the vertical projection profile.

        A valley is a local minimum where the projection count drops
        below the mean value.
        """
        if len(projection) < 5:
            return []

        mean_val = float(np.mean(projection))
        threshold = max(mean_val * 0.4, 1.0)

        valleys: list[int] = []
        in_valley = False
        valley_start = 0

        for i, val in enumerate(projection):
            if val <= threshold:
                if not in_valley:
                    in_valley = True
                    valley_start = i
            else:
                if in_valley:
                    # Use the midpoint of the valley
                    mid = (valley_start + i) // 2
                    valleys.append(mid)
                    in_valley = False

        return valleys

    # ------------------------------------------------------------------
    # Gap merging
    # ------------------------------------------------------------------

    def _merge_nearby(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge boxes that are too close together (likely one character)."""
        if len(boxes) < 2:
            return boxes

        # Sort by x
        sorted_boxes = sorted(boxes, key=lambda b: b[0])

        # Compute median width for gap threshold
        widths = [b[2] for b in sorted_boxes]
        median_w = float(np.median(widths))
        max_gap = int(median_w * self.merge_gap_ratio)

        merged: list[tuple[int, int, int, int]] = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            prev = merged[-1]
            gap = box[0] - (prev[0] + prev[2])

            if 0 < gap <= max_gap:
                # Merge: take the bounding union
                nx = min(prev[0], box[0])
                ny = min(prev[1], box[1])
                nr = max(prev[0] + prev[2], box[0] + box[2])
                nb = max(prev[1] + prev[3], box[1] + box[3])
                merged[-1] = (nx, ny, nr - nx, nb - ny)
            else:
                merged.append(box)

        return merged


__all__ = ["CharacterSegmenter", "CharBox"]
