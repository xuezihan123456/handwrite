"""Lighting adjustment for realistic AR overlay.

Matches the brightness and contrast of the handwriting overlay to the
target paper region, and optionally applies subtle shadow effects for
a more natural integration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class LightingAdjuster:
    """Adjusts overlay brightness and contrast to match the target paper.

    Parameters:
        brightness_weight: How strongly to match brightness (0.0 = no match,
            1.0 = full match).
        contrast_weight: How strongly to match contrast (0.0 = no match,
            1.0 = full match).
        shadow_strength: Opacity of the shadow gradient applied to edges
            of the paper region (0.0 = no shadow, 1.0 = full shadow).
        shadow_blur_size: Kernel size for the shadow blur effect. Must be odd.
    """

    def __init__(
        self,
        brightness_weight: float = 0.6,
        contrast_weight: float = 0.4,
        shadow_strength: float = 0.15,
        shadow_blur_size: int = 51,
    ) -> None:
        if not 0.0 <= brightness_weight <= 1.0:
            raise ValueError("brightness_weight must be in [0, 1]")
        if not 0.0 <= contrast_weight <= 1.0:
            raise ValueError("contrast_weight must be in [0, 1]")
        if not 0.0 <= shadow_strength <= 1.0:
            raise ValueError("shadow_strength must be in [0, 1]")
        if shadow_blur_size < 1 or shadow_blur_size % 2 == 0:
            raise ValueError("shadow_blur_size must be a positive odd number")

        self._brightness_weight = brightness_weight
        self._contrast_weight = contrast_weight
        self._shadow_strength = shadow_strength
        self._shadow_blur_size = shadow_blur_size

    def match_lighting(
        self,
        overlay: np.ndarray,
        target_region: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Match the overlay's lighting to the target paper region.

        Adjusts the overlay's mean brightness and standard deviation (contrast)
        to match those of the target region.

        Parameters:
            overlay: The handwriting overlay image, shape ``(H, W, 3)``
                in BGR, dtype ``uint8``.
            target_region: The paper region from the original photo,
                same shape as ``overlay``.
            target_mask: Optional binary mask for the target region.
                If provided, statistics are computed only inside the mask.

        Returns:
            Lighting-adjusted overlay image, same shape and dtype as input.
        """
        if overlay.shape != target_region.shape:
            raise ValueError(
                f"Shape mismatch: overlay {overlay.shape} vs "
                f"target {target_region.shape}"
            )

        overlay_lab = cv2.cvtColor(overlay, cv2.COLOR_BGR2LAB).astype(np.float64)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float64)

        # Compute statistics for L channel
        if target_mask is not None:
            mask_bool = target_mask > 0
            target_l = target_lab[:, :, 0][mask_bool]
            overlay_l = overlay_lab[:, :, 0]
        else:
            target_l = target_lab[:, :, 0].ravel()
            overlay_l = overlay_lab[:, :, 0]

        target_mean = float(np.mean(target_l))
        target_std = float(np.std(target_l)) + 1e-6
        overlay_mean = float(np.mean(overlay_l))
        overlay_std = float(np.std(overlay_l)) + 1e-6

        # Adjust L channel: blend original with matched values
        bw = self._brightness_weight
        cw = self._contrast_weight

        adjusted_l = overlay_l.copy()
        # Brightness match
        brightness_shift = target_mean - overlay_mean
        adjusted_l = adjusted_l + bw * brightness_shift
        # Contrast match
        contrast_scale = target_std / overlay_std
        adjusted_l = (
            overlay_mean
            + (adjusted_l - overlay_mean) * (1.0 + cw * (contrast_scale - 1.0))
        )

        overlay_lab[:, :, 0] = np.clip(adjusted_l, 0, 255)

        result = cv2.cvtColor(overlay_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def apply_shadow(
        self,
        image: np.ndarray,
        paper_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply a subtle shadow gradient at the paper edges.

        Creates a natural-looking shadow by darkening the area near the
        paper boundary.

        Parameters:
            image: Input BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
            paper_mask: Binary mask of the paper region, shape ``(H, W)``.

        Returns:
            Image with shadow applied.
        """
        if self._shadow_strength == 0.0:
            return image

        # Compute distance from paper edge (inside the paper)
        dist_inside = cv2.distanceTransform(
            paper_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        # Normalize to [0, 1]
        max_dist = float(dist_inside.max()) or 1.0
        dist_norm = dist_inside / max_dist

        # Shadow factor: stronger at edges, fading inward
        shadow_factor = 1.0 - self._shadow_strength * (1.0 - dist_norm)

        # Smooth the shadow
        shadow_factor = cv2.GaussianBlur(
            shadow_factor.astype(np.float32),
            (self._shadow_blur_size, self._shadow_blur_size),
            0,
        )

        # Apply shadow only inside paper region
        result = image.astype(np.float64)
        mask_3d = np.stack([paper_mask > 0] * 3, axis=-1)
        shadow_3d = np.stack([shadow_factor] * 3, axis=-1)
        result = np.where(mask_3d, result * shadow_3d, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    def compute_paper_lighting(
        self,
        paper_region: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Compute mean brightness and contrast of a paper region.

        Parameters:
            paper_region: BGR image of the paper area.
            mask: Optional binary mask.

        Returns:
            Tuple of ``(mean_brightness, std_contrast)`` for the L channel.
        """
        lab = cv2.cvtColor(paper_region, cv2.COLOR_BGR2LAB).astype(np.float64)
        l_channel = lab[:, :, 0]

        if mask is not None:
            l_values = l_channel[mask > 0]
        else:
            l_values = l_channel.ravel()

        return float(np.mean(l_values)), float(np.std(l_values))
