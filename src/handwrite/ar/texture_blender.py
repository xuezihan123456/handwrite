"""Texture blending for realistic handwriting overlay.

Provides alpha blending and ink penetration simulation to make
the overlaid handwriting look as if it was written directly on the paper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class TextureBlender:
    """Blends handwriting overlay onto paper with realistic texture effects.

    Parameters:
        ink_opacity: Base opacity of the ink strokes (0.0 = transparent,
            1.0 = fully opaque).
        paper_texture_weight: How much of the paper texture to preserve
            in the blended result (0.0 = no paper texture, 1.0 = full).
        ink_penetration_depth: Simulated ink penetration into paper fibers,
            creating a slight darkening halo around strokes (0.0 = none,
            1.0 = strong).
        feather_radius: Radius of the edge feathering applied to the blend
            boundary for smoother transitions.
    """

    def __init__(
        self,
        ink_opacity: float = 0.85,
        paper_texture_weight: float = 0.3,
        ink_penetration_depth: float = 0.2,
        feather_radius: int = 3,
    ) -> None:
        if not 0.0 <= ink_opacity <= 1.0:
            raise ValueError("ink_opacity must be in [0, 1]")
        if not 0.0 <= paper_texture_weight <= 1.0:
            raise ValueError("paper_texture_weight must be in [0, 1]")
        if not 0.0 <= ink_penetration_depth <= 1.0:
            raise ValueError("ink_penetration_depth must be in [0, 1]")
        if feather_radius < 0:
            raise ValueError("feather_radius must be >= 0")

        self._ink_opacity = ink_opacity
        self._paper_texture_weight = paper_texture_weight
        self._ink_penetration_depth = ink_penetration_depth
        self._feather_radius = feather_radius

    def alpha_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend foreground onto background using the given mask.

        Parameters:
            background: Background image ``(H, W, 3)``, ``uint8``.
            foreground: Foreground image, same shape as background.
            mask: Alpha mask ``(H, W)``, ``uint8`` (0-255).

        Returns:
            Blended image, same shape and dtype.
        """
        alpha = mask.astype(np.float64) / 255.0
        alpha_3d = np.stack([alpha] * 3, axis=-1)

        bg = background.astype(np.float64)
        fg = foreground.astype(np.float64)
        result = bg * (1.0 - alpha_3d) + fg * alpha_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def blend_handwriting(
        self,
        paper_image: np.ndarray,
        handwriting: np.ndarray,
        paper_mask: np.ndarray,
        ink_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Blend handwriting onto the paper image with texture effects.

        Simulates ink penetration by slightly darkening the paper around
        strokes, then composites the ink with adjustable opacity.

        Parameters:
            paper_image: The original paper photo, BGR ``(H, W, 3)``.
            handwriting: Handwriting image, same shape as paper_image.
                Dark pixels represent ink, light pixels represent empty.
            paper_mask: Binary mask of the paper region ``(H, W)``.
            ink_color: BGR color for the ink. If ``None``, uses the
                handwriting's own colors.

        Returns:
            Blended result image, same shape and dtype as paper_image.
        """
        if paper_image.shape != handwriting.shape:
            raise ValueError(
                f"Shape mismatch: paper {paper_image.shape} vs "
                f"handwriting {handwriting.shape}"
            )

        # Convert handwriting to grayscale for mask extraction
        hw_gray = cv2.cvtColor(handwriting, cv2.COLOR_BGR2GRAY)
        # Ink strokes are darker than paper background
        # Use inverted grayscale as ink mask
        ink_mask_raw = 255 - hw_gray

        # Threshold to separate ink from paper
        _, ink_binary = cv2.threshold(
            ink_mask_raw, 30, 255, cv2.THRESH_BINARY
        )

        # Apply ink penetration effect (dilate and blur for halo)
        if self._ink_penetration_depth > 0:
            penetration_mask = self._create_penetration_mask(ink_binary)
        else:
            penetration_mask = np.zeros_like(ink_binary)

        # Build the final ink alpha
        ink_alpha = self._build_ink_alpha(
            ink_binary, penetration_mask, paper_mask
        )

        # Prepare the ink layer
        ink_layer = self._prepare_ink_layer(
            paper_image, handwriting, ink_color
        )

        # Apply feathering at edges for smoother blending
        if self._feather_radius > 0:
            ink_alpha = self._feather_edges(ink_alpha)

        # Final alpha blend
        return self.alpha_blend(paper_image, ink_layer, ink_alpha)

    def _create_penetration_mask(self, ink_binary: np.ndarray) -> np.ndarray:
        """Create a penetration halo mask around ink strokes.

        Dilates the ink region slightly and applies Gaussian blur to
        simulate ink spreading into paper fibers.
        """
        depth = self._ink_penetration_depth
        kernel_size = max(3, int(7 * depth))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(ink_binary, kernel, iterations=1)

        blur_size = max(3, int(11 * depth))
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(
            dilated.astype(np.float32), (blur_size, blur_size), 0
        )
        return np.clip(blurred * depth, 0, 255).astype(np.uint8)

    def _build_ink_alpha(
        self,
        ink_binary: np.ndarray,
        penetration_mask: np.ndarray,
        paper_mask: np.ndarray,
    ) -> np.ndarray:
        """Build the combined ink alpha channel.

        Combines the solid ink mask with the penetration halo, weighted
        by ink opacity, and restricted to the paper region.
        """
        combined = np.maximum(
            (ink_binary.astype(np.float64) * self._ink_opacity),
            penetration_mask.astype(np.float64) * 0.3,
        )
        # Restrict to paper region
        combined = np.where(paper_mask > 0, combined, 0.0)
        return np.clip(combined, 0, 255).astype(np.uint8)

    def _prepare_ink_layer(
        self,
        paper_image: np.ndarray,
        handwriting: np.ndarray,
        ink_color: Optional[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Prepare the ink layer for blending.

        If ink_color is specified, replaces all ink pixels with that color.
        Otherwise, preserves the original handwriting colors while mixing
        in paper texture.
        """
        if ink_color is None:
            # Mix handwriting with paper texture
            if self._paper_texture_weight > 0:
                hw_gray = cv2.cvtColor(handwriting, cv2.COLOR_BGR2GRAY)
                texture = self._extract_paper_texture(paper_image)
                weight = self._paper_texture_weight
                ink_layer = (
                    handwriting.astype(np.float64) * (1.0 - weight)
                    + texture.astype(np.float64) * weight
                )
                return np.clip(ink_layer, 0, 255).astype(np.uint8)
            return handwriting
        else:
            # Solid ink color
            ink_layer = np.full_like(handwriting, ink_color, dtype=np.uint8)
            return ink_layer

    @staticmethod
    def _extract_paper_texture(image: np.ndarray) -> np.ndarray:
        """Extract paper texture by high-pass filtering.

        Returns an image containing only the fine texture details of the
        paper surface, which can be blended with ink for a natural look.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        texture = cv2.addWeighted(image, 1.5, blurred, -0.5, 128)
        return texture

    def _feather_edges(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to feather mask edges."""
        k = self._feather_radius * 2 + 1
        return cv2.GaussianBlur(mask, (k, k), 0)
