"""AR overlay engine: the main entry point for handwriting-on-paper compositing.

Orchestrates paper detection, perspective transformation, lighting
adjustment, and texture blending to produce a single composite image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from handwrite.ar.paper_detector import PaperDetector, PaperDetectionResult
from handwrite.ar.perspective_transform import PerspectiveTransformer
from handwrite.ar.lighting_adjuster import LightingAdjuster
from handwrite.ar.texture_blender import TextureBlender


@dataclass(frozen=True)
class AROverlayOptions:
    """Configuration for the AR overlay pipeline.

    Attributes:
        ink_color: BGR ink color, or ``None`` to use handwriting's own colors.
        brightness_weight: Strength of brightness matching (0-1).
        contrast_weight: Strength of contrast matching (0-1).
        shadow_strength: Shadow intensity at paper edges (0-1).
        ink_opacity: Ink opacity (0-1).
        paper_texture_weight: How much paper texture to blend into ink (0-1).
        ink_penetration_depth: Ink spreading into paper fibers (0-1).
        feather_radius: Edge feathering radius in pixels.
        output_dpi: DPI metadata for the output image.
    """

    ink_color: Optional[Tuple[int, int, int]] = None
    brightness_weight: float = 0.6
    contrast_weight: float = 0.4
    shadow_strength: float = 0.15
    ink_opacity: float = 0.85
    paper_texture_weight: float = 0.3
    ink_penetration_depth: float = 0.2
    feather_radius: int = 3
    output_dpi: int = 300


@dataclass(frozen=True)
class AROverlayResult:
    """Result of an AR overlay operation.

    Attributes:
        composite: The final composited image as a numpy BGR array.
        paper_detection: The paper detection result (if detection succeeded).
        warped_handwriting: The perspective-warped handwriting before blending.
        paper_mask: Binary mask of the detected paper region in photo space.
    """

    composite: np.ndarray
    paper_detection: Optional[PaperDetectionResult] = None
    warped_handwriting: Optional[np.ndarray] = None
    paper_mask: Optional[np.ndarray] = None


class AREngine:
    """High-level engine for AR handwriting overlay.

    Combines all sub-modules into a single ``overlay`` method that takes
    a paper photograph and a handwriting image, and returns the composited
    result.

    Parameters:
        detector: Paper detector instance. Uses defaults if ``None``.
        lighting: Lighting adjuster instance. Uses defaults if ``None``.
        blender: Texture blender instance. Uses defaults if ``None``.
    """

    def __init__(
        self,
        detector: Optional[PaperDetector] = None,
        lighting: Optional[LightingAdjuster] = None,
        blender: Optional[TextureBlender] = None,
    ) -> None:
        self._detector = detector or PaperDetector()
        self._lighting = lighting or LightingAdjuster()
        self._blender = blender or TextureBlender()

    def overlay(
        self,
        photo: np.ndarray,
        handwriting: np.ndarray,
        paper_corners: Optional[np.ndarray] = None,
        options: Optional[AROverlayOptions] = None,
    ) -> AROverlayResult:
        """Overlay handwriting onto a paper photograph.

        Parameters:
            photo: The paper photograph, BGR ``(H, W, 3)``, ``uint8``.
            handwriting: The handwriting image to overlay, BGR, ``uint8``.
                Will be resized to fit the detected paper region.
            paper_corners: Optional manual paper corners ``(4, 2)`` in
                ``(x, y)`` format (TL, TR, BR, BL). If ``None``, the
                detector will attempt automatic detection.
            options: Overlay configuration. Uses defaults if ``None``.

        Returns:
            An ``AROverlayResult`` containing the composite and metadata.

        Raises:
            ValueError: If inputs are invalid or detection fails and no
                corners are provided.
        """
        opts = options or AROverlayOptions()

        photo = self._validate_image(photo, "photo")
        handwriting = self._validate_image(handwriting, "handwriting")

        ph, pw = photo.shape[:2]
        hw, hh = handwriting.shape[1], handwriting.shape[0]

        # Step 1: Detect or use provided paper corners
        detection: Optional[PaperDetectionResult] = None
        if paper_corners is not None:
            corners = np.asarray(paper_corners, dtype=np.float64)
            if corners.shape != (4, 2):
                raise ValueError(
                    f"paper_corners must have shape (4, 2), got {corners.shape}"
                )
        else:
            detection = self._detector.detect(photo)
            if detection is None:
                raise ValueError(
                    "Paper detection failed. Provide paper_corners manually "
                    "or use a clearer photo."
                )
            corners = detection.corners

        # Step 2: Perspective transform - warp handwriting to paper shape
        target_size = (hw, hh)
        transformer = PerspectiveTransformer(target_size)
        transformer.compute(corners)
        warped_hw = transformer.warp_forward(handwriting)

        # Create paper mask in photo space
        paper_mask = transformer.warp_backward_mask((pw, ph))

        # Step 3: Resize warped handwriting to match photo dimensions
        warped_full = cv2.resize(warped_hw, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # Step 4: Extract paper region for lighting reference
        paper_region = self._extract_paper_region(photo, paper_mask)

        # Step 5: Match lighting
        adjusted = self._lighting.match_lighting(
            warped_full, paper_region, paper_mask
        )

        # Step 6: Apply shadow
        adjusted = self._lighting.apply_shadow(adjusted, paper_mask)

        # Step 7: Blend with texture effects
        composite = self._blender.blend_handwriting(
            photo, adjusted, paper_mask, ink_color=opts.ink_color
        )

        return AROverlayResult(
            composite=composite,
            paper_detection=detection,
            warped_handwriting=warped_full,
            paper_mask=paper_mask,
        )

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
        """Validate and convert an image to BGR uint8 format."""
        if image is None or image.size == 0:
            raise ValueError(f"'{name}' image is empty")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _extract_paper_region(
        image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Extract the paper region from the image using the mask.

        Returns a copy where non-paper pixels are filled with the mean
        paper color to avoid affecting lighting statistics.
        """
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return image.copy()

        mean_color = image[mask_bool].mean(axis=0).astype(np.uint8)
        region = np.full_like(image, mean_color)
        region[mask_bool] = image[mask_bool]
        return region


def overlay_on_paper(
    photo: np.ndarray,
    handwriting: np.ndarray,
    paper_corners: Optional[np.ndarray] = None,
    options: Optional[AROverlayOptions] = None,
) -> np.ndarray:
    """Convenience function to overlay handwriting onto a paper photo.

    This is the simplest entry point for the AR overlay pipeline.

    Parameters:
        photo: Paper photograph, BGR ``(H, W, 3)`` or grayscale ``(H, W)``.
        handwriting: Handwriting image to overlay, same format options.
        paper_corners: Optional manual corners ``(4, 2)``, or ``None``
            for automatic detection.
        options: Optional overlay configuration.

    Returns:
        The composited BGR image as a numpy array, same resolution as
        the input photo.

    Raises:
        ValueError: If paper detection fails and no corners are given.
    """
    engine = AREngine()
    result = engine.overlay(photo, handwriting, paper_corners, options)
    return result.composite


def detect_paper_edges(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
) -> Optional[np.ndarray]:
    """Detect paper edges in an image.

    A convenience wrapper around ``PaperDetector.detect()`` that returns
    just the corner array.

    Parameters:
        image: Input BGR image ``(H, W, 3)``.
        min_area_ratio: Minimum contour area as fraction of image area.

    Returns:
        Corner points ``(4, 2)`` as float64, or ``None`` if not found.
    """
    detector = PaperDetector(min_area_ratio=min_area_ratio)
    result = detector.detect(image)
    return result.corners if result is not None else None
