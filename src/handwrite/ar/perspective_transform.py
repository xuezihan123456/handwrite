"""Four-point perspective transformation.

Warps a quadrilateral region in a source image to a rectangular output,
and provides the inverse mapping for overlay operations.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    """Performs four-point perspective warping.

    Given four source corners and a target size, computes the homography
    matrix and can warp images forward (quad -> rect) or backward
    (rect -> quad) for overlay compositing.

    Parameters:
        target_size: ``(width, height)`` of the output rectangle.
        flags: OpenCV interpolation flag for warping.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        flags: int = cv2.INTER_LINEAR,
    ) -> None:
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("target_size dimensions must be positive")

        self._target_size = target_size
        self._flags = flags
        self._forward_matrix: np.ndarray | None = None
        self._inverse_matrix: np.ndarray | None = None

    @property
    def target_size(self) -> Tuple[int, int]:
        """The target rectangle size ``(width, height)``."""
        return self._target_size

    @property
    def forward_matrix(self) -> np.ndarray | None:
        """The forward (quad-to-rect) homography matrix, or ``None``."""
        return self._forward_matrix

    @property
    def inverse_matrix(self) -> np.ndarray | None:
        """The inverse (rect-to-quad) homography matrix, or ``None``."""
        return self._inverse_matrix

    def compute(self, src_corners: np.ndarray) -> None:
        """Compute the perspective transformation matrices.

        Parameters:
            src_corners: Four source corner points, shape ``(4, 2)``,
                ordered as top-left, top-right, bottom-right, bottom-left.

        Raises:
            ValueError: If ``src_corners`` does not have shape ``(4, 2)``.
        """
        src = np.asarray(src_corners, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError(
                f"src_corners must have shape (4, 2), got {src.shape}"
            )

        w, h = self._target_size
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )

        self._forward_matrix = cv2.getPerspectiveTransform(src, dst)
        self._inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    def warp_forward(self, image: np.ndarray) -> np.ndarray:
        """Warp the source image from quad to rectangle.

        Parameters:
            image: Source image, shape ``(H, W, C)`` or ``(H, W)``.

        Returns:
            Warped image of shape ``(target_h, target_w, C)``.

        Raises:
            RuntimeError: If ``compute()`` has not been called.
        """
        if self._forward_matrix is None:
            raise RuntimeError("Call compute() before warp_forward()")
        return cv2.warpPerspective(
            image,
            self._forward_matrix,
            self._target_size,
            flags=self._flags,
        )

    def warp_backward(self, image: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Warp a rectangular image back onto the original quad region.

        Parameters:
            image: Rectangular source image (the content to overlay).
            output_size: ``(width, height)`` of the output canvas
                (typically the original photo size).

        Returns:
            Warped image on the output canvas.

        Raises:
            RuntimeError: If ``compute()`` has not been called.
        """
        if self._inverse_matrix is None:
            raise RuntimeError("Call compute() before warp_backward()")
        return cv2.warpPerspective(
            image,
            self._inverse_matrix,
            output_size,
            flags=self._flags,
        )

    def warp_backward_mask(self, output_size: Tuple[int, int]) -> np.ndarray:
        """Generate a binary mask of the warped region on the output canvas.

        Parameters:
            output_size: ``(width, height)`` of the output canvas.

        Returns:
            Binary mask of shape ``(height, width)``, dtype ``uint8``,
            with 255 inside the paper region and 0 elsewhere.
        """
        w, h = self._target_size
        white = np.full((h, w), 255, dtype=np.uint8)
        return self.warp_backward(white, output_size)
