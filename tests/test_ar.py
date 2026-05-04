"""Tests for the AR handwriting overlay module."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from handwrite.ar import (
    AREngine,
    AROverlayOptions,
    AROverlayResult,
    LightingAdjuster,
    PaperDetectionResult,
    PaperDetector,
    PerspectiveTransformer,
    TextureBlender,
    detect_paper_edges,
    overlay_on_paper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_image(
    width: int = 640,
    height: int = 480,
    paper_color: Tuple = (240, 240, 235),
    bg_color: Tuple = (60, 60, 60),
) -> np.ndarray:
    """Create a synthetic image with a white paper on a dark background.

    The paper is a 80% centered rectangle with a slight border.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    cv2.rectangle(
        img,
        (margin_x, margin_y),
        (width - margin_x, height - margin_y),
        paper_color,
        -1,
    )
    return img


def _make_handwriting_image(width: int, height: int) -> np.ndarray:
    """Create a synthetic handwriting image with dark strokes on white."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    # Draw some "handwriting" strokes
    cv2.putText(
        img, "Hello", (int(width * 0.1), int(height * 0.4)),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 3,
    )
    cv2.line(
        img,
        (int(width * 0.1), int(height * 0.6)),
        (int(width * 0.8), int(height * 0.6)),
        (10, 10, 10),
        2,
    )
    return img


def _make_paper_corners(width: int, height: int) -> np.ndarray:
    """Get the paper corners for the synthetic image."""
    mx = int(width * 0.1)
    my = int(height * 0.1)
    return np.array(
        [[mx, my], [width - mx, my], [width - mx, height - my], [mx, height - my]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# PaperDetector tests
# ---------------------------------------------------------------------------


class TestPaperDetector:
    def test_detect_finds_paper_in_synthetic_image(self) -> None:
        img = _make_paper_image(640, 480)
        detector = PaperDetector(min_area_ratio=0.01)
        result = detector.detect(img)

        assert result is not None
        assert isinstance(result, PaperDetectionResult)
        assert result.corners.shape == (4, 2)
        assert 0.0 <= result.confidence <= 1.0
        assert result.mask.shape == (480, 640)

    def test_detect_returns_none_for_blank_image(self) -> None:
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        detector = PaperDetector(min_area_ratio=0.01)
        result = detector.detect(img)
        assert result is None

    def test_detect_raises_on_empty_input(self) -> None:
        detector = PaperDetector()
        with pytest.raises(ValueError, match="empty"):
            detector.detect(np.array([]))

    def test_detect_raises_on_wrong_channels(self) -> None:
        detector = PaperDetector()
        with pytest.raises(ValueError, match="BGR"):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))

    def test_corners_are_ordered_tl_tr_br_bl(self) -> None:
        img = _make_paper_image(640, 480)
        detector = PaperDetector(min_area_ratio=0.01)
        result = detector.detect(img)
        assert result is not None

        corners = result.corners
        # TL should have smallest coordinates
        assert corners[0][0] < corners[1][0]  # TL.x < TR.x
        assert corners[0][1] < corners[3][1]  # TL.y < BL.y

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            PaperDetector(min_area_ratio=0.0)
        with pytest.raises(ValueError):
            PaperDetector(canny_low=-1)
        with pytest.raises(ValueError):
            PaperDetector(canny_low=100, canny_high=50)
        with pytest.raises(ValueError):
            PaperDetector(morph_kernel_size=0)


# ---------------------------------------------------------------------------
# PerspectiveTransformer tests
# ---------------------------------------------------------------------------


class TestPerspectiveTransformer:
    def test_warp_forward_produces_correct_size(self) -> None:
        src_corners = np.array(
            [[50, 50], [590, 50], [590, 430], [50, 430]], dtype=np.float32
        )
        src_img = np.zeros((480, 640, 3), dtype=np.uint8)

        transformer = PerspectiveTransformer((200, 300))
        transformer.compute(src_corners)
        result = transformer.warp_forward(src_img)

        assert result.shape == (300, 200, 3)

    def test_warp_backward_maps_to_output_size(self) -> None:
        src_corners = np.array(
            [[100, 50], [540, 50], [540, 430], [100, 430]], dtype=np.float32
        )
        rect_img = np.full((300, 200, 3), 200, dtype=np.uint8)

        transformer = PerspectiveTransformer((200, 300))
        transformer.compute(src_corners)
        result = transformer.warp_backward(rect_img, (640, 480))

        assert result.shape == (480, 640, 3)

    def test_warp_backward_mask_has_correct_shape(self) -> None:
        src_corners = np.array(
            [[100, 50], [540, 50], [540, 430], [100, 430]], dtype=np.float32
        )
        transformer = PerspectiveTransformer((200, 300))
        transformer.compute(src_corners)
        mask = transformer.warp_backward_mask((640, 480))

        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8

    def test_raises_before_compute(self) -> None:
        t = PerspectiveTransformer((100, 100))
        with pytest.raises(RuntimeError, match="compute"):
            t.warp_forward(np.zeros((100, 100, 3), dtype=np.uint8))
        with pytest.raises(RuntimeError, match="compute"):
            t.warp_backward(np.zeros((100, 100, 3), dtype=np.uint8), (200, 200))

    def test_invalid_target_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            PerspectiveTransformer((0, 100))
        with pytest.raises(ValueError):
            PerspectiveTransformer((100, -1))

    def test_invalid_corners_shape_rejected(self) -> None:
        t = PerspectiveTransformer((100, 100))
        with pytest.raises(ValueError, match="shape"):
            t.compute(np.zeros((3, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# LightingAdjuster tests
# ---------------------------------------------------------------------------


class TestLightingAdjuster:
    def test_match_lighting_preserves_shape(self) -> None:
        overlay = np.full((100, 200, 3), 128, dtype=np.uint8)
        target = np.full((100, 200, 3), 200, dtype=np.uint8)

        adjuster = LightingAdjuster()
        result = adjuster.match_lighting(overlay, target)

        assert result.shape == overlay.shape
        assert result.dtype == np.uint8

    def test_match_lighting_with_mask(self) -> None:
        overlay = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (180, 80), 255, -1)

        adjuster = LightingAdjuster()
        result = adjuster.match_lighting(overlay, target, mask)

        assert result.shape == overlay.shape

    def test_match_lighting_raises_on_shape_mismatch(self) -> None:
        overlay = np.zeros((100, 200, 3), dtype=np.uint8)
        target = np.zeros((100, 300, 3), dtype=np.uint8)

        adjuster = LightingAdjuster()
        with pytest.raises(ValueError, match="Shape mismatch"):
            adjuster.match_lighting(overlay, target)

    def test_apply_shadow_returns_same_shape(self) -> None:
        img = np.full((100, 200, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (190, 90), 255, -1)

        adjuster = LightingAdjuster(shadow_strength=0.3)
        result = adjuster.apply_shadow(img, mask)

        assert result.shape == img.shape

    def test_zero_shadow_returns_original(self) -> None:
        img = np.full((100, 200, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (190, 90), 255, -1)

        adjuster = LightingAdjuster(shadow_strength=0.0)
        result = adjuster.apply_shadow(img, mask)

        np.testing.assert_array_equal(result, img)

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            LightingAdjuster(brightness_weight=-0.1)
        with pytest.raises(ValueError):
            LightingAdjuster(contrast_weight=1.5)
        with pytest.raises(ValueError):
            LightingAdjuster(shadow_strength=-0.1)
        with pytest.raises(ValueError):
            LightingAdjuster(shadow_blur_size=4)
        with pytest.raises(ValueError):
            LightingAdjuster(shadow_blur_size=0)


# ---------------------------------------------------------------------------
# TextureBlender tests
# ---------------------------------------------------------------------------


class TestTextureBlender:
    def test_alpha_blend_full_mask_returns_foreground(self) -> None:
        bg = np.zeros((100, 200, 3), dtype=np.uint8)
        fg = np.full((100, 200, 3), 255, dtype=np.uint8)
        mask = np.full((100, 200), 255, dtype=np.uint8)

        blender = TextureBlender()
        result = blender.alpha_blend(bg, fg, mask)

        np.testing.assert_array_equal(result, fg)

    def test_alpha_blend_zero_mask_returns_background(self) -> None:
        bg = np.full((100, 200, 3), 100, dtype=np.uint8)
        fg = np.full((100, 200, 3), 255, dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)

        blender = TextureBlender()
        result = blender.alpha_blend(bg, fg, mask)

        np.testing.assert_array_equal(result, bg)

    def test_blend_handwriting_preserves_shape(self) -> None:
        paper = np.full((200, 300, 3), 230, dtype=np.uint8)
        hw = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.putText(hw, "test", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        mask = np.full((200, 300), 255, dtype=np.uint8)

        blender = TextureBlender()
        result = blender.blend_handwriting(paper, hw, mask)

        assert result.shape == paper.shape
        assert result.dtype == np.uint8

    def test_blend_handwriting_with_custom_ink_color(self) -> None:
        paper = np.full((200, 300, 3), 230, dtype=np.uint8)
        hw = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.putText(hw, "test", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        mask = np.full((200, 300), 255, dtype=np.uint8)

        blender = TextureBlender()
        result = blender.blend_handwriting(paper, hw, mask, ink_color=(0, 0, 200))

        assert result.shape == paper.shape

    def test_blend_raises_on_shape_mismatch(self) -> None:
        paper = np.full((200, 300, 3), 230, dtype=np.uint8)
        hw = np.full((100, 300, 3), 255, dtype=np.uint8)
        mask = np.full((200, 300), 255, dtype=np.uint8)

        blender = TextureBlender()
        with pytest.raises(ValueError, match="Shape mismatch"):
            blender.blend_handwriting(paper, hw, mask)

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            TextureBlender(ink_opacity=-0.1)
        with pytest.raises(ValueError):
            TextureBlender(paper_texture_weight=1.5)
        with pytest.raises(ValueError):
            TextureBlender(ink_penetration_depth=-0.1)
        with pytest.raises(ValueError):
            TextureBlender(feather_radius=-1)


# ---------------------------------------------------------------------------
# AREngine tests
# ---------------------------------------------------------------------------


class TestAREngine:
    def test_overlay_with_manual_corners(self) -> None:
        photo = _make_paper_image(640, 480)
        hw = _make_handwriting_image(440, 320)
        corners = _make_paper_corners(640, 480)

        engine = AREngine()
        result = engine.overlay(photo, hw, paper_corners=corners)

        assert isinstance(result, AROverlayResult)
        assert result.composite.shape == photo.shape
        assert result.composite.dtype == np.uint8

    def test_overlay_auto_detection(self) -> None:
        photo = _make_paper_image(640, 480)
        hw = _make_handwriting_image(440, 320)

        engine = AREngine()
        result = engine.overlay(photo, hw)

        assert result.paper_detection is not None
        assert result.composite.shape == photo.shape

    def test_overlay_raises_when_detection_fails_and_no_corners(self) -> None:
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        hw = _make_handwriting_image(200, 100)

        engine = AREngine()
        with pytest.raises(ValueError, match="detection failed"):
            engine.overlay(blank, hw)

    def test_overlay_with_custom_options(self) -> None:
        photo = _make_paper_image(640, 480)
        hw = _make_handwriting_image(440, 320)
        corners = _make_paper_corners(640, 480)

        opts = AROverlayOptions(
            ink_color=(0, 0, 180),
            brightness_weight=0.8,
            shadow_strength=0.0,
        )
        engine = AREngine()
        result = engine.overlay(photo, hw, paper_corners=corners, options=opts)

        assert result.composite.shape == photo.shape

    def test_overlay_accepts_grayscale_input(self) -> None:
        photo_gray = cv2.cvtColor(_make_paper_image(640, 480), cv2.COLOR_BGR2GRAY)
        hw = _make_handwriting_image(440, 320)
        corners = _make_paper_corners(640, 480)

        engine = AREngine()
        result = engine.overlay(photo_gray, hw, paper_corners=corners)

        assert result.composite.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_overlay_on_paper_returns_ndarray(self) -> None:
        photo = _make_paper_image(640, 480)
        hw = _make_handwriting_image(440, 320)
        corners = _make_paper_corners(640, 480)

        result = overlay_on_paper(photo, hw, paper_corners=corners)

        assert isinstance(result, np.ndarray)
        assert result.shape == photo.shape

    def test_detect_paper_edges_returns_corners(self) -> None:
        img = _make_paper_image(640, 480)
        corners = detect_paper_edges(img, min_area_ratio=0.01)

        assert corners is not None
        assert corners.shape == (4, 2)

    def test_detect_paper_edges_returns_none_for_blank(self) -> None:
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        corners = detect_paper_edges(img)
        assert corners is None


# ---------------------------------------------------------------------------
# Round-trip test: overlay then check output resolution preserved
# ---------------------------------------------------------------------------


class TestResolutionPreservation:
    def test_output_matches_photo_resolution(self) -> None:
        """The composited image must have the same resolution as the input photo."""
        for w, h in [(640, 480), (1920, 1080), (300, 400)]:
            photo = _make_paper_image(w, h)
            hw = _make_handwriting_image(w - 100, h - 100)
            corners = _make_paper_corners(w, h)

            result = overlay_on_paper(photo, hw, paper_corners=corners)
            assert result.shape == (h, w, 3), (
                f"Expected ({h}, {w}, 3), got {result.shape}"
            )
