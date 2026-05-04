"""Tests for the dynamics simulation module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from handwrite.dynamics import DynamicsParams, apply_dynamics
from handwrite.dynamics.stroke_analyzer import (
    analyze_stroke_structure,
    get_pressure_profile,
)
from handwrite.dynamics.pressure_simulator import (
    pressure_width_map,
    simulate_pressure,
)
from handwrite.dynamics.ink_simulator import ink_density_map, simulate_ink
from handwrite.dynamics.speed_simulator import simulate_speed, speed_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(size: int = 64) -> np.ndarray:
    """Create a simple test character image with a horizontal stroke."""
    img = np.full((size, size), 255, dtype=np.uint8)
    # Draw a horizontal stroke in the middle
    y = size // 2
    x_start, x_end = size // 6, 5 * size // 6
    thickness = max(3, size // 16)
    for dy in range(-thickness, thickness + 1):
        row = y + dy
        if 0 <= row < size:
            img[row, x_start:x_end] = 0
    return img


def _make_test_image_cross(size: int = 64) -> np.ndarray:
    """Create a test image with a cross (two strokes)."""
    img = np.full((size, size), 255, dtype=np.uint8)
    mid = size // 2
    t = max(2, size // 20)
    # Horizontal stroke
    for dy in range(-t, t + 1):
        row = mid + dy
        if 0 <= row < size:
            img[row, size // 6 : 5 * size // 6] = 0
    # Vertical stroke
    for dx in range(-t, t + 1):
        col = mid + dx
        if 0 <= col < size:
            img[size // 6 : 5 * size // 6, col] = 0
    return img


def _make_blank_image(size: int = 64) -> np.ndarray:
    """Create a blank (all white) image."""
    return np.full((size, size), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# stroke_analyzer tests
# ---------------------------------------------------------------------------

class TestStrokeAnalyzer:
    def test_analyze_basic_stroke(self):
        img = _make_test_image(64)
        result = analyze_stroke_structure(img)

        assert "skeleton" in result
        assert "distance_map" in result
        assert "path" in result
        assert "stroke_zones" in result
        assert "stroke_length" in result
        assert result["stroke_length"] > 0
        assert result["skeleton"].shape == img.shape
        assert result["distance_map"].shape == img.shape
        assert result["stroke_zones"].shape == img.shape

    def test_analyze_blank_image(self):
        img = _make_blank_image(64)
        result = analyze_stroke_structure(img)
        assert result["stroke_length"] == 0
        assert len(result["path"]) == 0

    def test_stroke_zones_cover_ink(self):
        img = _make_test_image(64)
        result = analyze_stroke_structure(img)
        binary = img < 128
        zones = result["stroke_zones"]

        # All ink pixels should be assigned to some zone (1, 2, or 3)
        ink_with_zone = binary & (zones > 0)
        coverage = ink_with_zone.sum() / max(binary.sum(), 1)
        assert coverage > 0.5, f"Zone coverage too low: {coverage:.2%}"

    def test_pressure_profile_shape(self):
        path = [(i, 32) for i in range(10, 54)]
        dist = np.ones((64, 64), dtype=np.float64)
        profile = get_pressure_profile(path, dist, "natural")
        assert len(profile) == len(path)
        assert profile.min() >= 0.0
        assert profile.max() <= 1.0

    def test_pressure_profile_heavy_start(self):
        path = [(i, 32) for i in range(10, 54)]
        dist = np.ones((64, 64), dtype=np.float64)
        profile = get_pressure_profile(path, dist, "heavy_start")
        assert profile[0] > profile[-1], "Heavy start should begin stronger"

    def test_pressure_profile_uniform(self):
        path = [(i, 32) for i in range(10, 54)]
        dist = np.ones((64, 64), dtype=np.float64)
        profile = get_pressure_profile(path, dist, "uniform")
        assert np.allclose(profile, profile[0]), "Uniform profile should be constant"

    def test_pressure_profile_empty_path(self):
        dist = np.ones((64, 64), dtype=np.float64)
        profile = get_pressure_profile([], dist)
        assert len(profile) == 0


# ---------------------------------------------------------------------------
# pressure_simulator tests
# ---------------------------------------------------------------------------

class TestPressureSimulator:
    def test_simulate_pressure_output_shape(self):
        img = _make_test_image(64)
        result = simulate_pressure(img, pressure_strength=0.5)
        assert result.shape == img.shape

    def test_simulate_pressure_preserves_background(self):
        img = _make_test_image(64)
        result = simulate_pressure(img)
        # Background should remain white
        assert result[0, 0] == 255

    def test_simulate_pressure_blank_image(self):
        img = _make_blank_image(64)
        result = simulate_pressure(img)
        np.testing.assert_array_equal(result, img)

    def test_pressure_strength_affects_output(self):
        img = _make_test_image(64)
        r1 = simulate_pressure(img, pressure_strength=0.1)
        r2 = simulate_pressure(img, pressure_strength=0.9)
        # Different strength should produce different results
        assert not np.array_equal(r1, r2)

    def test_pressure_width_map_range(self):
        img = _make_test_image(64)
        wmap = pressure_width_map(img)
        assert wmap.shape == img.shape
        assert wmap.min() >= 0.5
        assert wmap.max() <= 1.5


# ---------------------------------------------------------------------------
# ink_simulator tests
# ---------------------------------------------------------------------------

class TestInkSimulator:
    def test_simulate_ink_output_shape(self):
        img = _make_test_image(64)
        result = simulate_ink(img, ink_density=0.6)
        assert result.shape == img.shape

    def test_simulate_ink_preserves_background(self):
        img = _make_test_image(64)
        result = simulate_ink(img)
        assert result[0, 0] == 255

    def test_simulate_ink_blank_image(self):
        img = _make_blank_image(64)
        result = simulate_ink(img)
        np.testing.assert_array_equal(result, img)

    def test_ink_density_affects_darkness(self):
        img = _make_test_image(64)
        light = simulate_ink(img, ink_density=0.2)
        dark = simulate_ink(img, ink_density=0.9)
        # Darker density should produce lower (darker) values in stroke area
        binary = img < 128
        if binary.any():
            assert dark[binary].mean() <= light[binary].mean()

    def test_ink_density_map_range(self):
        img = _make_test_image(64)
        dmap = ink_density_map(img, ink_density=0.7)
        assert dmap.shape == img.shape
        assert dmap.min() >= 0.0
        assert dmap.max() <= 1.0


# ---------------------------------------------------------------------------
# speed_simulator tests
# ---------------------------------------------------------------------------

class TestSpeedSimulator:
    def test_simulate_speed_output_shape(self):
        img = _make_test_image(64)
        result = simulate_speed(img, speed_variation=0.3)
        assert result.shape == img.shape

    def test_simulate_speed_blank_image(self):
        img = _make_blank_image(64)
        result = simulate_speed(img)
        assert result.shape == img.shape

    def test_speed_variation_affects_output(self):
        img = _make_test_image(64)
        r1 = simulate_speed(img, speed_variation=0.1)
        r2 = simulate_speed(img, speed_variation=0.8)
        assert not np.array_equal(r1, r2)

    def test_speed_profile_shape(self):
        path = [(i, 32) for i in range(10, 54)]
        profile = speed_profile(path, base_speed=0.5, speed_variation=0.3)
        assert len(profile) == len(path)
        assert profile.min() >= 0.0
        assert profile.max() <= 1.0

    def test_speed_profile_empty(self):
        profile = speed_profile([])
        assert len(profile) == 0


# ---------------------------------------------------------------------------
# dynamics_engine tests
# ---------------------------------------------------------------------------

class TestDynamicsEngine:
    def _make_pil_image(self, size: int = 64) -> Image.Image:
        return Image.fromarray(_make_test_image(size), mode="L")

    def test_apply_dynamics_returns_pil(self):
        img = self._make_pil_image(64)
        result = apply_dynamics(img)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_apply_dynamics_output_size(self):
        img = self._make_pil_image(64)
        params = DynamicsParams(output_size=(256, 256))
        result = apply_dynamics(img, params)
        assert result.size == (256, 256)

    def test_apply_dynamics_custom_size(self):
        img = self._make_pil_image(64)
        params = DynamicsParams(output_size=(128, 128))
        result = apply_dynamics(img, params)
        assert result.size == (128, 128)

    def test_apply_dynamics_disabled(self):
        img = self._make_pil_image(64)
        params = DynamicsParams(enabled=False)
        result = apply_dynamics(img, params)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_apply_dynamics_default_params(self):
        img = self._make_pil_image(64)
        result = apply_dynamics(img)
        assert result.size == (256, 256)

    def test_apply_dynamics_with_rgb_input(self):
        arr = _make_test_image(64)
        rgb = np.stack([arr, arr, arr], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
        result = apply_dynamics(img)
        assert result.mode == "L"
        assert result.size == (256, 256)

    def test_dynamics_params_frozen(self):
        params = DynamicsParams()
        with pytest.raises(AttributeError):
            params.pressure_strength = 0.8  # type: ignore[misc]

    def test_dynamics_params_custom_values(self):
        params = DynamicsParams(
            pressure_strength=0.8,
            pressure_curve="heavy_start",
            ink_density=0.9,
            ink_diffusion=4,
            speed_variation=0.5,
            base_speed=0.7,
            output_size=(512, 512),
        )
        assert params.pressure_strength == 0.8
        assert params.pressure_curve == "heavy_start"
        assert params.ink_density == 0.9
        assert params.ink_diffusion == 4
        assert params.speed_variation == 0.5
        assert params.base_speed == 0.7
        assert params.output_size == (512, 512)

    def test_apply_dynamics_produces_visible_strokes(self):
        """Ensure the output still has visible ink (not all white)."""
        img = self._make_pil_image(64)
        result = apply_dynamics(img)
        arr = np.array(result)
        assert (arr < 200).any(), "Output should contain visible ink strokes"

    def test_apply_dynamics_cross_image(self):
        """Test with a more complex image (cross shape)."""
        arr = _make_test_image_cross(64)
        img = Image.fromarray(arr, mode="L")
        result = apply_dynamics(img)
        assert result.mode == "L"
        assert result.size == (256, 256)
