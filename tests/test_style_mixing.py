"""Tests for the style_mixing module."""

from __future__ import annotations

import math

import pytest
from PIL import Image

from handwrite.style_mixing import (
    MixEngine,
    MixRecipe,
    StyleVector,
    TransferResult,
    bezier,
    cosine_similarity,
    describe_mixture,
    euclidean_distance,
    lerp,
    mix_multi,
    mix_styles,
    slerp,
    transfer_style,
    weighted_blend,
)


# ---------------------------------------------------------------------------
# StyleVector
# ---------------------------------------------------------------------------


class TestStyleVector:
    def test_default_construction(self) -> None:
        sv = StyleVector()
        assert sv.neatness == 0.7
        assert sv.connectivity == 0.3
        assert sv.slant_angle == 0.0
        assert sv.stroke_width == 1.0
        assert sv.ink_density == 1.0

    def test_custom_values(self) -> None:
        sv = StyleVector(neatness=0.5, connectivity=0.8, slant_angle=10.0,
                         stroke_width=1.1, ink_density=0.9)
        assert sv.neatness == 0.5
        assert sv.connectivity == 0.8
        assert sv.slant_angle == 10.0

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="neatness"):
            StyleVector(neatness=1.5)
        with pytest.raises(ValueError, match="slant_angle"):
            StyleVector(slant_angle=20.0)
        with pytest.raises(ValueError, match="stroke_width"):
            StyleVector(stroke_width=0.5)
        with pytest.raises(ValueError, match="ink_density"):
            StyleVector(ink_density=2.0)

    def test_clamped(self) -> None:
        sv = StyleVector.clamped(neatness=2.0, slant_angle=-30.0, stroke_width=0.1)
        assert sv.neatness == 1.0
        assert sv.slant_angle == -15.0
        assert sv.stroke_width == 0.8

    def test_to_dict(self) -> None:
        sv = StyleVector()
        d = sv.to_dict()
        assert set(d.keys()) == {"neatness", "connectivity", "slant_angle",
                                  "stroke_width", "ink_density"}
        assert d["neatness"] == 0.7

    def test_from_dict(self) -> None:
        d = {"neatness": 0.5, "connectivity": 0.4, "slant_angle": 3.0,
             "stroke_width": 0.95, "ink_density": 1.1, "extra_key": 42}
        sv = StyleVector.from_dict(d)
        assert sv.neatness == 0.5
        assert sv.connectivity == 0.4
        assert sv.stroke_width == 0.95

    def test_presets_exist(self) -> None:
        assert StyleVector.neat().neatness > 0.8
        assert StyleVector.cursive().connectivity > 0.7
        assert StyleVector.messy().neatness < 0.3
        sv = StyleVector.default()
        assert sv.neatness == 0.7

    def test_frozen(self) -> None:
        sv = StyleVector()
        with pytest.raises(AttributeError):
            sv.neatness = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Distance / similarity
# ---------------------------------------------------------------------------


class TestDistanceAndSimilarity:
    def test_identical_vectors_zero_distance(self) -> None:
        sv = StyleVector()
        assert euclidean_distance(sv, sv) == pytest.approx(0.0)

    def test_different_vectors_positive_distance(self) -> None:
        d = euclidean_distance(StyleVector.neat(), StyleVector.messy())
        assert d > 0

    def test_distance_symmetry(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_cosine_similarity_identical(self) -> None:
        sv = StyleVector()
        assert cosine_similarity(sv, sv) == pytest.approx(1.0)

    def test_cosine_similarity_range(self) -> None:
        sim = cosine_similarity(StyleVector.neat(), StyleVector.messy())
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Interpolation engine
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_lerp_at_zero(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = lerp(a, b, 0.0)
        assert result.neatness == pytest.approx(a.neatness)

    def test_lerp_at_one(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = lerp(a, b, 1.0)
        assert result.neatness == pytest.approx(b.neatness)

    def test_lerp_at_half(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = lerp(a, b, 0.5)
        expected = (a.neatness + b.neatness) / 2
        assert result.neatness == pytest.approx(expected)

    def test_slerp_at_half_is_not_linear(self) -> None:
        a = StyleVector(neatness=0.0)
        b = StyleVector(neatness=1.0)
        linear = lerp(a, b, 0.5)
        smooth = slerp(a, b, 0.5)
        # Smooth-step should differ from linear at t=0.5
        # linear = 0.5, smooth = 3*0.25 - 2*0.125 = 0.5 (same at midpoint)
        # They differ at other points
        smooth_quarter = slerp(a, b, 0.25)
        linear_quarter = lerp(a, b, 0.25)
        assert smooth_quarter.neatness != pytest.approx(linear_quarter.neatness, abs=0.01)

    def test_bezier_endpoints(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        assert bezier(a, b, 0.0).neatness == pytest.approx(a.neatness)
        assert bezier(a, b, 1.0).neatness == pytest.approx(b.neatness)

    def test_weighted_blend_two_styles(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = weighted_blend([a, b], [0.7, 0.3])
        expected = a.neatness * 0.7 + b.neatness * 0.3
        assert result.neatness == pytest.approx(expected, abs=0.01)

    def test_weighted_blend_three_styles(self) -> None:
        styles = [StyleVector.neat(), StyleVector.cursive(), StyleVector.messy()]
        weights = [0.5, 0.3, 0.2]
        result = weighted_blend(styles, weights)
        assert 0 <= result.neatness <= 1

    def test_weighted_blend_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            weighted_blend([StyleVector()], [0.5, 0.5])

    def test_weighted_blend_empty(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            weighted_blend([], [])

    def test_weighted_blend_zero_weights(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            weighted_blend([StyleVector()], [0.0])


# ---------------------------------------------------------------------------
# Style mixer
# ---------------------------------------------------------------------------


class TestStyleMixer:
    def test_mix_ratio_zero_returns_a(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = mix_styles(a, b, 0.0)
        assert result.neatness == pytest.approx(a.neatness)

    def test_mix_ratio_one_returns_b(self) -> None:
        a, b = StyleVector.neat(), StyleVector.messy()
        result = mix_styles(a, b, 1.0)
        assert result.neatness == pytest.approx(b.neatness)

    def test_mix_ratio_half(self) -> None:
        a = StyleVector(neatness=0.0)
        b = StyleVector(neatness=1.0)
        result = mix_styles(a, b, 0.5)
        assert result.neatness == pytest.approx(0.5)

    def test_mix_seventy_thirty(self) -> None:
        """The canonical 70% neat + 30% messy use case."""
        neat = StyleVector.neat()
        messy = StyleVector.messy()
        # 70% neat means ratio=0.3 (30% of messy)
        result = mix_styles(neat, messy, 0.3)
        expected_neatness = neat.neatness * 0.7 + messy.neatness * 0.3
        assert result.neatness == pytest.approx(expected_neatness, abs=0.01)

    def test_mix_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            mix_styles(StyleVector(), StyleVector(), 1.5)

    def test_mix_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown mixing method"):
            mix_styles(StyleVector(), StyleVector(), 0.5, method="invalid")

    def test_mix_smooth(self) -> None:
        result = mix_styles(StyleVector.neat(), StyleVector.messy(), 0.5, method="smooth")
        assert isinstance(result, StyleVector)

    def test_mix_bezier(self) -> None:
        result = mix_styles(StyleVector.neat(), StyleVector.messy(), 0.5, method="bezier")
        assert isinstance(result, StyleVector)

    def test_mix_multi(self) -> None:
        styles = [StyleVector.neat(), StyleVector.cursive(), StyleVector.messy()]
        weights = [0.5, 0.3, 0.2]
        result = mix_multi(styles, weights)
        assert isinstance(result, StyleVector)

    def test_describe_mixture(self) -> None:
        info = describe_mixture(StyleVector.neat(), StyleVector.messy(), 0.3)
        assert "ratio_b" in info
        assert info["ratio_b"] == pytest.approx(0.3)
        assert "neatness" in info


# ---------------------------------------------------------------------------
# Style transfer (image-based)
# ---------------------------------------------------------------------------


class TestStyleTransfer:
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Create a simple test image with some dark strokes on white background."""
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        # Draw some "strokes"
        for x in range(20, 80):
            img.putpixel((x, 50), (0, 0, 0, 255))
        for y in range(30, 70):
            img.putpixel((50, y), (0, 0, 0, 255))
        return img

    def test_transfer_identity(self, sample_image: Image.Image) -> None:
        """Transferring to the same style should be a no-op."""
        sv = StyleVector()
        result = transfer_style(sample_image, sv, source_style=sv)
        assert result.image.size == sample_image.size
        assert result.operations_applied == []

    def test_transfer_returns_result(self, sample_image: Image.Image) -> None:
        result = transfer_style(sample_image, StyleVector.messy())
        assert isinstance(result, TransferResult)
        assert result.image is not None

    def test_transfer_slant(self, sample_image: Image.Image) -> None:
        target = StyleVector(slant_angle=10.0)
        source = StyleVector(slant_angle=0.0)
        result = transfer_style(sample_image, target, source_style=source)
        assert any("shear" in op for op in result.operations_applied)

    def test_transfer_ink_density(self, sample_image: Image.Image) -> None:
        target = StyleVector(ink_density=1.3)
        source = StyleVector(ink_density=0.7)
        result = transfer_style(sample_image, target, source_style=source)
        assert any("ink_density" in op for op in result.operations_applied)

    def test_transfer_stroke_width(self, sample_image: Image.Image) -> None:
        target = StyleVector(stroke_width=1.2)
        source = StyleVector(stroke_width=0.8)
        result = transfer_style(sample_image, target, source_style=source)
        assert any("stroke_width" in op for op in result.operations_applied)

    def test_transfer_default_source(self, sample_image: Image.Image) -> None:
        result = transfer_style(sample_image, StyleVector(slant_angle=8.0))
        assert result.source_style == StyleVector.default()

    def test_transfer_neatness(self, sample_image: Image.Image) -> None:
        target = StyleVector(neatness=0.1)
        source = StyleVector(neatness=0.9)
        result = transfer_style(sample_image, target, source_style=source)
        assert any("neatness" in op for op in result.operations_applied)


# ---------------------------------------------------------------------------
# MixEngine
# ---------------------------------------------------------------------------


class TestMixEngine:
    def test_blend(self) -> None:
        engine = MixEngine()
        result = engine.blend(StyleVector.neat(), StyleVector.messy(), 0.3)
        assert isinstance(result, StyleVector)

    def test_blend_multi(self) -> None:
        engine = MixEngine()
        styles = [StyleVector.neat(), StyleVector.cursive(), StyleVector.messy()]
        weights = [0.5, 0.3, 0.2]
        result = engine.blend_multi(styles, weights)
        assert isinstance(result, StyleVector)

    def test_transfer(self) -> None:
        engine = MixEngine()
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        result = engine.transfer(img, StyleVector(slant_angle=5.0))
        assert isinstance(result, TransferResult)

    def test_apply_recipe(self) -> None:
        engine = MixEngine()
        recipe = MixRecipe(
            name="test",
            styles=[StyleVector.neat(), StyleVector.messy()],
            weights=[0.7, 0.3],
        )
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        result = engine.apply(img, recipe)
        assert isinstance(result, TransferResult)

    def test_preset_recipes(self) -> None:
        engine = MixEngine()
        for factory in [
            engine.preset_neat_with_touch_of_cursive,
            engine.preset_balanced_mix,
            engine.preset_casual_cursive,
        ]:
            recipe = factory()
            assert isinstance(recipe, MixRecipe)
            assert isinstance(recipe.blended, StyleVector)

    def test_recipe_blended_property(self) -> None:
        recipe = MixRecipe(
            name="test",
            styles=[StyleVector.neat(), StyleVector.messy()],
            weights=[0.7, 0.3],
        )
        blended = recipe.blended
        expected = StyleVector.neat().neatness * 0.7 + StyleVector.messy().neatness * 0.3
        assert blended.neatness == pytest.approx(expected, abs=0.01)

    def test_recipe_single_style(self) -> None:
        recipe = MixRecipe(
            name="single",
            styles=[StyleVector.cursive()],
            weights=[1.0],
        )
        assert recipe.blended == StyleVector.cursive()


# ---------------------------------------------------------------------------
# Integration: 70% neat + 30% messy end-to-end
# ---------------------------------------------------------------------------


class TestSeventyThirtyIntegration:
    """End-to-end test for the canonical 70% neat + 30% messy workflow."""

    def test_blend_and_transfer(self) -> None:
        neat = StyleVector.neat()
        messy = StyleVector.messy()

        # Blend: 70% neat, 30% messy
        mixed = mix_styles(neat, messy, ratio=0.3)
        assert 0.0 <= mixed.neatness <= 1.0

        # Verify the mixed vector is closer to neat than messy
        d_to_neat = euclidean_distance(mixed, neat)
        d_to_messy = euclidean_distance(mixed, messy)
        assert d_to_neat < d_to_messy

        # Apply to image
        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        result = transfer_style(img, mixed)
        assert isinstance(result, TransferResult)
        assert result.image.mode == "RGBA"

    def test_engine_recipe_workflow(self) -> None:
        engine = MixEngine()
        recipe = MixRecipe(
            name="70_neat_30_messy",
            styles=[StyleVector.neat(), StyleVector.messy()],
            weights=[0.7, 0.3],
        )
        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        result = engine.apply(img, recipe)
        assert result.target_style.neatness == pytest.approx(
            recipe.blended.neatness, abs=0.01
        )
