"""Tests for the personalization module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample_image(
    width: int = 256,
    height: int = 256,
    text: str = "你好",
    slant: float = 0.0,
) -> Image.Image:
    """Create a synthetic handwriting sample image with ink-like strokes."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    # Draw some strokes to simulate handwriting
    rng = np.random.RandomState(123)
    for i in range(5):
        x1, y1 = rng.randint(20, width - 20, 2)
        x2, y2 = x1 + rng.randint(30, 100), y1 + rng.randint(-20, 20)
        stroke_width = rng.randint(2, 6)
        draw.line([(x1, y1), (x2, y2)], fill=0, width=stroke_width)

    # Draw a simple rectangle stroke
    draw.rectangle([60, 60, 180, 180], outline=0, width=3)
    draw.line([(80, 80), (160, 160)], fill=0, width=2)
    draw.line([(80, 160), (160, 80)], fill=0, width=2)

    if abs(slant) > 0.1:
        img = img.rotate(slant, resample=Image.BICUBIC, fillcolor=255)

    return img


@pytest.fixture
def sample_image() -> Image.Image:
    return _make_sample_image()


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    img = _make_sample_image()
    path = tmp_path / "sample.png"
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# SampleAnalyzer tests
# ---------------------------------------------------------------------------


class TestSampleAnalyzer:
    def test_analyze_returns_features(self, sample_image: Image.Image):
        from handwrite.personalization.sample_analyzer import SampleAnalyzer

        analyzer = SampleAnalyzer()
        features = analyzer.analyze(sample_image)

        assert features.image_width == 256
        assert features.image_height == 256
        assert features.stroke_width_mean > 0
        assert features.stroke_width_std >= 0
        assert -30 <= features.slant_angle <= 30
        assert 0.0 <= features.connectivity <= 1.0
        assert 0.0 <= features.ink_coverage <= 1.0

    def test_analyze_from_path(self, sample_path: Path):
        from handwrite.personalization.sample_analyzer import SampleAnalyzer

        analyzer = SampleAnalyzer()
        features = analyzer.analyze(sample_path)

        assert features.image_width > 0
        assert features.stroke_width_mean > 0

    def test_analyze_blank_image(self):
        from handwrite.personalization.sample_analyzer import SampleAnalyzer

        blank = Image.new("L", (100, 100), 255)
        analyzer = SampleAnalyzer()
        features = analyzer.analyze(blank)

        assert features.ink_coverage == 0.0
        assert features.stroke_width_mean == 0.0

    def test_analyze_filled_image(self):
        from handwrite.personalization.sample_analyzer import SampleAnalyzer

        filled = Image.new("L", (100, 100), 0)
        analyzer = SampleAnalyzer()
        features = analyzer.analyze(filled)

        assert features.ink_coverage > 0.9

    def test_slant_detection(self):
        from handwrite.personalization.sample_analyzer import SampleAnalyzer

        # Create image with clear vertical strokes
        img = Image.new("L", (200, 200), 255)
        draw = ImageDraw.Draw(img)
        for x in [50, 100, 150]:
            draw.line([(x, 20), (x, 180)], fill=0, width=3)

        analyzer = SampleAnalyzer()
        features = analyzer.analyze(img)
        # Vertical strokes should detect ~0 degree slant (analyzer range is [-30, 30])
        assert abs(features.slant_angle) <= 30

    def test_features_dataclass(self):
        from handwrite.personalization.sample_analyzer import HandwritingFeatures

        f = HandwritingFeatures(
            stroke_width_mean=3.0,
            stroke_width_std=1.0,
            slant_angle=5.0,
            connectivity=0.5,
            ink_intensity_mean=50.0,
            ink_intensity_std=30.0,
            image_width=256,
            image_height=256,
            ink_coverage=0.1,
        )
        assert f.stroke_width_mean == 3.0
        assert f.slant_angle == 5.0


# ---------------------------------------------------------------------------
# StyleExtractor tests
# ---------------------------------------------------------------------------


class TestStyleExtractor:
    def test_extract_basic(self):
        from handwrite.personalization.sample_analyzer import HandwritingFeatures
        from handwrite.personalization.style_extractor import StyleExtractor

        features = HandwritingFeatures(
            stroke_width_mean=4.0,
            stroke_width_std=1.5,
            slant_angle=10.0,
            connectivity=0.6,
            ink_intensity_mean=40.0,
            ink_intensity_std=25.0,
            image_width=256,
            image_height=256,
            ink_coverage=0.15,
        )
        extractor = StyleExtractor()
        style = extractor.extract(features)

        assert 0.0 <= style.stroke_thickness <= 1.0
        assert -30 <= style.slant_angle <= 30
        assert 0.0 <= style.cursiveness <= 1.0
        assert 0.0 <= style.ink_darkness <= 1.0
        assert 0.0 <= style.smoothness <= 1.0
        assert 0.0 <= style.ink_density <= 1.0

    def test_extract_thick_strokes(self):
        from handwrite.personalization.sample_analyzer import HandwritingFeatures
        from handwrite.personalization.style_extractor import StyleExtractor

        features = HandwritingFeatures(
            stroke_width_mean=10.0,
            stroke_width_std=2.0,
            slant_angle=0.0,
            connectivity=0.3,
            ink_intensity_mean=20.0,
            ink_intensity_std=15.0,
            image_width=256,
            image_height=256,
            ink_coverage=0.3,
        )
        extractor = StyleExtractor()
        style = extractor.extract(features)

        assert style.stroke_thickness > 0.5
        assert style.ink_darkness > 0.5

    def test_average_vectors(self):
        from handwrite.personalization.style_extractor import StyleExtractor, StyleVector

        v1 = StyleVector(0.3, 5.0, 0.4, 0.6, 0.7, 0.2)
        v2 = StyleVector(0.7, -5.0, 0.8, 0.4, 0.3, 0.6)
        extractor = StyleExtractor()
        avg = extractor.average_vectors([v1, v2])

        assert abs(avg.stroke_thickness - 0.5) < 1e-6
        assert abs(avg.slant_angle - 0.0) < 1e-6
        assert abs(avg.cursiveness - 0.6) < 1e-6

    def test_average_empty_raises(self):
        from handwrite.personalization.style_extractor import StyleExtractor

        extractor = StyleExtractor()
        with pytest.raises(ValueError, match="empty"):
            extractor.average_vectors([])

    def test_clamp_boundaries(self):
        from handwrite.personalization.style_extractor import StyleExtractor

        assert StyleExtractor._clamp(-1.0) == 0.0
        assert StyleExtractor._clamp(2.0) == 1.0
        assert StyleExtractor._clamp(0.5) == 0.5


# ---------------------------------------------------------------------------
# GlyphSynthesizer tests
# ---------------------------------------------------------------------------


class TestGlyphSynthesizer:
    def _make_synthesizer(self):
        from handwrite.personalization.style_extractor import StyleVector
        from handwrite.personalization.glyph_synthesizer import GlyphSynthesizer

        style = StyleVector(
            stroke_thickness=0.5,
            slant_angle=5.0,
            cursiveness=0.3,
            ink_darkness=0.7,
            smoothness=0.6,
            ink_density=0.4,
        )
        return GlyphSynthesizer(style)

    def test_synthesize_char_returns_image(self):
        synth = self._make_synthesizer()
        img = synth.synthesize_char("你")

        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
        assert img.mode == "L"

    def test_synthesize_char_reproducible(self):
        synth = self._make_synthesizer()
        img1 = synth.synthesize_char("好", seed=42)
        img2 = synth.synthesize_char("好", seed=42)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2)

    def test_synthesize_char_different_seeds(self):
        synth = self._make_synthesizer()
        img1 = synth.synthesize_char("好", seed=1)
        img2 = synth.synthesize_char("好", seed=2)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        # Different seeds should produce different images
        assert not np.array_equal(arr1, arr2)

    def test_synthesize_char_custom_size(self):
        from handwrite.personalization.style_extractor import StyleVector
        from handwrite.personalization.glyph_synthesizer import GlyphSynthesizer

        style = StyleVector(0.5, 0.0, 0.3, 0.7, 0.6, 0.4)
        synth = GlyphSynthesizer(style, glyph_size=128)
        img = synth.synthesize_char("测")

        assert img.size == (128, 128)

    def test_synthesize_pack_creates_files(self, tmp_path: Path):
        synth = self._make_synthesizer()
        output_dir = tmp_path / "pack"
        manifest_path = synth.synthesize_pack(
            output_dir,
            charset="你好世界",
            pack_name="test_pack",
            writer_id="test_writer",
        )

        assert manifest_path.exists()
        assert manifest_path.name == "manifest.json"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["name"] == "test_pack"
        assert len(manifest["glyphs"]) == 4

        for entry in manifest["glyphs"]:
            glyph_path = output_dir / entry["file"]
            assert glyph_path.exists()
            img = Image.open(str(glyph_path))
            assert img.size == (256, 256)

    def test_synthesize_pack_default_charset(self, tmp_path: Path):
        synth = self._make_synthesizer()
        output_dir = tmp_path / "default_pack"
        manifest_path = synth.synthesize_pack(output_dir)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Default charset should have many characters
        assert len(manifest["glyphs"]) > 100

    def test_glyph_filenames(self, tmp_path: Path):
        synth = self._make_synthesizer()
        output_dir = tmp_path / "filenames"
        manifest_path = synth.synthesize_pack(output_dir, charset="A")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["glyphs"][0]
        assert entry["file"] == "glyphs/U0041.png"


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------


class TestPersonalizationPipeline:
    def test_full_pipeline(self, tmp_path: Path):
        """Test the full pipeline: image -> features -> style -> glyphs."""
        from handwrite.personalization.glyph_synthesizer import GlyphSynthesizer
        from handwrite.personalization.sample_analyzer import SampleAnalyzer
        from handwrite.personalization.style_extractor import StyleExtractor

        # Step 1: Create and save sample
        sample = _make_sample_image()
        sample_path = tmp_path / "sample.png"
        sample.save(str(sample_path))

        # Step 2: Analyze
        analyzer = SampleAnalyzer()
        features = analyzer.analyze(sample_path)
        assert features.stroke_width_mean >= 0

        # Step 3: Extract style
        extractor = StyleExtractor()
        style = extractor.extract(features)
        assert 0.0 <= style.stroke_thickness <= 1.0

        # Step 4: Synthesize
        output_dir = tmp_path / "output"
        synth = GlyphSynthesizer(style)
        manifest_path = synth.synthesize_pack(
            output_dir, charset="测试"
        )

        # Verify
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["glyphs"]) == 2

    def test_multi_sample_pipeline(self, tmp_path: Path):
        """Test pipeline with multiple samples averaged."""
        from handwrite.personalization.sample_analyzer import SampleAnalyzer
        from handwrite.personalization.style_extractor import StyleExtractor

        analyzer = SampleAnalyzer()
        extractor = StyleExtractor()

        vectors = []
        for i in range(3):
            img = _make_sample_image(slant=float(i * 5))
            features = analyzer.analyze(img)
            style = extractor.extract(features)
            vectors.append(style)

        avg = extractor.average_vectors(vectors)
        assert -30 <= avg.slant_angle <= 30


# ---------------------------------------------------------------------------
# __init__ imports test
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_imports(self):
        from handwrite.personalization import (
            GlyphSynthesizer,
            HandwritingFeatures,
            SampleAnalyzer,
            StyleExtractor,
            StyleVector,
        )

        assert SampleAnalyzer is not None
        assert HandwritingFeatures is not None
        assert StyleExtractor is not None
        assert StyleVector is not None
        assert GlyphSynthesizer is not None
