"""Tests for the OCR-based handwriting style extraction module."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from handwrite.ocr_style.image_preprocessor import ImagePreprocessor, PreprocessResult
from handwrite.ocr_style.character_segmenter import CharacterSegmenter, CharBox
from handwrite.ocr_style.style_extractor import StyleExtractor, StyleFeatures
from handwrite.ocr_style.prototype_generator import PrototypeGenerator, GlyphEntry
from handwrite.ocr_style import (
    ImagePreprocessor as IP2,
    CharacterSegmenter as CS2,
    StyleExtractor as SE2,
    PrototypeGenerator as PG2,
)

# ======================================================================
# Test fixtures: synthetic test images
# ======================================================================


def _make_synthetic_scan(
    text: str = "你好",
    char_size: int = 80,
    gap: int = 20,
    padding: int = 40,
) -> np.ndarray:
    """Create a synthetic binarized image with white rectangles as characters.

    Returns a white-on-black image simulating binarized handwriting.
    """
    n = len(text)
    w = padding * 2 + char_size * n + gap * (n - 1)
    h = padding * 2 + char_size
    img = np.zeros((h, w), dtype=np.uint8)

    for i in range(n):
        x = padding + i * (char_size + gap)
        y = padding
        # Draw a filled rectangle as a fake character
        cv2.rectangle(img, (x, y), (x + char_size - 1, y + char_size - 1), 255, -1)
        # Add some inner pattern to simulate strokes
        cv2.rectangle(
            img,
            (x + char_size // 4, y + char_size // 4),
            (x + 3 * char_size // 4, y + 3 * char_size // 4),
            0,
            -1,
        )

    return img


def _make_scan_with_chars(n: int = 5, char_size: int = 60, gap: int = 25) -> np.ndarray:
    """Create a synthetic image with n distinct character-like regions."""
    padding = 30
    w = padding * 2 + char_size * n + gap * (n - 1)
    h = padding * 2 + char_size
    img = np.zeros((h, w), dtype=np.uint8)

    for i in range(n):
        x = padding + i * (char_size + gap)
        y = padding
        # Each character is a slightly different shape
        inset = 5 + (i % 3) * 5
        cv2.rectangle(img, (x, y), (x + char_size, y + char_size), 255, -1)
        cv2.rectangle(
            img,
            (x + inset, y + inset),
            (x + char_size - inset, y + char_size - inset),
            0,
            -1,
        )
        # Add cross strokes
        cx, cy = x + char_size // 2, y + char_size // 2
        cv2.line(img, (x + 5, cy), (x + char_size - 5, cy), 255, 2)
        cv2.line(img, (cx, y + 5), (cx, y + char_size - 5), 255, 2)

    return img


def _make_color_scan_image(tmp_path: Path) -> Path:
    """Create a synthetic color scan image (simulating a photographed page)."""
    h, w = 400, 600
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # light gray "paper"

    # Draw some dark "text" strokes
    cv2.rectangle(img, (50, 50), (130, 130), (30, 30, 30), -1)
    cv2.rectangle(img, (160, 50), (240, 130), (30, 30, 30), -1)
    cv2.rectangle(img, (280, 50), (360, 130), (30, 30, 30), -1)

    # Add some line-like features for skew detection
    cv2.line(img, (30, 200), (570, 200), (40, 40, 40), 1)
    cv2.line(img, (30, 250), (570, 250), (40, 40, 40), 1)

    path = tmp_path / "test_scan.png"
    cv2.imwrite(str(path), img)
    return path


# ======================================================================
# ImagePreprocessor tests
# ======================================================================


class TestImagePreprocessor:
    def test_preprocess_returns_result(self, tmp_path: Path) -> None:
        scan_path = _make_color_scan_image(tmp_path)
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(scan_path)

        assert isinstance(result, PreprocessResult)
        assert result.image.ndim == 2
        assert result.original_shape == (400, 600)

    def test_preprocess_binary_is_0_or_255(self, tmp_path: Path) -> None:
        scan_path = _make_color_scan_image(tmp_path)
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(scan_path)

        unique = set(np.unique(result.image))
        assert unique.issubset({0, 255})

    def test_preprocess_foreground_is_white(self, tmp_path: Path) -> None:
        """Ink pixels should be 255 (white)."""
        scan_path = _make_color_scan_image(tmp_path)
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(scan_path)

        # The ink regions we drew should be white (255)
        # Check that there are white pixels (ink exists)
        assert np.any(result.image == 255)

    def test_preprocess_file_not_found(self, tmp_path: Path) -> None:
        preprocessor = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess(tmp_path / "nonexistent.png")

    def test_skew_angle_is_finite(self, tmp_path: Path) -> None:
        scan_path = _make_color_scan_image(tmp_path)
        result = ImagePreprocessor().preprocess(scan_path)
        assert np.isfinite(result.skew_angle)

    def test_binarize_inverts_when_background_is_bright(self) -> None:
        """When input is mostly bright (paper-like), ink should end up white."""
        preprocessor = ImagePreprocessor()
        # Create a mostly-white image (paper with dark ink)
        img = np.full((100, 100), 230, dtype=np.uint8)
        img[30:70, 30:70] = 20  # dark ink region
        binary = preprocessor._binarize(img)
        # The ink region should be white (255)
        assert binary[50, 50] == 255

    def test_perspective_correction_no_paper_returns_original(self) -> None:
        """When no paper contour is found, should return original."""
        preprocessor = ImagePreprocessor()
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result, corrected = preprocessor._perspective_correction(gray)
        assert corrected is False
        np.testing.assert_array_equal(result, gray)

    def test_denoise_uses_median_filter(self) -> None:
        """Median filter should remove isolated noise pixels."""
        preprocessor = ImagePreprocessor(median_kernel=3)
        img = np.zeros((50, 50), dtype=np.uint8)
        # Add salt noise
        img[10, 10] = 255
        img[10, 11] = 255
        img[11, 10] = 255
        # An isolated pixel should be removed
        img[25, 25] = 255
        denoised = preprocessor._denoise(img)
        # The isolated pixel at (25,25) should be gone
        assert denoised[25, 25] == 0


# ======================================================================
# CharacterSegmenter tests
# ======================================================================


class TestCharacterSegmenter:
    def test_segment_finds_characters(self) -> None:
        img = _make_scan_with_chars(5)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)
        assert len(chars) >= 1

    def test_segment_returns_char_boxes_sorted_left_to_right(self) -> None:
        img = _make_scan_with_chars(4)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        for i in range(len(chars) - 1):
            assert chars[i].x <= chars[i + 1].x

    def test_segment_char_box_has_image(self) -> None:
        img = _make_scan_with_chars(3)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        for cb in chars:
            assert isinstance(cb, CharBox)
            assert cb.image.ndim == 2
            assert cb.w > 0
            assert cb.h > 0
            assert cb.image.shape == (cb.h, cb.w)

    def test_segment_empty_image(self) -> None:
        img = np.zeros((100, 100), dtype=np.uint8)
        segmenter = CharacterSegmenter()
        chars = segmenter.segment(img)
        assert chars == []

    def test_segment_filters_small_noise(self) -> None:
        img = np.zeros((200, 200), dtype=np.uint8)
        # Tiny noise dot
        img[10, 10] = 255
        # Large character
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

        segmenter = CharacterSegmenter(min_char_area=200, min_char_height=20)
        chars = segmenter.segment(img)
        # Should only find the large rectangle
        assert len(chars) >= 1
        for cb in chars:
            assert cb.area >= 200

    def test_sticky_region_splitting(self) -> None:
        """A very wide region should be split into sub-regions."""
        img = np.zeros((100, 400), dtype=np.uint8)
        # Wide sticky region with vertical gaps
        cv2.rectangle(img, (10, 10), (90, 90), 255, -1)
        cv2.rectangle(img, (100, 10), (180, 90), 255, -1)
        cv2.rectangle(img, (200, 10), (280, 90), 255, -1)
        # Connect them with a thin line to make one connected component
        cv2.line(img, (90, 50), (100, 50), 255, 1)
        cv2.line(img, (180, 50), (200, 50), 255, 1)

        segmenter = CharacterSegmenter(
            min_char_area=100,
            sticky_split_threshold=1.5,
        )
        chars = segmenter.segment(img)
        # Should split the wide region
        assert len(chars) >= 2

    def test_merge_nearby_merges_close_boxes(self) -> None:
        """Two boxes very close together should be merged."""
        segmenter = CharacterSegmenter(merge_gap_ratio=0.3)
        boxes = [(10, 10, 30, 50), (42, 10, 30, 50)]  # gap of 2px
        merged = segmenter._merge_nearby(boxes)
        assert len(merged) == 1

    def test_merge_nearby_keeps_far_boxes(self) -> None:
        """Two boxes far apart should not be merged."""
        segmenter = CharacterSegmenter(merge_gap_ratio=0.3)
        boxes = [(10, 10, 30, 50), (100, 10, 30, 50)]  # gap of 60px
        merged = segmenter._merge_nearby(boxes)
        assert len(merged) == 2


# ======================================================================
# StyleExtractor tests
# ======================================================================


class TestStyleExtractor:
    def test_extract_from_chars(self) -> None:
        img = _make_scan_with_chars(5)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        extractor = StyleExtractor()
        features = extractor.extract(chars)

        assert isinstance(features, StyleFeatures)
        assert features.num_characters == len(chars)

    def test_extract_empty_chars(self) -> None:
        extractor = StyleExtractor()
        features = extractor.extract([])

        assert features.num_characters == 0
        assert features.stroke_width_mean == 0.0
        assert features.aspect_ratio_mean == 0.0

    def test_stroke_width_is_positive(self) -> None:
        img = _make_scan_with_chars(3)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        extractor = StyleExtractor()
        features = extractor.extract(chars)

        assert features.stroke_width_mean > 0

    def test_ink_density_in_valid_range(self) -> None:
        img = _make_scan_with_chars(4)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        features = StyleExtractor().extract(chars)

        assert 0.0 <= features.ink_density_mean <= 1.0
        assert features.ink_density_std >= 0.0

    def test_aspect_ratio_is_positive(self) -> None:
        img = _make_scan_with_chars(3)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        features = StyleExtractor().extract(chars)

        assert features.aspect_ratio_mean > 0

    def test_as_dict_returns_all_keys(self) -> None:
        features = StyleFeatures(
            stroke_width_mean=5.0,
            stroke_width_std=1.0,
            stroke_width_median=4.5,
            aspect_ratio_mean=1.0,
            aspect_ratio_std=0.1,
            ink_density_mean=0.3,
            ink_density_std=0.05,
            curvature_mean=0.01,
            curvature_std=0.005,
            num_characters=10,
            avg_char_height=50.0,
            avg_char_width=40.0,
        )
        d = features.as_dict()
        assert len(d) == 12
        assert d["num_characters"] == 10
        assert d["stroke_width_mean"] == 5.0


# ======================================================================
# PrototypeGenerator tests
# ======================================================================


class TestPrototypeGenerator:
    def test_generate_creates_pack(self, tmp_path: Path) -> None:
        img = _make_scan_with_chars(3)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        output_dir = tmp_path / "pack"
        gen = PrototypeGenerator(writer_id="test")
        result = gen.generate(
            chars, labels=["你", "好", "世"], output_dir=output_dir, pack_name="test_pack",
        )

        assert result["pack_name"] == "test_pack"
        assert result["glyph_count"] == 3
        assert result["chars"] == ["你", "好", "世"]
        assert Path(result["manifest_path"]).is_file()

    def test_manifest_is_compatible_format(self, tmp_path: Path) -> None:
        img = _make_scan_with_chars(2)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        output_dir = tmp_path / "pack"
        gen = PrototypeGenerator(writer_id="test")
        gen.generate(chars, labels=["A", "B"], output_dir=output_dir, pack_name="compat")

        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert "name" in manifest
        assert "glyphs" in manifest
        assert isinstance(manifest["glyphs"], list)
        for g in manifest["glyphs"]:
            assert "char" in g
            assert "file" in g
            assert "writer_id" in g

    def test_glyph_files_are_256x256(self, tmp_path: Path) -> None:
        img = _make_scan_with_chars(2)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        output_dir = tmp_path / "pack"
        PrototypeGenerator(glyph_size=256, writer_id="t").generate(
            chars, labels=["X", "Y"], output_dir=output_dir, pack_name="sz",
        )

        for glyph_file in (output_dir / "glyphs").iterdir():
            loaded = cv2.imread(str(glyph_file), cv2.IMREAD_GRAYSCALE)
            assert loaded.shape == (256, 256)

    def test_generate_with_no_labels_uses_positional(self, tmp_path: Path) -> None:
        img = _make_scan_with_chars(2)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        output_dir = tmp_path / "pack"
        result = PrototypeGenerator(writer_id="t").generate(
            chars, labels=None, output_dir=output_dir, pack_name="pos",
        )

        # Without labels, chars should be _pos0, _pos1, ...
        for ch in result["chars"]:
            assert ch.startswith("_pos")

    def test_generate_deduplicates_labels(self, tmp_path: Path) -> None:
        img = _make_scan_with_chars(3)
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(img)

        output_dir = tmp_path / "pack"
        result = PrototypeGenerator(writer_id="t").generate(
            chars, labels=["A", "A", "B"], output_dir=output_dir, pack_name="dedup",
        )

        # Should keep first "A" and "B", skip the duplicate
        assert result["chars"] == ["A", "B"]
        assert result["glyph_count"] == 2

    def test_normalize_empty_image(self) -> None:
        gen = PrototypeGenerator(glyph_size=128)
        empty = np.zeros((0, 0), dtype=np.uint8)
        result = gen._normalize(empty)
        assert result.shape == (128, 128)
        assert np.sum(result) == 0

    def test_normalize_preserves_ink(self) -> None:
        gen = PrototypeGenerator(glyph_size=128, padding_ratio=0.1)
        # Simple square
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 255
        result = gen._normalize(img)
        assert result.shape == (128, 128)
        assert np.sum(result > 0) > 0


# ======================================================================
# Integration test: full pipeline
# ======================================================================


class TestFullPipeline:
    def test_end_to_end_with_synthetic_scan(self, tmp_path: Path) -> None:
        """Full pipeline: create image -> preprocess -> segment -> extract -> generate."""
        # Create a color image that simulates a scan
        h, w = 300, 500
        img = np.full((h, w, 3), 230, dtype=np.uint8)  # light paper
        cv2.rectangle(img, (40, 80), (120, 220), (20, 20, 20), -1)
        cv2.rectangle(img, (160, 80), (240, 220), (20, 20, 20), -1)
        cv2.rectangle(img, (280, 80), (360, 220), (20, 20, 20), -1)

        scan_path = tmp_path / "scan.png"
        cv2.imwrite(str(scan_path), img)

        # Preprocess
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(scan_path)
        assert result.image.ndim == 2

        # Segment
        segmenter = CharacterSegmenter(min_char_area=100)
        chars = segmenter.segment(result.image)
        assert len(chars) >= 1

        # Extract features
        extractor = StyleExtractor()
        features = extractor.extract(chars)
        assert features.num_characters >= 1

        # Generate prototypes
        output_dir = tmp_path / "output_pack"
        gen = PrototypeGenerator(writer_id="e2e_test")
        pack_result = gen.generate(
            chars,
            labels=["你", "好", "世"][: len(chars)],
            output_dir=output_dir,
            pack_name="e2e_pack",
        )

        assert pack_result["glyph_count"] >= 1
        manifest_path = Path(pack_result["manifest_path"])
        assert manifest_path.is_file()

        # Verify the manifest can be loaded by PrototypeLibrary
        from handwrite.prototypes import PrototypeLibrary

        lib = PrototypeLibrary.from_manifest(manifest_path, source_kind="custom")
        assert lib.name == "e2e_pack"
        for ch in pack_result["chars"]:
            assert lib.has_char(ch)


# ======================================================================
# Module-level imports test
# ======================================================================


def test_module_imports_all_classes() -> None:
    """Verify that the package __init__ exports the four main classes."""
    assert IP2 is ImagePreprocessor
    assert CS2 is CharacterSegmenter
    assert SE2 is StyleExtractor
    assert PG2 is PrototypeGenerator
