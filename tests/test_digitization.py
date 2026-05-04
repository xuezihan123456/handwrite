"""Tests for the handwriting digitization module."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from handwrite.digitization.handwriting_recognizer import (
    CharacterRecognition,
    HandwritingRecognizer,
    OCRBackend,
    OCRConfig,
    RecognitionResult,
)
from handwrite.digitization.text_editor import (
    CharacterResult,
    EditableTextDocument,
)
from handwrite.digitization.style_preserver import (
    ExtractedGlyph,
    StylePreserver,
)
from handwrite.digitization.round_trip_engine import (
    RoundTripEngine,
    RoundTripResult,
)


def _has_tesseract() -> bool:
    """Check if Tesseract OCR is installed."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a simple test image with drawn text-like content."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Draw some dark rectangles to simulate text
    cv2.rectangle(img, (50, 50), (100, 150), (0, 0, 0), -1)
    cv2.rectangle(img, (120, 50), (170, 150), (0, 0, 0), -1)
    cv2.rectangle(img, (190, 50), (240, 150), (0, 0, 0), -1)
    # Second line
    cv2.rectangle(img, (50, 180), (100, 280), (0, 0, 0), -1)
    cv2.rectangle(img, (120, 180), (170, 280), (0, 0, 0), -1)
    path = tmp_path / "test_scan.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a PIL Image for testing."""
    arr = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.rectangle(arr, (30, 30), (80, 120), (0, 0, 0), -1)
    cv2.rectangle(arr, (100, 30), (150, 120), (0, 0, 0), -1)
    return Image.fromarray(arr)


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Create a numpy array image for testing."""
    img = np.ones((200, 300), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 30), (80, 120), 0, -1)
    return img


@pytest.fixture
def mock_recognition() -> RecognitionResult:
    """Create a mock RecognitionResult."""
    chars = [
        CharacterRecognition(char="你", confidence=85.5, bbox=(10, 10, 40, 40), line_index=0),
        CharacterRecognition(char="好", confidence=90.2, bbox=(60, 10, 40, 40), line_index=0),
        CharacterRecognition(char="世", confidence=75.0, bbox=(110, 10, 40, 40), line_index=1),
        CharacterRecognition(char="己", confidence=45.0, bbox=(160, 10, 40, 40), line_index=1),
    ]
    return RecognitionResult(
        text="你好世己",
        characters=tuple(chars),
        lines=("你好", "世己"),
        average_confidence=73.93,
        processing_time_ms=150.0,
    )


@pytest.fixture
def mock_glyph() -> ExtractedGlyph:
    """Create a mock ExtractedGlyph."""
    return ExtractedGlyph(
        char="你",
        image=Image.new("L", (256, 256), color=128),
        bbox=(10, 10, 40, 40),
        confidence=85.5,
    )


# ---------------------------------------------------------------------------
# OCRConfig tests
# ---------------------------------------------------------------------------

class TestOCRConfig:
    def test_default_config(self):
        config = OCRConfig()
        assert config.backend == OCRBackend.TESSERACT
        assert config.languages == ("chi_sim", "eng")
        assert config.confidence_threshold == 0.0
        assert config.binarize_method == "otsu"
        assert config.deskew_enabled is True
        assert config.normalize_size == 256

    def test_custom_config(self):
        config = OCRConfig(
            backend=OCRBackend.EASYOCR,
            languages=("ch_tra",),
            confidence_threshold=50.0,
        )
        assert config.backend == OCRBackend.EASYOCR
        assert config.languages == ("ch_tra",)
        assert config.confidence_threshold == 50.0

    def test_frozen(self):
        config = OCRConfig()
        with pytest.raises(AttributeError):
            config.backend = OCRBackend.EASYOCR  # type: ignore


# ---------------------------------------------------------------------------
# HandwritingRecognizer tests
# ---------------------------------------------------------------------------

class TestHandwritingRecognizer:
    def test_init_default(self):
        recognizer = HandwritingRecognizer()
        assert recognizer.config.backend == OCRBackend.TESSERACT

    def test_init_custom(self):
        config = OCRConfig(backend=OCRBackend.EASYOCR)
        recognizer = HandwritingRecognizer(config)
        assert recognizer.config.backend == OCRBackend.EASYOCR

    def test_load_image_from_path(self, sample_image: Path):
        recognizer = HandwritingRecognizer()
        img = recognizer._load_image(sample_image)
        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_load_image_from_pil(self, sample_pil_image: Image.Image):
        recognizer = HandwritingRecognizer()
        img = recognizer._load_image(sample_pil_image)
        assert isinstance(img, np.ndarray)

    def test_load_image_from_numpy(self, sample_numpy_array: np.ndarray):
        recognizer = HandwritingRecognizer()
        img = recognizer._load_image(sample_numpy_array)
        assert isinstance(img, np.ndarray)

    def test_load_image_nonexistent(self):
        recognizer = HandwritingRecognizer()
        with pytest.raises(FileNotFoundError):
            recognizer._load_image(Path("/nonexistent/image.png"))

    def test_preprocess_grayscale(self, sample_image: Path):
        recognizer = HandwritingRecognizer()
        img = recognizer._load_image(sample_image)
        preprocessed = recognizer._preprocess(img)
        assert len(preprocessed.shape) == 2  # Should be grayscale
        # Should be binary (0 or 255)
        unique_values = set(np.unique(preprocessed))
        assert unique_values.issubset({0, 255})

    def test_preprocess_already_grayscale(self, sample_numpy_array: np.ndarray):
        recognizer = HandwritingRecognizer()
        preprocessed = recognizer._preprocess(sample_numpy_array)
        assert len(preprocessed.shape) == 2

    def test_preprocess_adaptive(self, sample_image: Path):
        config = OCRConfig(binarize_method="adaptive")
        recognizer = HandwritingRecognizer(config)
        img = recognizer._load_image(sample_image)
        preprocessed = recognizer._preprocess(img)
        assert len(preprocessed.shape) == 2

    def test_preprocess_fixed_threshold(self, sample_image: Path):
        config = OCRConfig(binarize_method="fixed")
        recognizer = HandwritingRecognizer(config)
        img = recognizer._load_image(sample_image)
        preprocessed = recognizer._preprocess(img)
        assert len(preprocessed.shape) == 2

    def test_preprocess_no_deskew(self, sample_image: Path):
        config = OCRConfig(deskew_enabled=False)
        recognizer = HandwritingRecognizer(config)
        img = recognizer._load_image(sample_image)
        preprocessed = recognizer._preprocess(img)
        assert len(preprocessed.shape) == 2

    def test_deskew_small_angle(self):
        recognizer = HandwritingRecognizer()
        # Create a slightly rotated image
        img = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (180, 80), 255, -1)
        result = recognizer._deskew(img)
        assert result.shape == img.shape

    def test_deskew_empty_image(self):
        recognizer = HandwritingRecognizer()
        img = np.zeros((100, 100), dtype=np.uint8)
        result = recognizer._deskew(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# CharacterRecognition tests
# ---------------------------------------------------------------------------

class TestCharacterRecognition:
    def test_create(self):
        cr = CharacterRecognition(char="A", confidence=95.0, bbox=(10, 20, 30, 40))
        assert cr.char == "A"
        assert cr.confidence == 95.0
        assert cr.bbox == (10, 20, 30, 40)
        assert cr.line_index == 0
        assert cr.word_index == 0

    def test_frozen(self):
        cr = CharacterRecognition(char="A", confidence=95.0, bbox=(0, 0, 10, 10))
        with pytest.raises(AttributeError):
            cr.char = "B"  # type: ignore


# ---------------------------------------------------------------------------
# RecognitionResult tests
# ---------------------------------------------------------------------------

class TestRecognitionResult:
    def test_create(self, mock_recognition: RecognitionResult):
        assert mock_recognition.text == "你好世己"
        assert len(mock_recognition.characters) == 4
        assert len(mock_recognition.lines) == 2
        assert mock_recognition.average_confidence > 0

    def test_lines_match_text(self, mock_recognition: RecognitionResult):
        reconstructed = "".join(mock_recognition.lines)
        assert reconstructed == mock_recognition.text


# ---------------------------------------------------------------------------
# CharacterResult tests
# ---------------------------------------------------------------------------

class TestCharacterResult:
    def test_create(self):
        cr = CharacterResult(
            char="你", original_char="你", confidence=85.0,
            bbox=(10, 10, 40, 40), line_index=0, word_index=0,
        )
        assert cr.char == "你"
        assert cr.original_char == "你"
        assert cr.corrected is False

    def test_correct(self):
        cr = CharacterResult(
            char="你", original_char="你", confidence=85.0,
            bbox=(10, 10, 40, 40), line_index=0, word_index=0,
        )
        cr.correct("好")
        assert cr.char == "好"
        assert cr.original_char == "你"
        assert cr.corrected is True

    def test_correct_same_char(self):
        cr = CharacterResult(
            char="你", original_char="你", confidence=85.0,
            bbox=(10, 10, 40, 40), line_index=0, word_index=0,
        )
        cr.correct("你")
        assert cr.corrected is False  # Same char, no correction

    def test_is_low_confidence(self):
        cr_low = CharacterResult(
            char="A", original_char="A", confidence=30.0,
            bbox=(0, 0, 10, 10), line_index=0, word_index=0,
        )
        cr_high = CharacterResult(
            char="B", original_char="B", confidence=90.0,
            bbox=(0, 0, 10, 10), line_index=0, word_index=0,
        )
        assert cr_low.is_low_confidence() is True
        assert cr_high.is_low_confidence() is False
        assert cr_low.is_low_confidence(threshold=20.0) is False

    def test_to_dict(self):
        cr = CharacterResult(
            char="你", original_char="你", confidence=85.5,
            bbox=(10, 20, 30, 40), line_index=0, word_index=1,
            suggestions=("好",),
        )
        d = cr.to_dict()
        assert d["char"] == "你"
        assert d["confidence"] == 85.5
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["suggestions"] == ["好"]

    def test_from_dict(self):
        data = {
            "char": "你", "original_char": "你", "confidence": 85.5,
            "bbox": [10, 20, 30, 40], "line_index": 0, "word_index": 1,
        }
        cr = CharacterResult.from_dict(data)
        assert cr.char == "你"
        assert cr.confidence == 85.5
        assert cr.bbox == (10, 20, 30, 40)

    def test_roundtrip_dict(self):
        cr = CharacterResult(
            char="好", original_char="好", confidence=92.3,
            bbox=(5, 5, 20, 20), line_index=1, word_index=0,
        )
        restored = CharacterResult.from_dict(cr.to_dict())
        assert restored.char == cr.char
        assert restored.confidence == cr.confidence


# ---------------------------------------------------------------------------
# EditableTextDocument tests
# ---------------------------------------------------------------------------

class TestEditableTextDocument:
    def test_from_recognition(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        assert doc.text == "你好世己"
        assert len(doc.characters) == 4
        assert doc.lines == ["你好", "世己"]

    def test_correct_character(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        assert doc.characters[0].char == "我"
        assert doc.characters[0].corrected is True
        assert doc.text == "我好世己"

    def test_correct_character_invalid_index(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        with pytest.raises(IndexError):
            doc.correct_character(100, "A")

    def test_correct_characters_batch(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_characters({0: "我", 2: "大"})
        assert doc.text == "我好大己"
        assert doc.characters[0].corrected is True
        assert doc.characters[2].corrected is True

    def test_get_low_confidence(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        low = doc.get_low_confidence_characters(threshold=60.0)
        # Characters with confidence < 60: 己 (45.0)
        assert len(low) == 1
        assert low[0][1].char == "己"

    def test_get_corrected_characters(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        corrected = doc.get_corrected_characters()
        assert len(corrected) == 1
        assert corrected[0][1].char == "我"

    def test_generate_suggestions(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.generate_suggestions()
        # Low-confidence chars should have suggestions
        low_chars = doc.get_low_confidence_characters()
        for _, ch in low_chars:
            # Suggestions may or may not exist depending on confusion pairs
            assert isinstance(ch.suggestions, tuple)

    def test_lines_rebuilt_after_correction(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        assert doc.lines[0] == "我好"

    def test_correction_history(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        doc.correct_character(2, "大")
        history = doc.correction_history
        assert len(history) == 2
        assert history[0]["old_char"] == "你"
        assert history[0]["new_char"] == "我"

    def test_to_dict(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition, source_path="test.png")
        d = doc.to_dict()
        assert d["text"] == "你好世己"
        assert d["source_path"] == "test.png"
        assert "statistics" in d
        assert d["statistics"]["total_characters"] == 4

    def test_to_json(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        j = doc.to_json()
        parsed = json.loads(j)
        assert parsed["text"] == "你好世己"

    def test_save_and_load_json(self, mock_recognition: RecognitionResult, tmp_path: Path):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        path = tmp_path / "doc.json"
        doc.save_json(path)
        assert path.exists()

        loaded = EditableTextDocument.load_json(path)
        assert loaded.text == "我好世己"
        assert loaded.characters[0].corrected is True

    def test_from_dict(self, mock_recognition: RecognitionResult):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        d = doc.to_dict()
        restored = EditableTextDocument.from_dict(d)
        assert restored.text == doc.text
        assert len(restored.characters) == len(doc.characters)


# ---------------------------------------------------------------------------
# StylePreserver tests
# ---------------------------------------------------------------------------

class TestStylePreserver:
    def test_init(self):
        preserver = StylePreserver(normalize_size=128, padding=5)
        assert preserver.normalize_size == 128

    def test_crop_and_normalize(self, sample_image: Path):
        preserver = StylePreserver(normalize_size=256)
        img = preserver._load_image(sample_image)
        glyph = preserver._crop_and_normalize(img, (50, 50, 50, 100))
        assert glyph is not None
        assert glyph.size == (256, 256)

    def test_crop_and_normalize_small_size(self, sample_image: Path):
        preserver = StylePreserver(normalize_size=64)
        img = preserver._load_image(sample_image)
        glyph = preserver._crop_and_normalize(img, (50, 50, 50, 100))
        assert glyph is not None
        assert glyph.size == (64, 64)

    def test_crop_invalid_bbox(self, sample_image: Path):
        preserver = StylePreserver()
        img = preserver._load_image(sample_image)
        glyph = preserver._crop_and_normalize(img, (0, 0, 0, 0))
        assert glyph is None

    def test_crop_out_of_bounds(self, sample_image: Path):
        preserver = StylePreserver()
        img = preserver._load_image(sample_image)
        # Bbox partially out of bounds
        glyph = preserver._crop_and_normalize(img, (550, 350, 100, 100))
        # Should handle gracefully (may return None or a valid image)
        # No assertion needed - just shouldn't crash

    def test_extract_glyphs(self, sample_image: Path, mock_recognition: RecognitionResult):
        preserver = StylePreserver()
        # Use the actual image with mock recognition bboxes adjusted to fit
        chars = [
            CharacterRecognition(char="你", confidence=85.5, bbox=(50, 50, 50, 100), line_index=0),
            CharacterRecognition(char="好", confidence=90.2, bbox=(120, 50, 50, 100), line_index=0),
        ]
        result = RecognitionResult(
            text="你好",
            characters=tuple(chars),
            lines=("你好",),
            average_confidence=87.85,
            processing_time_ms=100.0,
        )
        glyphs = preserver.extract_glyphs(sample_image, result)
        assert len(glyphs) == 2
        assert all(g.image.size == (256, 256) for g in glyphs)

    def test_extract_glyphs_skips_whitespace(self, sample_image: Path):
        preserver = StylePreserver()
        chars = [
            CharacterRecognition(char=" ", confidence=85.0, bbox=(50, 50, 50, 50), line_index=0),
            CharacterRecognition(char="你", confidence=85.0, bbox=(50, 50, 50, 100), line_index=0),
        ]
        result = RecognitionResult(
            text=" 你",
            characters=tuple(chars),
            lines=(" 你",),
            average_confidence=85.0,
            processing_time_ms=50.0,
        )
        glyphs = preserver.extract_glyphs(sample_image, result)
        assert len(glyphs) == 1
        assert glyphs[0].char == "你"

    def test_extract_glyphs_min_confidence(self, sample_image: Path):
        preserver = StylePreserver()
        chars = [
            CharacterRecognition(char="你", confidence=85.0, bbox=(50, 50, 50, 100), line_index=0),
            CharacterRecognition(char="好", confidence=20.0, bbox=(120, 50, 50, 100), line_index=0),
        ]
        result = RecognitionResult(
            text="你好",
            characters=tuple(chars),
            lines=("你好",),
            average_confidence=52.5,
            processing_time_ms=50.0,
        )
        glyphs = preserver.extract_glyphs(sample_image, result, min_confidence=50.0)
        assert len(glyphs) == 1
        assert glyphs[0].char == "你"

    def test_extract_deduplicated(self, sample_image: Path):
        preserver = StylePreserver()
        chars = [
            CharacterRecognition(char="你", confidence=70.0, bbox=(50, 50, 50, 100), line_index=0),
            CharacterRecognition(char="你", confidence=90.0, bbox=(50, 50, 50, 100), line_index=1),
        ]
        result = RecognitionResult(
            text="你你",
            characters=tuple(chars),
            lines=("你", "你"),
            average_confidence=80.0,
            processing_time_ms=50.0,
        )
        deduped = preserver.extract_deduplicated_glyphs(sample_image, result)
        assert len(deduped) == 1
        assert deduped["你"].confidence == 90.0  # Kept the higher confidence one

    def test_save_as_prototype_pack(self, mock_glyph: ExtractedGlyph, tmp_path: Path):
        preserver = StylePreserver()
        glyphs = {"你": mock_glyph}
        manifest_path = preserver.save_as_prototype_pack(glyphs, tmp_path, "test_pack")
        assert manifest_path.exists()
        assert manifest_path.name == "manifest.json"

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["name"] == "test_pack"
        assert data["glyph_count"] == 1
        assert data["glyphs"][0]["char"] == "你"

        # Check glyph file was saved
        glyph_dir = tmp_path / "glyphs"
        assert glyph_dir.exists()
        glyph_files = list(glyph_dir.glob("*.png"))
        assert len(glyph_files) == 1

    def test_save_as_prototype_pack_list(self, mock_glyph: ExtractedGlyph, tmp_path: Path):
        preserver = StylePreserver()
        manifest_path = preserver.save_as_prototype_pack([mock_glyph], tmp_path, "list_pack")
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["glyph_count"] == 1

    def test_save_multiple_glyphs(self, tmp_path: Path):
        preserver = StylePreserver()
        glyphs = {
            "你": ExtractedGlyph(
                char="你", image=Image.new("L", (256, 256), 128),
                bbox=(10, 10, 40, 40), confidence=85.0,
            ),
            "好": ExtractedGlyph(
                char="好", image=Image.new("L", (256, 256), 100),
                bbox=(60, 10, 40, 40), confidence=90.0,
            ),
        }
        manifest_path = preserver.save_as_prototype_pack(glyphs, tmp_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["glyph_count"] == 2


# ---------------------------------------------------------------------------
# ExtractedGlyph tests
# ---------------------------------------------------------------------------

class TestExtractedGlyph:
    def test_create(self, mock_glyph: ExtractedGlyph):
        assert mock_glyph.char == "你"
        assert mock_glyph.confidence == 85.5
        assert mock_glyph.image.size == (256, 256)

    def test_to_dict(self, mock_glyph: ExtractedGlyph):
        d = mock_glyph.to_dict()
        assert d["char"] == "你"
        assert d["confidence"] == 85.5
        assert d["bbox"] == [10, 10, 40, 40]

    def test_frozen(self, mock_glyph: ExtractedGlyph):
        with pytest.raises(AttributeError):
            mock_glyph.char = "好"  # type: ignore


# ---------------------------------------------------------------------------
# RoundTripEngine tests
# ---------------------------------------------------------------------------

class TestRoundTripEngine:
    def test_init(self):
        engine = RoundTripEngine()
        assert engine.recognizer is not None
        assert engine.preserver is not None

    def test_init_custom_config(self):
        config = OCRConfig(backend=OCRBackend.EASYOCR)
        engine = RoundTripEngine(ocr_config=config)
        assert engine.recognizer.config.backend == OCRBackend.EASYOCR

    @pytest.mark.skipif(not _has_tesseract(), reason="Tesseract not installed")
    def test_scan_to_editable(self, sample_image: Path):
        engine = RoundTripEngine()
        doc = engine.scan_to_editable(sample_image)
        assert isinstance(doc, EditableTextDocument)
        assert doc.source_path == str(sample_image)

    @pytest.mark.skipif(not _has_tesseract(), reason="Tesseract not installed")
    def test_scan_to_editable_pil(self, sample_pil_image: Image.Image):
        engine = RoundTripEngine()
        doc = engine.scan_to_editable(sample_pil_image)
        assert isinstance(doc, EditableTextDocument)

    def test_load_pil_image_from_path(self, sample_image: Path):
        engine = RoundTripEngine()
        img = engine._load_pil_image(sample_image)
        assert isinstance(img, Image.Image)

    def test_load_pil_image_from_pil(self, sample_pil_image: Image.Image):
        engine = RoundTripEngine()
        img = engine._load_pil_image(sample_pil_image)
        assert isinstance(img, Image.Image)

    def test_load_pil_image_from_numpy(self, sample_numpy_array: np.ndarray):
        engine = RoundTripEngine()
        img = engine._load_pil_image(sample_numpy_array)
        assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# RoundTripResult tests
# ---------------------------------------------------------------------------

class TestRoundTripResult:
    def test_create(self, mock_recognition: RecognitionResult, mock_glyph: ExtractedGlyph):
        doc = EditableTextDocument.from_recognition(mock_recognition)
        result = RoundTripResult(
            original_image=Image.new("RGB", (100, 100)),
            recognition=mock_recognition,
            document=doc,
            extracted_glyphs={"你": mock_glyph},
            prototype_pack_path=None,
            regenerated_image=None,
        )
        assert result.document.text == "你好世己"
        assert len(result.extracted_glyphs) == 1


# ---------------------------------------------------------------------------
# Integration tests (data flow)
# ---------------------------------------------------------------------------

class TestIntegrationFlow:
    def test_recognition_to_editor_flow(self, mock_recognition: RecognitionResult):
        """Test the flow from recognition to editable document."""
        doc = EditableTextDocument.from_recognition(mock_recognition)

        # Check initial state
        assert doc.text == "你好世己"
        low = doc.get_low_confidence_characters(60.0)
        assert len(low) >= 1  # 界 has 45% confidence

        # Generate suggestions
        doc.generate_suggestions()

        # Correct low-confidence chars
        for idx, ch in low:
            if ch.suggestions:
                doc.correct_character(idx, ch.suggestions[0])

        # Verify corrections applied
        assert doc.correction_history

    def test_editor_serialization_roundtrip(self, mock_recognition: RecognitionResult, tmp_path: Path):
        """Test full serialization roundtrip."""
        doc = EditableTextDocument.from_recognition(mock_recognition)
        doc.correct_character(0, "我")
        doc.generate_suggestions()

        # Save
        path = tmp_path / "test_doc.json"
        doc.save_json(path)

        # Load and verify
        loaded = EditableTextDocument.load_json(path)
        assert loaded.text == doc.text
        assert loaded.characters[0].corrected is True
        assert loaded.characters[0].char == "我"

    @pytest.mark.skipif(not _has_tesseract(), reason="Tesseract not installed")
    def test_style_preserver_with_recognizer(self, sample_image: Path):
        """Test style preserver integration with recognizer."""
        recognizer = HandwritingRecognizer()
        preserver = StylePreserver(normalize_size=128)

        result = recognizer.recognize(sample_image)
        # Use actual bboxes from recognition (they may be empty for synthetic image)
        if result.characters:
            glyphs = preserver.extract_glyphs(sample_image, result, min_confidence=0.0)
            # Verify glyph images are correct size
            for g in glyphs:
                assert g.image.size == (128, 128)


# ---------------------------------------------------------------------------
# Public API tests (digitize / round_trip from __init__)
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_digitize_import(self):
        """Test that digitize function is importable from handwrite."""
        from handwrite import digitize
        assert callable(digitize)

    def test_round_trip_import(self):
        """Test that round_trip function is importable from handwrite."""
        from handwrite import round_trip
        assert callable(round_trip)

    def test_digitize_module_imports(self):
        """Test that all digitization submodules import cleanly."""
        from handwrite.digitization import (
            HandwritingRecognizer,
            CharacterResult,
            EditableTextDocument,
            StylePreserver,
            RoundTripEngine,
        )
        assert HandwritingRecognizer is not None
        assert CharacterResult is not None
        assert EditableTextDocument is not None
        assert StylePreserver is not None
        assert RoundTripEngine is not None
