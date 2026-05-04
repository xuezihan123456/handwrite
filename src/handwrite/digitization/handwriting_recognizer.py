"""Handwriting recognition engine using OCR backends.

Supports pytesseract and easyocr for Chinese/mixed-language handwriting recognition,
with preprocessing optimized for handwritten content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image


class OCRBackend(str, Enum):
    """Supported OCR backends."""

    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for the handwriting recognizer."""

    backend: OCRBackend = OCRBackend.TESSERACT
    languages: tuple[str, ...] = ("chi_sim", "eng")
    confidence_threshold: float = 0.0
    binarize_method: str = "otsu"
    denoise_strength: int = 10
    deskew_enabled: bool = True
    normalize_size: int = 256


@dataclass(frozen=True)
class CharacterRecognition:
    """Recognition result for a single character."""

    char: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    line_index: int = 0
    word_index: int = 0


@dataclass(frozen=True)
class RecognitionResult:
    """Full recognition result from a handwriting scan."""

    text: str
    characters: tuple[CharacterRecognition, ...]
    lines: tuple[str, ...]
    average_confidence: float
    processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class HandwritingRecognizer:
    """Recognizes handwritten text from scanned images.

    Supports Chinese and mixed Chinese-English handwriting with
    preprocessing optimized for handwritten content.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self._config = config or OCRConfig()
        self._easyocr_reader = None

    @property
    def config(self) -> OCRConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> RecognitionResult:
        """Recognize handwritten text from an image.

        Args:
            image: File path, PIL Image, or numpy array (BGR/grayscale).

        Returns:
            RecognitionResult with text, per-character info, and confidence.
        """
        import time

        start = time.perf_counter()

        img_array = self._load_image(image)
        preprocessed = self._preprocess(img_array)
        raw_results = self._run_ocr(preprocessed, img_array)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return self._build_result(raw_results, elapsed_ms)

    def recognize_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image, np.ndarray]],
    ) -> list[RecognitionResult]:
        """Recognize multiple images in sequence."""
        return [self.recognize(img) for img in images]

    # ------------------------------------------------------------------
    # Image preprocessing pipeline
    # ------------------------------------------------------------------

    def _load_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image.copy()
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        path = Path(image)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline: grayscale, denoise, binarize, deskew."""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Denoise
        if self._config.denoise_strength > 0:
            gray = cv2.fastNlMeansDenoising(
                gray, None, h=self._config.denoise_strength, templateWindowSize=7, searchWindowSize=21
            )

        # Binarize
        if self._config.binarize_method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif self._config.binarize_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
            )
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Deskew
        if self._config.deskew_enabled:
            binary = self._deskew(binary)

        return binary

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        """Correct skew angle of the document."""
        coords = np.column_stack(np.where(binary > 0))
        if coords.shape[0] < 50:
            return binary

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct small angles to avoid over-rotation
        if abs(angle) > 10:
            return binary

        h, w = binary.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    # ------------------------------------------------------------------
    # OCR backend dispatch
    # ------------------------------------------------------------------

    def _run_ocr(
        self, preprocessed: np.ndarray, original: np.ndarray
    ) -> list[CharacterRecognition]:
        backend = self._config.backend
        if backend == OCRBackend.TESSERACT:
            return self._run_tesseract(preprocessed)
        if backend == OCRBackend.EASYOCR:
            return self._run_easyocr(original)
        raise ValueError(f"Unsupported OCR backend: {backend}")

    def _run_tesseract(self, binary: np.ndarray) -> list[CharacterRecognition]:
        """Run Tesseract OCR with detailed per-character output."""
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for Tesseract backend. "
                "Install with: pip install pytesseract"
            )

        # Tesseract expects dark text on light background
        inverted = cv2.bitwise_not(binary)
        pil_image = Image.fromarray(inverted)

        lang = "+".join(self._config.languages)

        # Get per-character data
        data = pytesseract.image_to_data(
            pil_image, lang=lang, output_type=pytesseract.Output.DICT
        )

        results: list[CharacterRecognition] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            line_idx = int(data["line_num"][i])
            word_idx = int(data["word_num"][i])

            # Split multi-character words into individual characters for CJK
            for ch_idx, ch in enumerate(text):
                char_conf = max(0.0, min(100.0, conf))
                if len(text) > 1 and ch_idx > 0:
                    # Estimate sub-character bbox
                    char_w = max(1, w // len(text))
                    char_x = x + ch_idx * char_w
                    bbox = (char_x, y, char_w, h)
                else:
                    bbox = (x, y, w, h)

                results.append(
                    CharacterRecognition(
                        char=ch,
                        confidence=char_conf,
                        bbox=bbox,
                        line_index=line_idx,
                        word_index=word_idx,
                    )
                )

        return results

    def _run_easyocr(self, original: np.ndarray) -> list[CharacterRecognition]:
        """Run easyocr with per-character bounding boxes."""
        if self._easyocr_reader is None:
            try:
                import easyocr
            except ImportError:
                raise ImportError(
                    "easyocr is required for easyocr backend. "
                    "Install with: pip install easyocr"
                )
            lang_list = list(self._config.languages)
            # Map Tesseract language codes to easyocr codes
            mapped = []
            for lang in lang_list:
                if lang == "chi_sim":
                    mapped.append("ch_sim")
                elif lang == "chi_tra":
                    mapped.append("ch_tra")
                else:
                    mapped.append(lang)
            self._easyocr_reader = easyocr.Reader(mapped, gpu=False)

        # easyocr expects RGB
        if len(original.shape) == 2:
            rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        elif original.shape[2] == 4:
            rgb = cv2.cvtColor(original, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        raw = self._easyocr_reader.readtext(rgb, detail=1)

        results: list[CharacterRecognition] = []
        for line_idx, (bbox_pts, text, conf) in enumerate(raw):
            conf_score = float(conf) * 100.0
            if conf_score < self._config.confidence_threshold:
                continue

            # Convert polygon to bounding rect
            pts = np.array(bbox_pts, dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            w = int(x_max - x_min)
            h = int(y_max - y_min)

            for ch_idx, ch in enumerate(text):
                if len(text) > 1:
                    char_w = max(1, w // len(text))
                    char_x = int(x_min) + ch_idx * char_w
                    bbox = (char_x, int(y_min), char_w, int(h))
                else:
                    bbox = (int(x_min), int(y_min), w, int(h))

                results.append(
                    CharacterRecognition(
                        char=ch,
                        confidence=conf_score,
                        bbox=bbox,
                        line_index=line_idx,
                        word_index=0,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(
        self,
        characters: list[CharacterRecognition],
        elapsed_ms: float,
    ) -> RecognitionResult:
        # Filter by confidence threshold
        filtered = [
            c for c in characters if c.confidence >= self._config.confidence_threshold
        ]

        # Build text and lines
        text = "".join(c.char for c in filtered)
        lines_dict: dict[int, list[str]] = {}
        for c in filtered:
            lines_dict.setdefault(c.line_index, []).append(c.char)
        lines = tuple("".join(chs) for _, chs in sorted(lines_dict.items()))

        # Average confidence
        if filtered:
            avg_conf = sum(c.confidence for c in filtered) / len(filtered)
        else:
            avg_conf = 0.0

        return RecognitionResult(
            text=text,
            characters=tuple(filtered),
            lines=lines,
            average_confidence=round(avg_conf, 2),
            processing_time_ms=round(elapsed_ms, 2),
            metadata={
                "backend": self._config.backend.value,
                "languages": list(self._config.languages),
                "total_characters": len(filtered),
            },
        )


__all__ = [
    "OCRBackend",
    "OCRConfig",
    "CharacterRecognition",
    "RecognitionResult",
    "HandwritingRecognizer",
]
