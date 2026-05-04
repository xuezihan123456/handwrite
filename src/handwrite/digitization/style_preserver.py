"""Style preserver - extracts and stores character glyphs from recognized scans.

Takes the bounding boxes from OCR recognition and extracts individual character
glyphs, normalizing them to a standard size for reuse as prototype library entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from handwrite.digitization.handwriting_recognizer import (
    CharacterRecognition,
    RecognitionResult,
)


@dataclass(frozen=True)
class ExtractedGlyph:
    """A single extracted character glyph with metadata."""

    char: str
    image: Image.Image
    bbox: tuple[int, int, int, int]
    confidence: float
    writer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "char": self.char,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 2),
            "writer_id": self.writer_id,
        }


class StylePreserver:
    """Extracts character glyphs from scanned handwriting and stores them
    as a reusable prototype library.
    """

    def __init__(
        self,
        normalize_size: int = 256,
        padding: int = 10,
        writer_id: str | None = None,
    ) -> None:
        self._normalize_size = normalize_size
        self._padding = padding
        self._writer_id = writer_id

    @property
    def normalize_size(self) -> int:
        return self._normalize_size

    # ------------------------------------------------------------------
    # Glyph extraction
    # ------------------------------------------------------------------

    def extract_glyphs(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        recognition: RecognitionResult,
        min_confidence: float = 30.0,
    ) -> list[ExtractedGlyph]:
        """Extract character glyphs from the source image using recognition bboxes.

        Args:
            image: The original scanned image (not preprocessed).
            recognition: RecognitionResult containing character bboxes.
            min_confidence: Minimum confidence to extract a glyph.

        Returns:
            List of ExtractedGlyph objects.
        """
        img_array = self._load_image(image)
        glyphs: list[ExtractedGlyph] = []

        for char_rec in recognition.characters:
            if char_rec.char.isspace():
                continue
            if char_rec.confidence < min_confidence:
                continue

            glyph_image = self._crop_and_normalize(img_array, char_rec.bbox)
            if glyph_image is None:
                continue

            glyphs.append(
                ExtractedGlyph(
                    char=char_rec.char,
                    image=glyph_image,
                    bbox=char_rec.bbox,
                    confidence=char_rec.confidence,
                    writer_id=self._writer_id,
                )
            )

        return glyphs

    def extract_deduplicated_glyphs(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        recognition: RecognitionResult,
        min_confidence: float = 30.0,
    ) -> dict[str, ExtractedGlyph]:
        """Extract glyphs and deduplicate by character, keeping the highest-confidence instance."""
        all_glyphs = self.extract_glyphs(image, recognition, min_confidence)
        best: dict[str, ExtractedGlyph] = {}
        for glyph in all_glyphs:
            existing = best.get(glyph.char)
            if existing is None or glyph.confidence > existing.confidence:
                best[glyph.char] = glyph
        return best

    # ------------------------------------------------------------------
    # Prototype pack export
    # ------------------------------------------------------------------

    def save_as_prototype_pack(
        self,
        glyphs: dict[str, ExtractedGlyph] | list[ExtractedGlyph],
        output_dir: str | Path,
        pack_name: str = "extracted_style",
    ) -> Path:
        """Save extracted glyphs as a prototype pack compatible with the HandWrite system.

        Creates:
            output_dir/
                manifest.json
                glyphs/
                    <char>.png

        Returns:
            Path to the created manifest.json.
        """
        output = Path(output_dir)
        glyphs_dir = output / "glyphs"
        glyphs_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(glyphs, list):
            deduped: dict[str, ExtractedGlyph] = {}
            for g in glyphs:
                existing = deduped.get(g.char)
                if existing is None or g.confidence > existing.confidence:
                    deduped[g.char] = g
            glyphs = deduped

        manifest_entries = []
        for char, glyph in sorted(glyphs.items()):
            safe_name = f"U{ord(char):04X}.png"
            glyph_path = glyphs_dir / safe_name
            glyph.image.save(str(glyph_path))
            manifest_entries.append(
                {
                    "char": char,
                    "file": f"glyphs/{safe_name}",
                    "confidence": round(glyph.confidence, 2),
                    "writer_id": glyph.writer_id,
                }
            )

        manifest = {
            "name": pack_name,
            "version": 1,
            "source": "digitization",
            "normalize_size": self._normalize_size,
            "glyph_count": len(manifest_entries),
            "glyphs": manifest_entries,
        }

        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image.copy()
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        path = Path(image)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def _crop_and_normalize(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image | None:
        """Crop a character region and normalize to the standard size."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None

        img_h, img_w = img.shape[:2]
        pad = self._padding

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Binarize with Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the content bounding box and crop to it
        coords = np.column_stack(np.where(binary < 128))
        if coords.shape[0] > 0:
            cy_min, cx_min = coords.min(axis=0)
            cy_max, cx_max = coords.max(axis=0)
            # Add a small margin
            margin = 2
            cy_min = max(0, cy_min - margin)
            cx_min = max(0, cx_min - margin)
            cy_max = min(binary.shape[0], cy_max + margin)
            cx_max = min(binary.shape[1], cx_max + margin)
            binary = binary[cy_min:cy_max, cx_min:cx_max]

        # Normalize to target size
        target = self._normalize_size
        h_crop, w_crop = binary.shape[:2]
        scale = min(target / max(h_crop, 1), target / max(w_crop, 1))
        new_w = max(1, int(w_crop * scale))
        new_h = max(1, int(h_crop * scale))
        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center on a white canvas
        canvas = np.ones((target, target), dtype=np.uint8) * 255
        y_offset = (target - new_h) // 2
        x_offset = (target - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return Image.fromarray(canvas, mode="L")


__all__ = [
    "ExtractedGlyph",
    "StylePreserver",
]
