"""Round-trip engine: scan -> recognize -> edit -> regenerate handwriting.

Orchestrates the full pipeline of digitizing a handwritten scan, allowing
corrections, and regenerating the text in the original handwriting style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from handwrite.digitization.handwriting_recognizer import (
    HandwritingRecognizer,
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


@dataclass(frozen=True)
class RoundTripResult:
    """Result of a full round-trip: scan -> recognize -> edit -> regenerate."""

    original_image: Image.Image | None
    recognition: RecognitionResult
    document: EditableTextDocument
    extracted_glyphs: dict[str, ExtractedGlyph]
    prototype_pack_path: Path | None
    regenerated_image: Image.Image | None
    metadata: dict[str, Any] = field(default_factory=dict)


class RoundTripEngine:
    """Orchestrates the complete round-trip pipeline.

    Pipeline stages:
    1. Scan: Load and preprocess the input image
    2. Recognize: OCR to extract text with per-character confidence
    3. Edit: Create an editable document for review and correction
    4. Preserve: Extract character glyphs for style preservation
    5. Regenerate: Generate new handwriting in the original style
    """

    def __init__(
        self,
        ocr_config: OCRConfig | None = None,
        normalize_size: int = 256,
        writer_id: str | None = None,
    ) -> None:
        self._recognizer = HandwritingRecognizer(ocr_config)
        self._preserver = StylePreserver(
            normalize_size=normalize_size,
            writer_id=writer_id,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def recognizer(self) -> HandwritingRecognizer:
        return self._recognizer

    @property
    def preserver(self) -> StylePreserver:
        return self._preserver

    # ------------------------------------------------------------------
    # Full round-trip pipeline
    # ------------------------------------------------------------------

    def scan_to_editable(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        source_path: str | None = None,
    ) -> EditableTextDocument:
        """Stage 1-3: Scan -> Recognize -> Create editable document.

        Args:
            image: The scanned handwriting image.
            source_path: Optional file path for metadata.

        Returns:
            EditableTextDocument ready for review and correction.
        """
        recognition = self._recognizer.recognize(image)
        if source_path is None and isinstance(image, (str, Path)):
            source_path = str(image)
        return EditableTextDocument.from_recognition(recognition, source_path=source_path)

    def extract_style(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        recognition: RecognitionResult | None = None,
        min_confidence: float = 30.0,
    ) -> dict[str, ExtractedGlyph]:
        """Stage 4: Extract character glyphs for style preservation.

        Args:
            image: The original scanned image.
            recognition: Optional pre-computed recognition result.
            min_confidence: Minimum confidence to extract a glyph.

        Returns:
            Dictionary mapping characters to their best ExtractedGlyph.
        """
        if recognition is None:
            recognition = self._recognizer.recognize(image)
        return self._preserver.extract_deduplicated_glyphs(
            image, recognition, min_confidence=min_confidence
        )

    def save_style_pack(
        self,
        glyphs: dict[str, ExtractedGlyph],
        output_dir: str | Path,
        pack_name: str = "extracted_style",
    ) -> Path:
        """Save extracted glyphs as a prototype pack.

        Returns:
            Path to the created manifest.json.
        """
        return self._preserver.save_as_prototype_pack(glyphs, output_dir, pack_name)

    def regenerate(
        self,
        text: str,
        prototype_pack_path: str | Path | None = None,
        style: str = "行书流畅",
        paper: str = "白纸",
        layout: str = "自然",
        font_size: int = 80,
    ) -> Image.Image:
        """Stage 5: Regenerate text as handwriting.

        Uses the HandWrite public API to generate handwriting with the
        extracted prototype pack (if available).

        Args:
            text: The text to regenerate.
            prototype_pack_path: Path to the extracted prototype pack.
            style: Handwriting style name.
            paper: Paper type.
            layout: Layout style.
            font_size: Character font size.

        Returns:
            Generated handwriting page as PIL Image.
        """
        import handwrite

        kwargs: dict[str, Any] = {
            "style": style,
            "paper": paper,
            "layout": layout,
            "font_size": font_size,
        }
        if prototype_pack_path is not None:
            kwargs["prototype_pack"] = str(prototype_pack_path)

        return handwrite.generate(text, **kwargs)

    # ------------------------------------------------------------------
    # Convenience: full round-trip
    # ------------------------------------------------------------------

    def round_trip(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        output_dir: str | Path | None = None,
        pack_name: str = "extracted_style",
        style: str = "行书流畅",
        paper: str = "白纸",
        layout: str = "自然",
        font_size: int = 80,
        corrections: dict[int, str] | None = None,
        save_pack: bool = True,
    ) -> RoundTripResult:
        """Execute the full round-trip pipeline.

        Args:
            image: The scanned handwriting image.
            output_dir: Directory to save the extracted prototype pack.
            pack_name: Name for the extracted prototype pack.
            style: Handwriting style for regeneration.
            paper: Paper type for regeneration.
            layout: Layout style for regeneration.
            font_size: Font size for regeneration.
            corrections: Optional corrections to apply before regeneration.
            save_pack: Whether to save the extracted style as a prototype pack.

        Returns:
            RoundTripResult with all intermediate artifacts.
        """
        source_path = str(image) if isinstance(image, (str, Path)) else None

        # Load original image for reference
        original = self._load_pil_image(image)

        # Stage 1-3: Recognize and create editable document
        recognition = self._recognizer.recognize(image)
        document = EditableTextDocument.from_recognition(recognition, source_path=source_path)

        # Apply corrections if provided
        if corrections:
            document.correct_characters(corrections)

        # Stage 4: Extract style glyphs
        glyphs = self._preserver.extract_deduplicated_glyphs(
            self._preserver._load_image(image), recognition
        )

        # Save prototype pack
        pack_path = None
        if save_pack and output_dir is not None and glyphs:
            pack_path = self.save_style_pack(glyphs, output_dir, pack_name)

        # Stage 5: Regenerate
        regenerated = None
        corrected_text = document.text
        if corrected_text.strip():
            regenerated = self.regenerate(
                corrected_text,
                prototype_pack_path=pack_path,
                style=style,
                paper=paper,
                layout=layout,
                font_size=font_size,
            )

        return RoundTripResult(
            original_image=original,
            recognition=recognition,
            document=document,
            extracted_glyphs=glyphs,
            prototype_pack_path=pack_path,
            regenerated_image=regenerated,
            metadata={
                "style": style,
                "paper": paper,
                "layout": layout,
                "font_size": font_size,
                "glyph_count": len(glyphs),
                "corrections_applied": len(corrections or {}),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pil_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.copy()
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return Image.open(image)


__all__ = [
    "RoundTripResult",
    "RoundTripEngine",
]
