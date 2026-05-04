"""OCR-based handwriting style extraction module.

Extracts handwriting style features from scanned notes and generates
a prototype glyph library compatible with the existing PrototypeLibrary.
"""

from __future__ import annotations

from handwrite.ocr_style.image_preprocessor import ImagePreprocessor
from handwrite.ocr_style.character_segmenter import CharacterSegmenter
from handwrite.ocr_style.style_extractor import StyleExtractor
from handwrite.ocr_style.prototype_generator import PrototypeGenerator

__all__ = [
    "ImagePreprocessor",
    "CharacterSegmenter",
    "StyleExtractor",
    "PrototypeGenerator",
]
