"""Handwriting digitization module.

Provides bidirectional conversion between handwritten scans and editable text,
while preserving the original handwriting style for regeneration.
"""

from handwrite.digitization.handwriting_recognizer import HandwritingRecognizer
from handwrite.digitization.text_editor import (
    CharacterResult,
    EditableTextDocument,
)
from handwrite.digitization.style_preserver import StylePreserver
from handwrite.digitization.round_trip_engine import RoundTripEngine

__all__ = [
    "HandwritingRecognizer",
    "CharacterResult",
    "EditableTextDocument",
    "StylePreserver",
    "RoundTripEngine",
]
