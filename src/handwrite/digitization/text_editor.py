"""Editable text document with per-character confidence and correction support.

Provides data structures for reviewing OCR results, correcting individual characters,
and managing correction suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Optional

from handwrite.digitization.handwriting_recognizer import (
    CharacterRecognition,
    RecognitionResult,
)


@dataclass
class CharacterResult:
    """A single character with recognition metadata and correction support."""

    char: str
    original_char: str
    confidence: float
    bbox: tuple[int, int, int, int]
    line_index: int
    word_index: int
    corrected: bool = False
    suggestions: tuple[str, ...] = ()

    def correct(self, new_char: str) -> None:
        """Apply a correction to this character."""
        if new_char != self.char:
            self.char = new_char
            self.corrected = True

    def is_low_confidence(self, threshold: float = 60.0) -> bool:
        """Check if this character has low confidence."""
        return self.confidence < threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "char": self.char,
            "original_char": self.original_char,
            "confidence": round(self.confidence, 2),
            "bbox": list(self.bbox),
            "line_index": self.line_index,
            "word_index": self.word_index,
            "corrected": self.corrected,
            "suggestions": list(self.suggestions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterResult:
        return cls(
            char=data["char"],
            original_char=data["original_char"],
            confidence=float(data["confidence"]),
            bbox=tuple(data["bbox"]),
            line_index=int(data["line_index"]),
            word_index=int(data["word_index"]),
            corrected=bool(data.get("corrected", False)),
            suggestions=tuple(data.get("suggestions", ())),
        )


class EditableTextDocument:
    """An editable document built from OCR recognition results.

    Supports per-character review, batch correction, suggestion generation,
    and export to JSON/plain text.
    """

    def __init__(
        self,
        characters: list[CharacterResult],
        lines: list[str],
        source_path: str | None = None,
    ) -> None:
        self._characters = characters
        self._lines = lines
        self._source_path = source_path
        self._correction_history: list[dict[str, Any]] = []

    @classmethod
    def from_recognition(
        cls,
        result: RecognitionResult,
        source_path: str | None = None,
    ) -> EditableTextDocument:
        """Create an editable document from a RecognitionResult."""
        characters = [
            CharacterResult(
                char=c.char,
                original_char=c.char,
                confidence=c.confidence,
                bbox=c.bbox,
                line_index=c.line_index,
                word_index=c.word_index,
            )
            for c in result.characters
        ]
        return cls(
            characters=characters,
            lines=list(result.lines),
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def characters(self) -> list[CharacterResult]:
        return list(self._characters)

    @property
    def text(self) -> str:
        return "".join(c.char for c in self._characters)

    @property
    def lines(self) -> list[str]:
        return list(self._lines)

    @property
    def source_path(self) -> str | None:
        return self._source_path

    @property
    def correction_history(self) -> list[dict[str, Any]]:
        return list(self._correction_history)

    # ------------------------------------------------------------------
    # Editing operations
    # ------------------------------------------------------------------

    def correct_character(self, index: int, new_char: str) -> None:
        """Correct a character at the given index."""
        if index < 0 or index >= len(self._characters):
            raise IndexError(f"Character index {index} out of range")
        old_char = self._characters[index].char
        self._characters[index].correct(new_char)
        self._correction_history.append(
            {
                "index": index,
                "old_char": old_char,
                "new_char": new_char,
            }
        )
        self._rebuild_lines()

    def correct_characters(self, corrections: dict[int, str]) -> None:
        """Apply multiple corrections at once.

        Args:
            corrections: mapping from character index to new character.
        """
        for index, new_char in corrections.items():
            self.correct_character(index, new_char)

    def get_low_confidence_characters(
        self, threshold: float = 60.0
    ) -> list[tuple[int, CharacterResult]]:
        """Return characters below the confidence threshold with their indices."""
        return [
            (i, c) for i, c in enumerate(self._characters) if c.is_low_confidence(threshold)
        ]

    def get_corrected_characters(self) -> list[tuple[int, CharacterResult]]:
        """Return all corrected characters with their indices."""
        return [
            (i, c) for i, c in enumerate(self._characters) if c.corrected
        ]

    # ------------------------------------------------------------------
    # Suggestion engine
    # ------------------------------------------------------------------

    def generate_suggestions(
        self,
        confusion_pairs: dict[str, list[str]] | None = None,
    ) -> None:
        """Generate correction suggestions for low-confidence characters.

        Uses common Chinese character confusion pairs to suggest alternatives.
        """
        pairs = confusion_pairs or _default_confusion_pairs()
        for ch in self._characters:
            if ch.is_low_confidence():
                alternatives = pairs.get(ch.char, [])
                ch.suggestions = tuple(alternatives)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lines": self._lines,
            "source_path": self._source_path,
            "characters": [c.to_dict() for c in self._characters],
            "correction_history": self._correction_history,
            "statistics": self._statistics(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditableTextDocument:
        characters = [CharacterResult.from_dict(c) for c in data.get("characters", [])]
        return cls(
            characters=characters,
            lines=data.get("lines", []),
            source_path=data.get("source_path"),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> EditableTextDocument:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_lines(self) -> None:
        """Rebuild line text from current character state."""
        lines_dict: dict[int, list[str]] = {}
        for c in self._characters:
            lines_dict.setdefault(c.line_index, []).append(c.char)
        self._lines = ["".join(chs) for _, chs in sorted(lines_dict.items())]

    def _statistics(self) -> dict[str, Any]:
        total = len(self._characters)
        corrected = sum(1 for c in self._characters if c.corrected)
        low_conf = sum(1 for c in self._characters if c.is_low_confidence())
        if total > 0:
            avg_conf = sum(c.confidence for c in self._characters) / total
        else:
            avg_conf = 0.0
        return {
            "total_characters": total,
            "corrected_characters": corrected,
            "low_confidence_characters": low_conf,
            "average_confidence": round(avg_conf, 2),
        }


def _default_confusion_pairs() -> dict[str, list[str]]:
    """Common Chinese character confusion pairs for OCR suggestion."""
    return {
        "己": ["已", "巳"],
        "已": ["己", "巳"],
        "巳": ["己", "已"],
        "未": ["末"],
        "末": ["未"],
        "天": ["夫"],
        "夫": ["天"],
        "大": ["太", "犬"],
        "太": ["大", "犬"],
        "犬": ["大", "太"],
        "人": ["入", "八"],
        "入": ["人", "八"],
        "八": ["人", "入"],
        "日": ["曰", "目"],
        "曰": ["日", "目"],
        "目": ["日", "曰"],
        "田": ["由", "甲"],
        "由": ["田", "甲"],
        "甲": ["田", "由"],
        "土": ["士"],
        "士": ["土"],
        "干": ["千", "于"],
        "千": ["干", "于"],
        "于": ["干", "千"],
        "刀": ["力"],
        "力": ["刀"],
        "了": ["子"],
        "子": ["了"],
        "王": ["玉"],
        "玉": ["王"],
        "白": ["自"],
        "自": ["白"],
        "月": ["目"],
        "口": ["中"],
        "中": ["口"],
        "木": ["本", "术"],
        "本": ["木", "术"],
        "术": ["木", "本"],
        "贝": ["见"],
        "见": ["贝"],
        "不": ["下"],
        "下": ["不"],
        "出": ["击"],
        "击": ["出"],
    }


__all__ = [
    "CharacterResult",
    "EditableTextDocument",
]
