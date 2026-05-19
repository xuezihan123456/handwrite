"""Persist segmented character glyphs as a HandWrite prototype pack.

Writes a ``manifest.json`` + ``glyphs/<U-codepoint>.png`` directory layout
compatible with :class:`handwrite.prototypes.PrototypeLibrary`.

The builder consumes a list of :class:`SegmentedChar` records (a thin,
serialisable struct independent of the OCR backend) so the pipeline can
plug in either Tesseract output, easyocr output, or purely heuristic
segmentation. Synthetic positional labels (``_pos0``, ``_pos1`` ...) are
used when an OCR label is not provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


_DEFAULT_GLYPH_SIZE = 256
_DEFAULT_PADDING_RATIO = 0.1


@dataclass(frozen=True)
class SegmentedChar:
    """Single segmented character ready for export.

    Attributes:
        image: Cropped greyscale array. Either dark ink on light paper
            (default) or white ink on black (set ``ink_is_white=True``).
        bbox: ``(x, y, w, h)`` location in the source image.
        label: Optional character label (e.g. from OCR). When ``None`` a
            positional pseudo-label is generated.
        confidence: OCR confidence (0..100). Higher wins on dedup.
        ink_is_white: True if foreground is white on black background.
        writer_id: Identifier stored in the manifest entry.
    """

    image: np.ndarray
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    label: Optional[str] = None
    confidence: float = 0.0
    ink_is_white: bool = False
    writer_id: Optional[str] = None


@dataclass(frozen=True)
class GlyphPackResult:
    """Summary of a generated pack."""

    pack_name: str
    manifest_path: Path
    glyph_count: int
    chars: tuple[str, ...] = field(default_factory=tuple)


class GlyphBuilder:
    """Build a prototype pack from segmented characters."""

    def __init__(
        self,
        *,
        glyph_size: int = _DEFAULT_GLYPH_SIZE,
        padding_ratio: float = _DEFAULT_PADDING_RATIO,
        writer_id: str = "photo_capture",
    ) -> None:
        if glyph_size <= 0:
            raise ValueError("glyph_size must be positive")
        if not 0.0 <= padding_ratio < 0.5:
            raise ValueError("padding_ratio must be in [0.0, 0.5)")
        self._glyph_size = glyph_size
        self._padding_ratio = padding_ratio
        self._writer_id = writer_id

    @property
    def glyph_size(self) -> int:
        return self._glyph_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        chars: Sequence[SegmentedChar],
        output_dir: str | Path,
        *,
        pack_name: str = "my_handwriting",
        source: str = "photo_style",
    ) -> GlyphPackResult:
        """Write *chars* as a HandWrite prototype pack.

        Args:
            chars: Iterable of segmented characters.
            output_dir: Destination directory; will be created.
            pack_name: Pack name stored inside ``manifest.json``.
            source: Free-form provenance string for the manifest.

        Returns:
            :class:`GlyphPackResult` with paths and counts.
        """
        out_root = Path(output_dir)
        glyph_dir = out_root / "glyphs"
        glyph_dir.mkdir(parents=True, exist_ok=True)

        chosen = self._deduplicate(chars)
        manifest_entries: list[dict[str, object]] = []
        saved_chars: list[str] = []

        for index, item in enumerate(chosen):
            label = item.label or f"_pos{index}"
            normalised = self._normalise(item)
            file_name = _safe_glyph_filename(label, index)
            target_path = glyph_dir / file_name
            normalised.save(str(target_path))
            manifest_entries.append(
                {
                    "char": label,
                    "file": f"glyphs/{file_name}",
                    "writer_id": item.writer_id or self._writer_id,
                    "confidence": round(float(item.confidence), 2),
                    "bbox": [int(v) for v in item.bbox],
                }
            )
            saved_chars.append(label)

        manifest = {
            "name": pack_name,
            "version": 1,
            "source": source,
            "normalize_size": self._glyph_size,
            "glyph_count": len(manifest_entries),
            "glyphs": manifest_entries,
        }
        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return GlyphPackResult(
            pack_name=pack_name,
            manifest_path=manifest_path,
            glyph_count=len(manifest_entries),
            chars=tuple(saved_chars),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(
        self, chars: Iterable[SegmentedChar]
    ) -> list[SegmentedChar]:
        """Drop empty crops and keep the highest-confidence per label."""
        kept_by_label: dict[str, SegmentedChar] = {}
        positional: list[SegmentedChar] = []
        for item in chars:
            if item.image is None or item.image.size == 0:
                continue
            label = item.label
            if not label:
                positional.append(item)
                continue
            existing = kept_by_label.get(label)
            if existing is None or item.confidence > existing.confidence:
                kept_by_label[label] = item
        ordered = list(kept_by_label.values()) + positional
        return ordered

    def _normalise(self, item: SegmentedChar) -> Image.Image:
        """Centre + resize the character on a white 256x256 canvas."""
        target = self._glyph_size
        pad = int(target * self._padding_ratio)
        usable = max(target - 2 * pad, 1)

        crop = item.image
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = crop.astype(np.uint8)

        # Ensure ink is white on black for content detection.
        if item.ink_is_white:
            ink_white = crop
        else:
            ink_white = cv2.bitwise_not(crop)

        coords = cv2.findNonZero(ink_white)
        if coords is None:
            return Image.fromarray(
                np.ones((target, target), dtype=np.uint8) * 255, mode="L"
            )
        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return Image.fromarray(
                np.ones((target, target), dtype=np.uint8) * 255, mode="L"
            )

        cropped = ink_white[y : y + h, x : x + w]
        scale = min(usable / max(w, 1), usable / max(h, 1))
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        resized = cv2.resize(
            cropped, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        # Output canvas is white paper, dark ink -> invert back at the end.
        canvas_ink = np.zeros((target, target), dtype=np.uint8)
        x_off = (target - new_w) // 2
        y_off = (target - new_h) // 2
        canvas_ink[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        final = cv2.bitwise_not(canvas_ink)
        return Image.fromarray(final, mode="L")


def _safe_glyph_filename(label: str, index: int) -> str:
    """Produce a filesystem-safe glyph filename for *label*."""
    if not label:
        return f"pos_{index:04d}.png"
    if label.startswith("_pos"):
        return f"{label}.png"
    if len(label) == 1:
        return f"U{ord(label):04X}.png"
    safe = "_".join(f"U{ord(ch):04X}" for ch in label)
    return f"{safe}.png"


__all__ = [
    "GlyphBuilder",
    "GlyphPackResult",
    "SegmentedChar",
]
