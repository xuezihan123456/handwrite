"""Generate a prototype glyph library from segmented characters.

Takes segmented character images (optionally with OCR labels),
normalizes them to 256x256, and writes a manifest.json compatible
with the existing PrototypeLibrary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from handwrite.ocr_style.character_segmenter import CharBox


@dataclass(frozen=True)
class GlyphEntry:
    """A single glyph ready to be saved."""

    char: str
    image: np.ndarray  # 256x256 grayscale, white ink on black
    writer_id: str = "scan"


class PrototypeGenerator:
    """Generate a prototype glyph pack from segmented characters."""

    def __init__(
        self,
        *,
        glyph_size: int = 256,
        padding_ratio: float = 0.1,
        writer_id: str = "scan",
    ) -> None:
        """
        Parameters
        ----------
        glyph_size:
            Output size for each glyph image (glyph_size x glyph_size).
        padding_ratio:
            Fraction of glyph_size to use as padding around the character.
        writer_id:
            Writer identifier stored in the manifest.
        """
        self.glyph_size = glyph_size
        self.padding_ratio = padding_ratio
        self.writer_id = writer_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        chars: Sequence[CharBox],
        labels: Sequence[str] | None,
        output_dir: str | os.PathLike[str],
        *,
        pack_name: str = "scan_pack",
    ) -> dict[str, Any]:
        """Generate a prototype pack from segmented characters.

        Parameters
        ----------
        chars:
            Segmented character boxes with cropped images.
        labels:
            Optional character labels (e.g., from OCR).  If provided,
            must have the same length as *chars*.  When *None*,
            characters are labelled by their position index.
        output_dir:
            Directory to write the pack (glyphs/ subdirectory and manifest).
        pack_name:
            Name stored in manifest.json.

        Returns
        -------
        dict
            Summary with keys: ``pack_name``, ``glyph_count``, ``chars``,
            ``manifest_path``.
        """
        output_root = Path(output_dir)
        glyph_dir = output_root / "glyphs"
        glyph_dir.mkdir(parents=True, exist_ok=True)

        entries: list[GlyphEntry] = []
        seen_chars: set[str] = set()

        for idx, cb in enumerate(chars):
            # Determine the character label
            if labels is not None and idx < len(labels):
                char = labels[idx]
            else:
                char = f"_pos{idx}"

            # Skip duplicates (keep the first occurrence)
            if char in seen_chars:
                continue
            seen_chars.add(char)

            # Normalize the image
            normalized = self._normalize(cb.image)
            entries.append(GlyphEntry(
                char=char,
                image=normalized,
                writer_id=self.writer_id,
            ))

        # Save glyphs and build manifest
        manifest_glyphs: list[dict[str, str]] = []
        for entry in entries:
            if len(entry.char) == 1:
                code = f"U{ord(entry.char):04X}"
            else:
                code = entry.char
            filename = f"{code}.png"
            filepath = glyph_dir / filename
            cv2.imwrite(str(filepath), entry.image)
            manifest_glyphs.append({
                "char": entry.char,
                "file": f"glyphs/{filename}",
                "writer_id": entry.writer_id,
            })

        manifest = {
            "name": pack_name,
            "glyphs": manifest_glyphs,
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "pack_name": pack_name,
            "glyph_count": len(manifest_glyphs),
            "chars": [e.char for e in entries],
            "manifest_path": str(manifest_path),
        }

    # ------------------------------------------------------------------
    # Image normalization
    # ------------------------------------------------------------------

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize a character crop to glyph_size x glyph_size.

        Steps:
          1. Find the bounding box of the ink pixels.
          2. Crop to the ink region.
          3. Scale to fit within the padded area while preserving aspect ratio.
          4. Center on a white-background canvas.
        """
        size = self.glyph_size
        pad = int(size * self.padding_ratio)
        usable = size - 2 * pad

        if img.size == 0:
            return np.zeros((size, size), dtype=np.uint8)

        # Find ink bounding box
        coords = cv2.findNonZero(img)
        if coords is None:
            return np.zeros((size, size), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return np.zeros((size, size), dtype=np.uint8)

        cropped = img[y:y + h, x:x + w]

        # Scale to fit
        scale = min(usable / w, usable / h)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        resized = cv2.resize(
            cropped, (new_w, new_h), interpolation=cv2.INTER_AREA,
        )

        # Center on canvas
        canvas = np.zeros((size, size), dtype=np.uint8)
        x_off = (size - new_w) // 2
        y_off = (size - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return canvas


__all__ = ["PrototypeGenerator", "GlyphEntry"]
