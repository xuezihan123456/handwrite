"""Build a prototype glyph library from local processed handwriting metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


def build_prototype_library(
    *,
    metadata_path: str | Path,
    output_dir: str | Path,
    writer_id: str | None = None,
    pack_name: str = "prototype_pack",
) -> dict[str, Any]:
    metadata_file = Path(metadata_path)
    output_root = Path(output_dir)
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])

    chosen_by_char: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        pair_writer_id = pair.get("writer_id")
        char = pair.get("char")
        handwrite_path = pair.get("handwrite")
        if not char or not handwrite_path:
            continue
        if writer_id is not None and pair_writer_id != writer_id:
            continue
        if char in chosen_by_char:
            continue
        chosen_by_char[char] = {
            "writer_id": pair_writer_id,
            "handwrite_path": handwrite_path,
        }

    glyph_output_dir = output_root / "glyphs"
    glyph_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_glyphs: list[dict[str, str]] = []
    copied_chars: list[str] = []

    for char in sorted(chosen_by_char):
        source_path = metadata_file.parent / chosen_by_char[char]["handwrite_path"]
        unicode_hex = f"U{ord(char):04X}"
        destination_path = glyph_output_dir / f"{unicode_hex}.png"
        shutil.copyfile(source_path, destination_path)
        copied_chars.append(char)
        manifest_glyphs.append(
            {
                "char": char,
                "file": f"glyphs/{unicode_hex}.png",
                "writer_id": str(chosen_by_char[char]["writer_id"]),
            }
        )

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
        "writer_id": writer_id,
        "glyph_count": len(manifest_glyphs),
        "chars": copied_chars,
        "manifest_path": str(manifest_path),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a prototype glyph pack from processed handwriting metadata."
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.json.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output pack directory.")
    parser.add_argument("--writer_id", type=str, default=None, help="Optional writer filter.")
    parser.add_argument("--pack_name", type=str, default="prototype_pack")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    summary = build_prototype_library(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        writer_id=args.writer_id,
        pack_name=args.pack_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
