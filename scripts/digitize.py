#!/usr/bin/env python3
"""CLI tool for handwriting digitization.

Usage:
    python scripts/digitize.py <image_path> [options]

Examples:
    # Basic recognition
    python scripts/digitize.py scan.png

    # With style extraction
    python scripts/digitize.py scan.png --save-glyphs ./extracted_style

    # With custom OCR backend
    python scripts/digitize.py scan.png --backend easyocr

    # Full round-trip
    python scripts/digitize.py scan.png --round-trip --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Digitize handwritten scans into editable text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the scanned handwriting image",
    )
    parser.add_argument(
        "--backend",
        choices=["tesseract", "easyocr"],
        default="tesseract",
        help="OCR backend to use (default: tesseract)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["chi_sim", "eng"],
        help="OCR languages (default: chi_sim eng)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (0-100, default: 0)",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable skew correction",
    )
    parser.add_argument(
        "--binarize",
        choices=["otsu", "adaptive", "fixed"],
        default="otsu",
        help="Binarization method (default: otsu)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for JSON results (default: stdout)",
    )
    parser.add_argument(
        "--save-glyphs",
        type=Path,
        default=None,
        help="Save extracted character glyphs as a prototype pack to this directory",
    )
    parser.add_argument(
        "--pack-name",
        default="extracted_style",
        help="Name for the extracted prototype pack (default: extracted_style)",
    )
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="Run the full round-trip pipeline (scan -> recognize -> regenerate)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for round-trip results",
    )
    parser.add_argument(
        "--style",
        default="行书流畅",
        help="Handwriting style for regeneration (default: 行书流畅)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=80,
        help="Font size for regeneration (default: 80)",
    )

    args = parser.parse_args(argv)

    # Validate input
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return 1

    # Add src to path for development usage
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from handwrite.digitization.handwriting_recognizer import (
        HandwritingRecognizer,
        OCRBackend,
        OCRConfig,
    )
    from handwrite.digitization.style_preserver import StylePreserver

    # Build OCR config
    config = OCRConfig(
        backend=OCRBackend(args.backend),
        languages=tuple(args.languages),
        confidence_threshold=args.confidence_threshold,
        binarize_method=args.binarize,
        deskew_enabled=not args.no_deskew,
    )

    if args.round_trip:
        return _run_round_trip(args, config)
    else:
        return _run_recognition(args, config)


def _run_recognition(args: argparse.Namespace, config) -> int:
    """Run recognition only, optionally extracting glyphs."""
    from handwrite.digitization.handwriting_recognizer import HandwritingRecognizer
    from handwrite.digitization.style_preserver import StylePreserver

    recognizer = HandwritingRecognizer(config)
    result = recognizer.recognize(args.image)

    # Build output data
    output_data = {
        "text": result.text,
        "lines": list(result.lines),
        "average_confidence": result.average_confidence,
        "processing_time_ms": result.processing_time_ms,
        "total_characters": len(result.characters),
        "characters": [
            {
                "char": c.char,
                "confidence": round(c.confidence, 2),
                "bbox": list(c.bbox),
                "line_index": c.line_index,
            }
            for c in result.characters
        ],
        "metadata": result.metadata,
    }

    # Output results
    output_json = json.dumps(output_data, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Save glyphs if requested
    if args.save_glyphs:
        preserver = StylePreserver()
        glyphs = preserver.extract_deduplicated_glyphs(str(args.image), result)
        if glyphs:
            pack_path = preserver.save_as_prototype_pack(
                glyphs, args.save_glyphs, args.pack_name
            )
            print(f"Prototype pack saved to: {pack_path}", file=sys.stderr)
            print(f"  Glyph count: {len(glyphs)}", file=sys.stderr)
        else:
            print("No glyphs extracted (try lowering --confidence-threshold)", file=sys.stderr)

    # Print summary to stderr
    print(f"\n--- Recognition Summary ---", file=sys.stderr)
    print(f"Text length: {len(result.text)} characters", file=sys.stderr)
    print(f"Average confidence: {result.average_confidence:.1f}%", file=sys.stderr)
    print(f"Processing time: {result.processing_time_ms:.1f}ms", file=sys.stderr)

    low_conf = sum(1 for c in result.characters if c.confidence < 60)
    if low_conf > 0:
        print(f"Low-confidence characters (<60%): {low_conf}", file=sys.stderr)

    return 0


def _run_round_trip(args: argparse.Namespace, config) -> int:
    """Run the full round-trip pipeline."""
    from handwrite.digitization.round_trip_engine import RoundTripEngine

    engine = RoundTripEngine(ocr_config=config)

    output_dir = args.output_dir or Path("./round_trip_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running round-trip pipeline...", file=sys.stderr)
    result = engine.round_trip(
        image=args.image,
        output_dir=output_dir,
        pack_name=args.pack_name,
        style=args.style,
        font_size=args.font_size,
    )

    # Save recognition results
    doc_path = output_dir / "recognition.json"
    result.document.save_json(doc_path)
    print(f"Recognition results: {doc_path}", file=sys.stderr)

    # Save original image reference
    if result.original_image:
        orig_path = output_dir / "original.png"
        result.original_image.save(str(orig_path))
        print(f"Original image: {orig_path}", file=sys.stderr)

    # Save regenerated image
    if result.regenerated_image:
        regen_path = output_dir / "regenerated.png"
        result.regenerated_image.save(str(regen_path))
        print(f"Regenerated image: {regen_path}", file=sys.stderr)

    # Print summary
    print(f"\n--- Round-Trip Summary ---", file=sys.stderr)
    print(f"Recognized text: {result.document.text[:100]}...", file=sys.stderr)
    print(f"Extracted glyphs: {len(result.extracted_glyphs)}", file=sys.stderr)
    if result.prototype_pack_path:
        print(f"Prototype pack: {result.prototype_pack_path}", file=sys.stderr)

    stats = result.document.to_dict().get("statistics", {})
    print(f"Average confidence: {stats.get('average_confidence', 0):.1f}%", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
