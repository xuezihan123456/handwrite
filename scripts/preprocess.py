"""Preprocessing helpers and dataset-building utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from handwrite.data.casia_parser import parse_gnt_directory
from handwrite.data.charsets import get_charset
from handwrite.data.font_renderer import render_standard_char


def is_valid_bitmap(bitmap: np.ndarray) -> bool:
    """Return whether a raw bitmap passes the coarse quality checks."""
    height, width = bitmap.shape[:2]
    if height < 32 or width < 32:
        return False
    if height > 300 or width > 300:
        return False

    mean_value = float(bitmap.mean())
    if mean_value > 250 or mean_value < 5:
        return False

    return True


def normalize_bitmap(
    bitmap: np.ndarray,
    canvas_size: int = 256,
    content_size: int = 200,
) -> np.ndarray:
    """Binarize, crop, resize, and center a handwriting bitmap."""
    source = np.asarray(bitmap, dtype=np.uint8)
    _, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = np.argwhere(binary < 255)
    if foreground.size == 0:
        return np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    top_left = foreground.min(axis=0)
    bottom_right = foreground.max(axis=0) + 1
    cropped = binary[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

    height, width = cropped.shape[:2]
    scale = min(content_size / max(height, 1), content_size / max(width, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = Image.fromarray(cropped, mode="L").resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    offset_x = (canvas_size - resized_width) // 2
    offset_y = (canvas_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


def compute_writer_coverage(
    parsed_data: dict[str, list[tuple[str, np.ndarray]]],
    charset: Iterable[str],
) -> dict[str, int]:
    """Count the unique charset hits for each writer."""
    charset_set = set(charset)
    coverage: dict[str, int] = {}

    for writer_id, samples in parsed_data.items():
        seen_chars = {char for char, _ in samples if char in charset_set}
        coverage[writer_id] = len(seen_chars)

    return coverage


def _coerce_charset(charset: Iterable[str] | None) -> list[str]:
    if charset is None:
        return []

    unique_chars: list[str] = []
    seen_chars: set[str] = set()
    for char in charset:
        if char in seen_chars:
            continue
        seen_chars.add(char)
        unique_chars.append(char)

    return unique_chars


def _is_gnt_label_char(char: str) -> bool:
    try:
        return len(char.encode("gbk")) == 2
    except UnicodeEncodeError:
        return False


def _coerce_gnt_charset(charset: Iterable[str] | None) -> list[str]:
    return [char for char in _coerce_charset(charset) if _is_gnt_label_char(char)]


def _filter_valid_samples(
    parsed_data: dict[str, list[tuple[str, np.ndarray]]],
    charset: Iterable[str] | None,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    charset_set = set(charset or [])
    restrict_charset = bool(charset_set)
    filtered: dict[str, list[tuple[str, np.ndarray]]] = {}

    for writer_id, samples in parsed_data.items():
        valid_samples: list[tuple[str, np.ndarray]] = []
        for char, bitmap in samples:
            if restrict_charset and char not in charset_set:
                continue
            if not is_valid_bitmap(bitmap):
                continue
            valid_samples.append((char, bitmap))

        filtered[writer_id] = valid_samples

    return filtered


def _writer_meets_coverage(
    covered_chars: int,
    *,
    min_writer_coverage: float | int,
    charset_size: int,
) -> bool:
    if charset_size == 0:
        return True

    if isinstance(min_writer_coverage, int) or min_writer_coverage > 1:
        required_chars = int(min_writer_coverage)
    else:
        required_chars = int(np.ceil(charset_size * float(min_writer_coverage)))

    return covered_chars >= required_chars


def _relative_metadata_path(path: Path, *, output_dir: Path) -> str:
    base_path = output_dir.resolve()
    return path.resolve().relative_to(base_path).as_posix()


def build_processed_dataset(
    parsed_data: dict[str, list[tuple[str, np.ndarray]]],
    output_dir: str | Path,
    font_path: str | Path,
    *,
    charset: Iterable[str] | None = None,
    min_writer_coverage: float | int = 0.0,
    canvas_size: int = 256,
    content_size: int = 200,
) -> dict[str, Any]:
    """Build paired standard/handwrite images and metadata from parsed samples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charset_list = _coerce_charset(charset)
    filtered_samples = _filter_valid_samples(parsed_data, charset_list or None)
    coverage = compute_writer_coverage(filtered_samples, charset_list) if charset_list else {}

    kept_writers: list[str] = []
    pairs: list[dict[str, str]] = []

    for writer_id in sorted(filtered_samples):
        writer_samples = filtered_samples[writer_id]
        if charset_list and not _writer_meets_coverage(
            coverage.get(writer_id, 0),
            min_writer_coverage=min_writer_coverage,
            charset_size=len(charset_list),
        ):
            continue
        if not writer_samples:
            continue

        writer_dir = output_path / f"writer_{writer_id}"
        writer_dir.mkdir(parents=True, exist_ok=True)
        kept_writers.append(writer_id)

        seen_chars: set[str] = set()
        for char, bitmap in writer_samples:
            if char in seen_chars:
                continue
            seen_chars.add(char)

            unicode_hex = f"{ord(char):04X}"
            handwrite_path = writer_dir / f"{unicode_hex}_handwrite.png"
            standard_path = writer_dir / f"{unicode_hex}_standard.png"

            normalized = normalize_bitmap(
                bitmap,
                canvas_size=canvas_size,
                content_size=content_size,
            )
            Image.fromarray(normalized, mode="L").save(handwrite_path)

            standard_image = render_standard_char(
                char,
                str(Path(font_path)),
                image_size=canvas_size,
                char_size=content_size,
            )
            standard_image.save(standard_path)

            pairs.append(
                {
                    "writer_id": writer_id,
                    "char": char,
                    "unicode_hex": unicode_hex,
                    "standard": _relative_metadata_path(standard_path, output_dir=output_path),
                    "handwrite": _relative_metadata_path(handwrite_path, output_dir=output_path),
                }
            )

    if charset_list:
        metadata_charset = charset_list
    else:
        metadata_charset = []
        seen_chars: set[str] = set()
        for pair in pairs:
            char = pair["char"]
            if char in seen_chars:
                continue
            seen_chars.add(char)
            metadata_charset.append(char)

    metadata = {
        "writers": kept_writers,
        "charset": metadata_charset,
        "pairs": pairs,
        "stats": {
            "total_pairs": len(pairs),
            "num_writers": len(kept_writers),
            "num_chars": len(metadata_charset),
        },
    }

    metadata_path = output_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata


def preprocess_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    font_path: str | Path,
    *,
    charset: Iterable[str] | None = None,
    min_writer_coverage: float | int = 0.9,
    canvas_size: int = 256,
    content_size: int = 200,
) -> dict[str, Any]:
    """Parse raw `.gnt` files and build the processed dataset."""
    parsed_data = parse_gnt_directory(str(Path(raw_dir)))
    gnt_charset = _coerce_gnt_charset(charset)
    return build_processed_dataset(
        parsed_data,
        Path(output_dir),
        Path(font_path),
        charset=gnt_charset,
        min_writer_coverage=min_writer_coverage,
        canvas_size=canvas_size,
        content_size=content_size,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse CASIA .gnt files and build a processed dataset.")
    parser.add_argument("--raw_dir", type=Path, required=True, help="Directory containing raw .gnt files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for processed output.")
    parser.add_argument("--font_path", type=Path, required=True, help="Font used to render standard characters.")
    parser.add_argument(
        "--charset_level",
        default="500",
        help="Named charset level to load before preprocessing.",
    )
    parser.add_argument(
        "--min_writer_coverage",
        type=float,
        default=0.9,
        help="Minimum charset coverage ratio required to keep a writer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    charset = get_charset(str(args.charset_level))
    metadata = preprocess_dataset(
        args.raw_dir,
        args.output_dir,
        args.font_path,
        charset=charset,
        min_writer_coverage=args.min_writer_coverage,
    )
    print(
        json.dumps(
            {
                "metadata_path": str((args.output_dir / "metadata.json").resolve()),
                "stats": metadata["stats"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
