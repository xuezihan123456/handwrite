"""Utilities for parsing CASIA-HWDB `.gnt` files."""

from pathlib import Path
import struct

import numpy as np
from PIL import Image


def parse_gnt_file(gnt_path: str) -> list[tuple[str, np.ndarray]]:
    """Parse a single `.gnt` file into `(character, grayscale_bitmap)` pairs."""
    samples: list[tuple[str, np.ndarray]] = []
    path = Path(gnt_path)

    with path.open("rb") as handle:
        while True:
            size_bytes = handle.read(4)
            if not size_bytes:
                break
            if len(size_bytes) != 4:
                break

            sample_size = struct.unpack("<I", size_bytes)[0]
            payload = handle.read(sample_size - 4)
            if len(payload) != sample_size - 4:
                break

            label_bytes = payload[:2]
            width = struct.unpack("<H", payload[2:4])[0]
            height = struct.unpack("<H", payload[4:6])[0]
            bitmap = payload[6:]

            if len(bitmap) != width * height:
                continue

            try:
                char = label_bytes.decode("gbk")
            except UnicodeDecodeError:
                continue

            image = np.frombuffer(bitmap, dtype=np.uint8).reshape((height, width)).copy()
            samples.append((char, image))

    return samples


def parse_gnt_directory(dir_path: str) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Parse all `.gnt` files in a directory grouped by writer id."""
    base_path = Path(dir_path)
    parsed: dict[str, list[tuple[str, np.ndarray]]] = {}

    for gnt_path in sorted(base_path.glob("*.gnt")):
        parsed[gnt_path.stem] = parse_gnt_file(str(gnt_path))

    return parsed


def save_parsed_images(parsed_data: dict[str, list[tuple[str, np.ndarray]]], output_dir: str) -> None:
    """Save parsed bitmap arrays as PNG files grouped by writer id."""
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for writer_id, samples in parsed_data.items():
        writer_dir = base_path / f"writer_{writer_id}"
        writer_dir.mkdir(parents=True, exist_ok=True)

        for index, (char, bitmap) in enumerate(samples):
            output_path = writer_dir / f"{index:05d}_{ord(char):04X}.png"
            Image.fromarray(bitmap, mode="L").save(output_path)
