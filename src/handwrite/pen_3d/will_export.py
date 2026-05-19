"""Lightweight Wacom WILL-like JSON serialization for 3D pen strokes.

This module emits / consumes a small JSON schema inspired by Wacom's
WILL stroke format. We do **not** depend on the proprietary Wacom SDK;
the goal is a portable JSON envelope that captures the same essential
fields (per-sample x, y, t, pressure, tilt, rotation, velocity).

The top-level schema is::

    {
      "format": "will-lite",
      "version": "1.0",
      "strokes": [
        {
          "metadata": {...},
          "samples": [
            {"x": ..., "y": ..., "t": ..., "pressure": ...,
             "tilt_x": ..., "tilt_y": ..., "rotation": ...,
             "velocity": ...},
            ...
          ]
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .samples import PenSample3D, PenStroke3D


FORMAT_NAME = "will-lite"
FORMAT_VERSION = "1.0"


def export_will_json(strokes: Sequence[PenStroke3D]) -> dict:
    """Serialize a sequence of ``PenStroke3D`` into a WILL-lite dictionary.

    Args:
        strokes: Iterable of ``PenStroke3D`` to serialize. Order is
            preserved.

    Returns:
        A plain ``dict`` ready to be passed to ``json.dumps`` /
        ``json.dump``.
    """
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "strokes": [stroke.to_dict() for stroke in strokes],
    }


def import_will_json(data: dict) -> list[PenStroke3D]:
    """Deserialize a WILL-lite dictionary back to ``PenStroke3D`` objects.

    Args:
        data: Dictionary produced by ``export_will_json`` (or another
            tool emitting the same schema).

    Returns:
        List of ``PenStroke3D``.

    Raises:
        ValueError: If ``data`` is missing required keys or uses an
            unsupported format.
    """
    if not isinstance(data, dict):
        raise ValueError("WILL data must be a dict")
    fmt = data.get("format")
    if fmt != FORMAT_NAME:
        raise ValueError(
            f"Unsupported WILL format: {fmt!r}; expected {FORMAT_NAME!r}"
        )
    strokes_data = data.get("strokes")
    if strokes_data is None or not isinstance(strokes_data, list):
        raise ValueError("WILL data must contain a 'strokes' list")

    return [PenStroke3D.from_dict(s) for s in strokes_data]


def save_will_file(strokes: Sequence[PenStroke3D], path: str | Path) -> Path:
    """Persist strokes to a UTF-8 JSON file at ``path``.

    Args:
        strokes: Strokes to serialize.
        path: Output file path; parent directories are created.

    Returns:
        Resolved ``Path`` to the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(export_will_json(strokes), fh, ensure_ascii=False, indent=2)
    return out_path


def load_will_file(path: str | Path) -> list[PenStroke3D]:
    """Load strokes from a UTF-8 JSON file at ``path``."""
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return import_will_json(data)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "export_will_json",
    "import_will_json",
    "save_will_file",
    "load_will_file",
]
