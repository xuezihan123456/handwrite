"""Custom paper loader: loads paper definitions from JSON files or images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_paper_json(path: str | Path) -> dict[str, Any]:
    """Load a paper definition from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Paper definition dict with at least 'name', 'size', and 'regions' keys.
    """
    json_path = Path(path)
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    if "name" not in raw:
        raw["name"] = json_path.stem
    if "size" not in raw:
        raw["size"] = [2480, 3508]
    if "regions" not in raw:
        raw["regions"] = []

    raw["_source"] = str(json_path)
    return raw


def load_paper_image(
    path: str | Path,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a paper definition backed by a background image.

    The image will be used as-is by the renderer instead of drawing regions.

    Args:
        path: Path to the image file.
        name: Optional paper name. Defaults to the file stem.

    Returns:
        Paper definition dict with an '_image' key.
    """
    img_path = Path(path)
    img = Image.open(img_path)
    return {
        "name": name or img_path.stem,
        "size": [img.width, img.height],
        "regions": [],
        "_image": str(img_path),
    }


def save_paper_json(definition: dict[str, Any], path: str | Path) -> None:
    """Save a paper definition dict to a JSON file.

    Internal keys (starting with '_') are excluded from the output.
    """
    out_path = Path(path)
    sanitized = {k: v for k, v in definition.items() if not k.startswith("_")}
    out_path.write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
