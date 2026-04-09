"""Export helpers for generated handwriting pages."""

from pathlib import Path
from typing import Iterable, List, Union

from PIL import Image


PathLike = Union[str, Path]
_INVALID_PREFIX_CHARS = '<>:"|?*'


def _sanitize_prefix(prefix: str) -> str:
    normalized = prefix.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", ".", ".."}]
    cleaned_parts = [
        "".join("_" if char in _INVALID_PREFIX_CHARS or ord(char) < 32 else char for char in part)
        for part in parts
    ]
    safe_prefix = "_".join(part for part in cleaned_parts if part)
    return safe_prefix or "page"


def export_png(page: Image.Image, output_path: PathLike, dpi: int = 300) -> Path:
    """Save a page image as PNG with embedded DPI metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, format="PNG", dpi=(dpi, dpi))
    return path


def export_pages_png(
    pages: Iterable[Image.Image],
    output_dir: PathLike,
    prefix: str = "page",
    dpi: int = 300,
) -> List[Path]:
    """Save multiple pages as sequentially numbered PNG files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = _sanitize_prefix(prefix)
    resolved_directory = directory.resolve(strict=False)

    written_paths: List[Path] = []
    for index, page in enumerate(pages, start=1):
        filename = f"{safe_prefix}_{index:03d}.png"
        output_path = directory / filename
        if output_path.resolve(strict=False).parent != resolved_directory:
            raise ValueError("prefix must resolve to a file inside output_dir")
        written_paths.append(export_png(page, output_path, dpi=dpi))

    return written_paths
