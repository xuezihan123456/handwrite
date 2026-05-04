"""Export helpers for generated handwriting pages."""

from pathlib import Path
from typing import Iterable, List, Union

import cv2
import numpy as np
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


def _prepare_pdf_page(page: Image.Image) -> Image.Image:
    if page.mode == "RGB":
        return page

    if page.mode in {"RGBA", "LA"} or (page.mode == "P" and "transparency" in page.info):
        alpha_page = page.convert("RGBA")
        background = Image.new("RGB", alpha_page.size, color="white")
        background.paste(alpha_page, mask=alpha_page.getchannel("A"))
        return background

    return page.convert("RGB")


def export_pdf(page: Image.Image, output_path: PathLike, dpi: int = 300) -> Path:
    """Save a page image as a single-page PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_page = _prepare_pdf_page(page)
    pdf_page.save(path, format="PDF", resolution=float(dpi))
    return path


def export_pages_pdf(
    pages: Iterable[Image.Image],
    output_path: PathLike,
    dpi: int = 300,
) -> Path:
    """Save multiple page images as a multi-page PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_pages = [_prepare_pdf_page(page) for page in pages]
    if not pdf_pages:
        raise ValueError("pages must contain at least one page")

    first_page, *remaining_pages = pdf_pages
    first_page.save(
        path,
        format="PDF",
        save_all=True,
        append_images=remaining_pages,
        resolution=float(dpi),
    )
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


def export_animation(
    frames: Iterable[Image.Image],
    output_path: PathLike,
    format: str = "gif",
    fps: int = 30,
    loop: int = 0,
) -> Path:
    """Export animation frames as GIF or MP4.

    Args:
        frames: Iterable of animation frame images.
        output_path: Destination file path.
        format: Output format, either 'gif' or 'mp4'.
        fps: Frames per second.
        loop: Number of loops for GIF (0 = infinite).

    Returns:
        Path to the written file.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("frames must contain at least one frame")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_format = format.lower()
    if normalized_format == "gif":
        return _export_animation_gif(frame_list, path, fps, loop)
    if normalized_format == "mp4":
        return _export_animation_mp4(frame_list, path, fps)

    raise ValueError("format must be 'gif' or 'mp4'")


def _export_animation_gif(
    frames: List[Image.Image],
    path: Path,
    fps: int,
    loop: int,
) -> Path:
    """Save animation frames as an animated GIF using Pillow."""
    duration_ms = max(1, int(1000 / fps))

    # Convert frames to palette mode for GIF compatibility
    gif_frames = []
    for frame in frames:
        rgb = _prepare_pdf_page(frame) if frame.mode != "RGB" else frame
        gif_frames.append(rgb.convert("P", palette=Image.Palette.ADAPTIVE))

    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    return path


def _export_animation_mp4(
    frames: List[Image.Image],
    path: Path,
    fps: int,
) -> Path:
    """Save animation frames as an MP4 video using OpenCV."""
    if not frames:
        raise ValueError("frames must contain at least one frame")

    width, height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height), True)

    try:
        for frame in frames:
            rgb = _prepare_pdf_page(frame) if frame.mode != "RGB" else frame
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    return path
