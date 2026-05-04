"""High-level animation engine for handwriting animation generation.

Orchestrates the full pipeline: character generation -> stroke extraction ->
trajectory generation -> frame rendering -> animation export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from .animation_composer import compose_text_animation
from .frame_renderer import render_animation_frames
from .stroke_order import extract_strokes
from .trajectory_generator import generate_trajectories

PathLike = Union[str, Path]

# Default animation parameters
_DEFAULT_FPS = 30
_DEFAULT_CHAR_DURATION = 1.0  # seconds
_DEFAULT_CHAR_SIZE = 256
_DEFAULT_BRUSH_SIZE = 3
_DEFAULT_INK_DIFFUSION = 1.5
_DEFAULT_SAMPLES_PER_STROKE = 40
_DEFAULT_INTER_CHAR_DELAY = 0.2  # seconds


def generate_char_animation(
    char: str,
    style: str = "行书流畅",
    fps: int = _DEFAULT_FPS,
    duration: float = _DEFAULT_CHAR_DURATION,
    char_size: int = _DEFAULT_CHAR_SIZE,
    brush_size: int = _DEFAULT_BRUSH_SIZE,
    ink_diffusion: float = _DEFAULT_INK_DIFFUSION,
    prototype_pack: str | Path | None = None,
) -> list[Image.Image]:
    """Generate a handwriting animation for a single character.

    Args:
        char: The character to animate (single character).
        style: Handwriting style name.
        fps: Frames per second.
        duration: Animation duration in seconds (0.5 to 2.0).
        char_size: Output character size in pixels.
        brush_size: Pen brush thickness.
        ink_diffusion: Ink diffusion effect radius.
        prototype_pack: Optional custom prototype pack path.

    Returns:
        List of PIL Images representing the animation frames.
    """
    from handwrite import _get_engine, BUILTIN_STYLES

    if len(char) != 1:
        raise ValueError(f"Expected a single character, got {char!r}")

    duration = max(0.5, min(2.0, duration))
    num_frames = max(2, int(fps * duration))

    # Generate the target character image
    style_id = BUILTIN_STYLES.get(style, 0)
    engine = _get_engine(prototype_pack=prototype_pack)
    char_image = engine.generate_char(char, style_id)

    # Resize to target size
    if char_image.size != (char_size, char_size):
        char_image = char_image.resize(
            (char_size, char_size), Image.Resampling.LANCZOS
        )

    return _animate_char_image(
        char_image,
        num_frames=num_frames,
        canvas_size=(char_size, char_size),
        brush_size=brush_size,
        ink_diffusion=ink_diffusion,
    )


def generate_text_animation(
    text: str,
    style: str = "行书流畅",
    fps: int = _DEFAULT_FPS,
    char_duration: float = _DEFAULT_CHAR_DURATION,
    char_size: int = _DEFAULT_CHAR_SIZE,
    chars_per_line: int = 8,
    brush_size: int = _DEFAULT_BRUSH_SIZE,
    ink_diffusion: float = _DEFAULT_INK_DIFFUSION,
    inter_char_delay: float = _DEFAULT_INTER_CHAR_DELAY,
    prototype_pack: str | Path | None = None,
) -> list[Image.Image]:
    """Generate a handwriting animation for a text string.

    Characters are animated sequentially in reading order.

    Args:
        text: The text to animate.
        style: Handwriting style name.
        fps: Frames per second.
        char_duration: Duration per character in seconds.
        char_size: Character cell size in pixels.
        chars_per_line: Maximum characters per line.
        brush_size: Pen brush thickness.
        ink_diffusion: Ink diffusion effect radius.
        inter_char_delay: Delay between characters in seconds.
        prototype_pack: Optional custom prototype pack path.

    Returns:
        List of PIL Images representing the composed animation frames.
    """
    from handwrite import _get_engine, BUILTIN_STYLES

    characters = [c for c in text if not c.isspace()]
    if not characters:
        blank = Image.new("L", (char_size, char_size), color=255)
        return [blank]

    style_id = BUILTIN_STYLES.get(style, 0)
    engine = _get_engine(prototype_pack=prototype_pack)
    char_duration = max(0.5, min(2.0, char_duration))
    num_frames_per_char = max(2, int(fps * char_duration))
    inter_char_delay_frames = max(0, int(fps * inter_char_delay))

    # Generate animation for each character
    char_animations: list[list[Image.Image]] = []

    for char in characters:
        char_image = engine.generate_char(char, style_id)
        if char_image.size != (char_size, char_size):
            char_image = char_image.resize(
                (char_size, char_size), Image.Resampling.LANCZOS
            )

        frames = _animate_char_image(
            char_image,
            num_frames=num_frames_per_char,
            canvas_size=(char_size, char_size),
            brush_size=brush_size,
            ink_diffusion=ink_diffusion,
        )
        char_animations.append(frames)

    # Compose into text animation
    return compose_text_animation(
        char_animations,
        char_size=char_size,
        char_gap=max(4, char_size // 8),
        line_gap=max(8, char_size // 4),
        chars_per_line=chars_per_line,
        inter_char_delay_frames=inter_char_delay_frames,
    )


def export_animation(
    frames: list[Image.Image],
    output_path: PathLike,
    format: str = "gif",
    fps: int = _DEFAULT_FPS,
    loop: int = 0,
) -> Path:
    """Export animation frames as GIF or MP4.

    Args:
        frames: List of animation frames.
        output_path: Output file path.
        format: Output format ('gif' or 'mp4').
        fps: Frames per second.
        loop: Number of loops for GIF (0 = infinite).

    Returns:
        Path to the exported file.
    """
    if not frames:
        raise ValueError("frames must contain at least one frame")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_format = format.lower()

    if normalized_format == "gif":
        return _export_gif(frames, path, fps, loop)
    elif normalized_format == "mp4":
        return _export_mp4(frames, path, fps)
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'gif' or 'mp4'.")


def _animate_char_image(
    char_image: Image.Image,
    num_frames: int,
    canvas_size: tuple[int, int],
    brush_size: int,
    ink_diffusion: float,
) -> list[Image.Image]:
    """Run the animation pipeline for a single character image."""
    # Step 1: Extract strokes
    strokes = extract_strokes(char_image)

    if not strokes:
        # Fallback: if no strokes detected, use simple reveal
        return _simple_reveal_frames(char_image, num_frames, canvas_size)

    # Step 2: Generate trajectories
    trajectories = generate_trajectories(strokes)

    if not trajectories:
        return _simple_reveal_frames(char_image, num_frames, canvas_size)

    # Step 3: Render frames
    frames = render_animation_frames(
        char_image,
        trajectories,
        num_frames,
        canvas_size,
        brush_size=brush_size,
        ink_diffusion=ink_diffusion,
    )

    return frames


def _simple_reveal_frames(
    char_image: Image.Image,
    num_frames: int,
    canvas_size: tuple[int, int],
) -> list[Image.Image]:
    """Fallback animation: simple left-to-right reveal when no strokes detected."""
    width, height = canvas_size
    target = np.array(char_image.convert("L").resize(canvas_size), dtype=np.float32)
    frames: list[Image.Image] = []

    for i in range(num_frames):
        progress = i / max(1, num_frames - 1)

        # Create a left-to-right reveal mask
        mask = np.zeros((height, width), dtype=np.float32)
        reveal_x = int(progress * width)
        if reveal_x > 0:
            # Smooth edge
            edge_width = max(1, width // 20)
            for x in range(min(reveal_x + edge_width, width)):
                if x < reveal_x:
                    mask[:, x] = 1.0
                else:
                    mask[:, x] = 1.0 - (x - reveal_x) / edge_width

        white = np.full_like(target, 255.0)
        result = white * (1.0 - mask) + target * mask
        frames.append(Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L"))

    return frames


def _export_gif(
    frames: list[Image.Image],
    path: Path,
    fps: int,
    loop: int,
) -> Path:
    """Save animation frames as an animated GIF."""
    duration_ms = max(1, int(1000 / fps))

    gif_frames: list[Image.Image] = []
    for frame in frames:
        if frame.mode not in ("RGB", "P"):
            frame = frame.convert("RGB")
        gif_frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    return path


def _export_mp4(
    frames: list[Image.Image],
    path: Path,
    fps: int,
) -> Path:
    """Save animation frames as an MP4 video."""
    width, height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height), True)

    try:
        for frame in frames:
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    return path


__all__ = [
    "generate_char_animation",
    "generate_text_animation",
    "export_animation",
]
