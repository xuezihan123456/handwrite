"""Multi-character animation composition.

Combines individual character animations into a coherent text animation
with proper spacing and timing.
"""

from __future__ import annotations

from PIL import Image


def compose_text_animation(
    char_animations: list[list[Image.Image]],
    char_size: int = 256,
    char_gap: int = 20,
    line_gap: int = 40,
    chars_per_line: int = 10,
    inter_char_delay_frames: int = 5,
) -> list[Image.Image]:
    """Compose multiple character animations into a text animation.

    Characters are laid out in reading order (left-to-right, top-to-bottom).
    Each character starts animating after the previous one finishes
    (plus a small delay).

    Args:
        char_animations: List of per-character frame lists.
        char_size: Size of each character cell in pixels.
        char_gap: Horizontal gap between characters.
        line_gap: Vertical gap between lines.
        chars_per_line: Maximum characters per line.
        inter_char_delay_frames: Extra blank frames between characters.

    Returns:
        List of composed animation frames showing all characters.
    """
    if not char_animations:
        return [Image.new("L", (char_size, char_size), color=255)]

    n_chars = len(char_animations)
    n_lines = (n_chars + chars_per_line - 1) // chars_per_line

    # Calculate canvas size
    canvas_width = chars_per_line * char_size + (chars_per_line - 1) * char_gap
    canvas_height = n_lines * char_size + (n_lines - 1) * line_gap

    # Calculate timing: when each character starts appearing
    char_start_frames: list[int] = []
    cumulative = 0
    for i, anim in enumerate(char_animations):
        char_start_frames.append(cumulative)
        cumulative += len(anim) + inter_char_delay_frames

    # Total number of frames
    total_frames = max(
        cumulative,
        max(
            start + len(anim)
            for start, anim in zip(char_start_frames, char_animations)
        ),
    )

    # Generate composed frames
    frames: list[Image.Image] = []

    for frame_idx in range(total_frames):
        canvas = Image.new("L", (canvas_width, canvas_height), color=255)

        for char_idx, (start_frame, anim) in enumerate(
            zip(char_start_frames, char_animations)
        ):
            if frame_idx < start_frame:
                # Character hasn't started yet
                continue

            local_frame_idx = frame_idx - start_frame
            if local_frame_idx >= len(anim):
                # Character animation is complete; use last frame
                local_frame_idx = len(anim) - 1

            # Calculate position
            line = char_idx // chars_per_line
            col = char_idx % chars_per_line
            x = col * (char_size + char_gap)
            y = line * (char_size + line_gap)

            # Paste character frame
            char_frame = anim[local_frame_idx]
            if char_frame.size != (char_size, char_size):
                char_frame = char_frame.resize(
                    (char_size, char_size), Image.Resampling.LANCZOS
                )
            canvas.paste(char_frame, (x, y))

        frames.append(canvas)

    return frames


def calculate_layout(
    num_chars: int,
    char_size: int = 256,
    char_gap: int = 20,
    line_gap: int = 40,
    chars_per_line: int = 10,
) -> tuple[int, int]:
    """Calculate the canvas size needed for a given number of characters.

    Returns:
        (width, height) of the canvas.
    """
    n_lines = (num_chars + chars_per_line - 1) // chars_per_line
    actual_cols = min(num_chars, chars_per_line)

    width = actual_cols * char_size + max(0, actual_cols - 1) * char_gap
    height = n_lines * char_size + max(0, n_lines - 1) * line_gap

    return (width, height)


__all__ = ["compose_text_animation", "calculate_layout"]
