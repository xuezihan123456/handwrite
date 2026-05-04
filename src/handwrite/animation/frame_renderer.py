"""Animation frame rendering from stroke trajectories.

Renders progressive handwriting animation frames by revealing the
character image along drawn trajectory paths with ink diffusion effects.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def render_animation_frames(
    char_image: Image.Image,
    trajectories: list[list[tuple[float, float]]],
    num_frames: int,
    canvas_size: tuple[int, int],
    brush_size: int = 3,
    ink_diffusion: float = 1.5,
) -> list[Image.Image]:
    """Render animation frames showing progressive character writing.

    Args:
        char_image: Target character image (grayscale, white background).
        trajectories: Ordered stroke trajectories (lists of (x, y) points).
        num_frames: Total number of frames to generate.
        canvas_size: Output frame size (width, height).
        brush_size: Pen brush thickness in pixels.
        ink_diffusion: Gaussian blur radius for ink diffusion effect.

    Returns:
        List of PIL Images representing animation frames.
        The first frame is blank (white), the last frame shows the full character.
    """
    if num_frames < 1:
        return []

    if not trajectories:
        # No strokes: return blank frames
        return [_blank_frame(canvas_size) for _ in range(num_frames)]

    width, height = canvas_size
    target_array = _prepare_target(char_image, canvas_size)

    # Pre-compute total trajectory length in points
    total_points = sum(len(t) for t in trajectories)
    if total_points == 0:
        return [_blank_frame(canvas_size) for _ in range(num_frames)]

    frames: list[Image.Image] = []

    for frame_idx in range(num_frames):
        # Progress from 0.0 (blank) to 1.0 (full character)
        progress = frame_idx / max(1, num_frames - 1)

        # Calculate how many trajectory points to draw
        drawn_points = int(progress * total_points)

        # Build the reveal mask
        mask = _build_reveal_mask(
            trajectories,
            drawn_points,
            (width, height),
            brush_size,
            ink_diffusion,
        )

        # Composite: reveal the character through the mask
        frame = _composite_frame(target_array, mask)
        frames.append(Image.fromarray(frame, mode="L"))

    return frames


def _prepare_target(
    char_image: Image.Image,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Resize and convert the character image to match canvas size."""
    resized = char_image.convert("L").resize(canvas_size, Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.float32)


def _blank_frame(canvas_size: tuple[int, int]) -> Image.Image:
    """Create a blank white frame."""
    return Image.new("L", canvas_size, color=255)


def _build_reveal_mask(
    trajectories: list[list[tuple[float, float]]],
    drawn_points: int,
    canvas_size: tuple[int, int],
    brush_size: int,
    ink_diffusion: float,
) -> np.ndarray:
    """Build a reveal mask showing which pixels have been drawn so far.

    Returns a float32 array in [0, 1] where 1.0 = fully revealed.
    """
    width, height = canvas_size
    mask = np.zeros((height, width), dtype=np.uint8)

    remaining = drawn_points

    for trajectory in trajectories:
        if remaining <= 0:
            break

        stroke_len = len(trajectory)
        points_to_draw = min(remaining, stroke_len)

        if points_to_draw >= 2:
            _draw_stroke_on_mask(mask, trajectory, points_to_draw, brush_size)

            # Add extra ink diffusion at the pen tip (wet ink effect)
            if points_to_draw < stroke_len:
                tip_x, tip_y = trajectory[points_to_draw - 1]
                tip_radius = brush_size + int(ink_diffusion * 2)
                cv2.circle(
                    mask,
                    (int(tip_x), int(tip_y)),
                    tip_radius,
                    200,
                    -1,
                )

        remaining -= stroke_len

    # Apply Gaussian blur for ink diffusion
    if ink_diffusion > 0:
        ksize = max(1, int(ink_diffusion * 4)) | 1  # Ensure odd kernel size
        mask = cv2.GaussianBlur(mask, (ksize, ksize), ink_diffusion)

    return mask.astype(np.float32) / 255.0


def _draw_stroke_on_mask(
    mask: np.ndarray,
    trajectory: list[tuple[float, float]],
    num_points: int,
    brush_size: int,
) -> None:
    """Draw stroke segments on the mask using anti-aliased lines."""
    for i in range(1, num_points):
        x1, y1 = trajectory[i - 1]
        x2, y2 = trajectory[i]
        cv2.line(
            mask,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            255,
            brush_size,
            cv2.LINE_AA,
        )
        # Draw circles at each point for smoother appearance
        cv2.circle(mask, (int(x1), int(y1)), brush_size // 2, 255, -1)

    # Draw the last point
    if num_points > 0:
        x, y = trajectory[num_points - 1]
        cv2.circle(mask, (int(x), int(y)), brush_size // 2, 255, -1)


def _composite_frame(
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Composite the character onto a white background using the reveal mask.

    Where mask=1.0, show the character; where mask=0.0, show white.
    The ink is darkened slightly near the drawing edge for a wet-ink look.
    """
    white = np.full_like(target, 255.0)
    result = white * (1.0 - mask) + target * mask
    return np.clip(result, 0, 255).astype(np.uint8)


__all__ = ["render_animation_frames"]
