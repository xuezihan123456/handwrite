"""Helpers for rendering standard-font reference glyphs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _fit_font(char: str, font_path: str, target_size: int) -> ImageFont.FreeTypeFont:
    low = 1
    high = max(target_size * 2, 1)
    best = ImageFont.truetype(font_path, size=target_size)

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, size=mid)
        left, top, right, bottom = font.getbbox(char)
        width = right - left
        height = bottom - top

        if width <= target_size and height <= target_size:
            best = font
            low = mid + 1
        else:
            high = mid - 1

    return best


def render_standard_char(
    char: str,
    font_path: str,
    image_size: int = 256,
    char_size: int = 200,
) -> Image.Image:
    """Render a centered grayscale glyph on a white canvas."""
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path}")

    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    font = _fit_font(char, str(path), char_size)
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    width = right - left
    height = bottom - top
    x = ((image_size - width) / 2) - left
    y = ((image_size - height) / 2) - top
    draw.text((x, y), char, font=font, fill=0)
    return image
