"""Image-based style transfer without neural networks.

Applies geometric and photometric transformations to a handwriting image
to approximate the appearance encoded in a target StyleVector.  All
operations use Pillow and standard affine transforms -- no ML model required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter

from .style_vector import StyleVector


@dataclass(frozen=True)
class TransferResult:
    """Output of a style transfer operation."""

    image: Image.Image
    source_style: StyleVector
    target_style: StyleVector
    operations_applied: list[str]


def _affine_shear(image: Image.Image, angle_deg: float) -> Image.Image:
    """Apply a horizontal shear to simulate slant.

    A positive angle tilts strokes to the right (common italic look).
    """
    if abs(angle_deg) < 0.1:
        return image

    w, h = image.size
    shear_factor = math.tan(math.radians(angle_deg))

    # Use affine transform: (x', y') = (x + shear*y, y)
    new_w = w + abs(int(shear_factor * h))
    coeffs = (1, shear_factor, 0, 0, 1, 0)

    return image.transform(
        (new_w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC
    )


def _adjust_stroke_width(image: Image.Image, multiplier: float) -> Image.Image:
    """Simulate thicker or thinner strokes via erosion / dilation.

    multiplier > 1.0 => thicker (dilate slightly)
    multiplier < 1.0 => thinner (erode slightly)
    """
    if abs(multiplier - 1.0) < 0.02:
        return image

    if multiplier > 1.0:
        # Dilate: expand dark pixels
        radius = max(1, int((multiplier - 1.0) * 3))
        return image.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))
    else:
        # Erode: shrink dark pixels
        radius = max(1, int((1.0 - multiplier) * 3))
        return image.filter(ImageFilter.MinFilter(size=radius * 2 + 1))


def _adjust_ink_density(image: Image.Image, multiplier: float) -> Image.Image:
    """Adjust ink darkness / contrast.

    multiplier > 1.0 => darker, more saturated
    multiplier < 1.0 => lighter, more washed out
    """
    if abs(multiplier - 1.0) < 0.02:
        return image

    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(multiplier)

    brightness_enhancer = ImageEnhance.Brightness(enhanced)
    # Invert the multiplier for brightness: darker ink = lower brightness
    return brightness_enhancer.enhance(1.0 / multiplier)


def _adjust_neatness(image: Image.Image, neatness: float) -> Image.Image:
    """Apply a slight Gaussian blur to reduce neatness, or sharpen to increase it.

    neatness near 1.0 => sharpen slightly
    neatness near 0.0 => blur slightly
    """
    if neatness > 0.7:
        # Sharpen
        sharpness = 1.0 + (neatness - 0.7) * 2.0  # 1.0 .. 1.6
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(sharpness)
    elif neatness < 0.3:
        # Blur
        radius = (0.3 - neatness) * 3.0  # 0 .. 0.9
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image


def _adjust_connectivity(image: Image.Image, connectivity: float) -> Image.Image:
    """Simulate connected strokes by dilating horizontally at high connectivity.

    This is a rough approximation: higher connectivity = more merging of nearby strokes.
    """
    if connectivity < 0.5:
        return image

    # Slight horizontal dilation for connected look
    strength = (connectivity - 0.5) * 2.0  # 0..1
    radius = max(1, int(strength * 2))
    kernel_size = radius * 2 + 1
    return image.filter(ImageFilter.MaxFilter(size=kernel_size))


def transfer_style(
    image: Image.Image,
    target_style: StyleVector,
    source_style: StyleVector | None = None,
) -> TransferResult:
    """Apply style transfer to a handwriting image.

    Transforms the image to approximate the target style.  If a source style
    is provided, only the *difference* between source and target is applied;
    otherwise a default neutral source is assumed.

    Args:
        image: Input handwriting image (RGBA or L recommended).
        target_style: Desired output style.
        source_style: Style of the input image. If None, defaults to StyleVector.default().

    Returns:
        A TransferResult containing the transformed image and metadata.
    """
    if source_style is None:
        source_style = StyleVector.default()

    ops: list[str] = []
    result = image.convert("RGBA")

    # 1. Slant / shear
    slant_diff = target_style.slant_angle - source_style.slant_angle
    if abs(slant_diff) > 0.1:
        result = _affine_shear(result, slant_diff)
        ops.append(f"shear({slant_diff:+.1f} deg)")

    # 2. Stroke width
    sw_ratio = target_style.stroke_width / max(source_style.stroke_width, 0.01)
    if abs(sw_ratio - 1.0) > 0.02:
        result = _adjust_stroke_width(result, sw_ratio)
        ops.append(f"stroke_width(x{sw_ratio:.2f})")

    # 3. Ink density
    ink_ratio = target_style.ink_density / max(source_style.ink_density, 0.01)
    if abs(ink_ratio - 1.0) > 0.02:
        result = _adjust_ink_density(result, ink_ratio)
        ops.append(f"ink_density(x{ink_ratio:.2f})")

    # 4. Neatness (blur/sharpen)
    neat_diff = target_style.neatness - source_style.neatness
    if abs(neat_diff) > 0.05:
        result = _adjust_neatness(result, target_style.neatness)
        ops.append(f"neatness({target_style.neatness:.2f})")

    # 5. Connectivity (horizontal dilation)
    conn_diff = target_style.connectivity - source_style.connectivity
    if abs(conn_diff) > 0.1:
        result = _adjust_connectivity(result, target_style.connectivity)
        ops.append(f"connectivity({target_style.connectivity:.2f})")

    return TransferResult(
        image=result,
        source_style=source_style,
        target_style=target_style,
        operations_applied=ops,
    )


__all__ = [
    "TransferResult",
    "transfer_style",
]
