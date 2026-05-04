"""Temporal handwriting engine.

High-level API for generating handwriting with age-dependent characteristics
and historical writing instrument styles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from handwrite.composer import (
    NATURAL_LAYOUT,
    WHITE_PAPER,
    compose_page,
)
from handwrite.styles import BUILTIN_STYLES

from .age_profiles import AgeGroup, AgeProfile, get_age_profile, interpolate_profiles
from .historical_style import HistoricalInstrument, apply_historical_style
from .skill_simulator import SkillSimulator
from .temporal_renderer import TemporalRenderer

PathLike = Union[str, Path]


def generate_with_age(
    text: str,
    age_group: AgeGroup | str = AgeGroup.ADULT,
    style: str = "行书流畅",
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    page_size: tuple[int, int] = (2480, 3508),
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    seed: int | None = None,
    prototype_pack: PathLike | None = None,
) -> Image.Image:
    """Generate a handwriting page with age-dependent characteristics.

    Simulates how a person at the given developmental stage would write:
    - Lower elementary (小学低年级): jittery, inconsistent sizes, unstable baseline
    - Upper elementary (小学高年级): improving but still somewhat uneven
    - Middle school (初中): neat, standardized strokes
    - High school (高中): fluent with connected strokes
    - Adult (成人): mature, flowing handwriting

    Args:
        text: The text to write.
        age_group: Target age group (AgeGroup enum or string key).
        style: Base handwriting style name.
        paper: Paper type.
        layout: Layout style.
        font_size: Base character size.
        page_size: Output page dimensions.
        margins: Page margins (top, right, bottom, left).
        seed: Random seed for reproducible output.
        prototype_pack: Optional custom prototype pack path.

    Returns:
        A page image with age-dependent handwriting characteristics.
    """
    # Resolve age group
    if isinstance(age_group, str):
        age_group = _resolve_age_group(age_group)

    # Generate base characters using the engine
    from handwrite import _get_engine

    style_id = BUILTIN_STYLES.get(style, 0)
    engine = _get_engine(prototype_pack=prototype_pack)

    characters = [c for c in text if not c.isspace()]
    if not characters:
        return Image.new("L", page_size, color=255)

    char_images: list[Image.Image] = []
    for char in characters:
        char_img = engine.generate_char(char, style_id)
        char_images.append(char_img)

    # Create temporal renderer and compose
    renderer = TemporalRenderer(age_group=age_group, seed=seed)
    return renderer.render_text(
        char_images,
        text,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
    )


def generate_historical(
    text: str,
    instrument: HistoricalInstrument | str = HistoricalInstrument.BRUSH_PEN,
    style: str = "行书流畅",
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    page_size: tuple[int, int] = (2480, 3508),
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    ink_color: tuple[int, int, int] = (20, 20, 60),
    prototype_pack: PathLike | None = None,
) -> Image.Image:
    """Generate a handwriting page with a historical writing instrument effect.

    Simulates writing with different historical instruments:
    - Brush pen (毛笔): thick/thin variation, ink pooling
    - Fountain pen (钢笔): moderate line variation, ink density changes
    - Ballpoint pen (圆珠笔): consistent lines with ink skipping
    - Reed pen (芦苇笔): broad uneven strokes with ink splatter

    Args:
        text: The text to write.
        instrument: Writing instrument type.
        style: Base handwriting style name.
        paper: Paper type.
        layout: Layout style.
        font_size: Base character size.
        page_size: Output page dimensions.
        margins: Page margins (top, right, bottom, left).
        ink_color: RGB ink color (default: dark blue-black).
        prototype_pack: Optional custom prototype pack path.

    Returns:
        A page image with historical instrument effects applied.
    """
    # Resolve instrument
    if isinstance(instrument, str):
        instrument = _resolve_instrument(instrument)

    # Generate base characters
    from handwrite import _get_engine

    style_id = BUILTIN_STYLES.get(style, 0)
    engine = _get_engine(prototype_pack=prototype_pack)

    characters = [c for c in text if not c.isspace()]
    if not characters:
        return Image.new("L", page_size, color=255)

    # Apply historical style to each character
    styled_images: list[Image.Image] = []
    for char in characters:
        char_img = engine.generate_char(char, style_id)
        styled = apply_historical_style(char_img, instrument, ink_color=ink_color)
        # Convert back to grayscale for composition
        styled_images.append(styled.convert("L"))

    # Compose using standard page layout
    return compose_page(
        styled_images,
        text,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
        layout=layout,
        paper=paper,
    )


def generate_with_age_historical(
    text: str,
    age_group: AgeGroup | str = AgeGroup.ADULT,
    instrument: HistoricalInstrument | str = HistoricalInstrument.BRUSH_PEN,
    style: str = "行书流畅",
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    page_size: tuple[int, int] = (2480, 3508),
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    ink_color: tuple[int, int, int] = (20, 20, 60),
    seed: int | None = None,
    prototype_pack: PathLike | None = None,
) -> Image.Image:
    """Generate handwriting combining age-dependent and historical effects.

    Combines age-dependent layout distortion with historical instrument
    styling, e.g., simulating a child writing with a brush pen.

    Args:
        text: The text to write.
        age_group: Target age group.
        instrument: Writing instrument type.
        style: Base handwriting style name.
        paper: Paper type.
        layout: Layout style.
        font_size: Base character size.
        page_size: Output page dimensions.
        margins: Page margins (top, right, bottom, left).
        ink_color: RGB ink color.
        seed: Random seed for reproducible output.
        prototype_pack: Optional custom prototype pack path.

    Returns:
        A page image combining both temporal effects.
    """
    # Resolve enum values
    if isinstance(age_group, str):
        age_group = _resolve_age_group(age_group)
    if isinstance(instrument, str):
        instrument = _resolve_instrument(instrument)

    # Generate base characters
    from handwrite import _get_engine

    style_id = BUILTIN_STYLES.get(style, 0)
    engine = _get_engine(prototype_pack=prototype_pack)

    characters = [c for c in text if not c.isspace()]
    if not characters:
        return Image.new("L", page_size, color=255)

    # Apply historical style first, then age-dependent rendering
    renderer = TemporalRenderer(age_group=age_group, seed=seed)

    styled_images: list[Image.Image] = []
    for char in characters:
        char_img = engine.generate_char(char, style_id)
        # Apply historical instrument effect
        historical = apply_historical_style(char_img, instrument, ink_color=ink_color)
        styled_images.append(historical)

    # Render with age-dependent layout
    return renderer.render_text(
        styled_images,
        text,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
    )


def _resolve_age_group(value: str) -> AgeGroup:
    """Resolve a string to an AgeGroup enum value."""
    # Try direct enum lookup
    try:
        return AgeGroup(value)
    except ValueError:
        pass

    # Try Chinese name mapping
    _chinese_map: dict[str, AgeGroup] = {
        "小学低年级": AgeGroup.LOWER_ELEMENTARY,
        "小学高年级": AgeGroup.UPPER_ELEMENTARY,
        "初中": AgeGroup.MIDDLE_SCHOOL,
        "高中": AgeGroup.HIGH_SCHOOL,
        "成人": AgeGroup.ADULT,
    }
    if value in _chinese_map:
        return _chinese_map[value]

    raise ValueError(
        f"Unknown age group: {value!r}. "
        f"Valid options: {[g.value for g in AgeGroup]} or "
        f"Chinese names: {list(_chinese_map.keys())}"
    )


def _resolve_instrument(value: str) -> HistoricalInstrument:
    """Resolve a string to a HistoricalInstrument enum value."""
    try:
        return HistoricalInstrument(value)
    except ValueError:
        pass

    _chinese_map: dict[str, HistoricalInstrument] = {
        "毛笔": HistoricalInstrument.BRUSH_PEN,
        "钢笔": HistoricalInstrument.FOUNTAIN_PEN,
        "圆珠笔": HistoricalInstrument.BALLPOINT_PEN,
        "芦苇笔": HistoricalInstrument.REED_PEN,
    }
    if value in _chinese_map:
        return _chinese_map[value]

    raise ValueError(
        f"Unknown instrument: {value!r}. "
        f"Valid options: {[i.value for i in HistoricalInstrument]} or "
        f"Chinese names: {list(_chinese_map.keys())}"
    )


__all__ = [
    "generate_with_age",
    "generate_historical",
    "generate_with_age_historical",
]
