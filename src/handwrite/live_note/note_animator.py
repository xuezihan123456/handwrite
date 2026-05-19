"""Live classroom-note writing engine.

Orchestrates `handwrite.animation` primitives plus paper composition and
pacing utilities to produce a paragraph-level "live writing" video. The
output is a stream of grayscale PIL frames showing the page being filled in
character by character with optional pen-tip cursor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PIL import Image

from handwrite.composer import (
    CURSIVE_LAYOUT,
    GRID_PAPER,
    MI_PAPER,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    RULED_PAPER,
    WHITE_PAPER,
    create_paper,
)

from .cursor import overlay_cursor_on_frames
from .pacing import PacingStrategy, plan_frame_budget


PathLike = Union[str, Path]

# Small page so live-note rendering remains fast. The composer's default
# page is 2480x3508 which is meant for print; we target the screen instead.
_DEFAULT_PAGE_SIZE = (640, 800)
_DEFAULT_MARGINS = (40, 40, 40, 40)
_GUIDED_PAPERS = {RULED_PAPER, GRID_PAPER, MI_PAPER}
_GRID_PAPERS = {GRID_PAPER, MI_PAPER}
_DEFAULT_STYLE = "\u884c\u4e66\u6d41\u7545"  # 行书流畅


@dataclass
class NoteAnimationConfig:
    """Configuration for the live-note writer.

    Attributes:
        style: Handwriting style name (passed through to the engine).
        paper: Paper background type.
        layout: Layout style (natural / neat / cursive).
        font_size: Glyph size in pixels.
        fps: Frames per second for the output.
        wpm: Writing speed (characters per minute baseline).
        cursor: Whether to render a pen-tip cursor.
        pacing_strategy: Pacing rule for per-character frame budgets.
        page_size: Canvas size (width, height) in pixels.
        margins: ``(top, right, bottom, left)`` padding inside the page.
        prototype_pack: Optional custom prototype pack path.
    """

    style: str = _DEFAULT_STYLE
    paper: str = RULED_PAPER
    layout: str = NATURAL_LAYOUT
    font_size: int = 80
    fps: int = 24
    wpm: int = 80
    cursor: bool = True
    pacing_strategy: PacingStrategy = "punctuation_pause"
    page_size: tuple[int, int] = _DEFAULT_PAGE_SIZE
    margins: tuple[int, int, int, int] = _DEFAULT_MARGINS
    prototype_pack: PathLike | None = None
    cursor_radius: int = 6
    cursor_halo: bool = True

    def total_canvas(self) -> tuple[int, int]:
        return self.page_size


@dataclass
class _Slot:
    """Internal layout slot describing where a glyph belongs."""

    char: str
    x: int
    y: int
    is_drawable: bool = True
    frame_budget: int = 1
    glyph_image: Image.Image | None = field(default=None, repr=False)


class LiveNoteEngine:
    """High-level orchestrator for paragraph-level live-writing videos.

    The engine is intentionally a thin wrapper around the existing
    `handwrite` engine. It is responsible for:

    1. Producing a glyph image for every character.
    2. Laying glyphs onto a paper canvas in reading order with pacing.
    3. Stitching frames together so each glyph fades-in over its budget.
    4. Adding an optional pen-tip cursor to indicate the writing position.
    """

    def __init__(self, config: NoteAnimationConfig | None = None) -> None:
        self.config = config or NoteAnimationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        text: str,
        style: str | None = None,
        paper: str | None = None,
        layout: str | None = None,
        font_size: int | None = None,
        fps: int | None = None,
        wpm: int | None = None,
        cursor: bool | None = None,
    ) -> list[Image.Image]:
        """Render ``text`` to a list of grayscale frames."""
        if text is None or not text.strip():
            raise ValueError("text must not be empty")

        cfg = self._merge_overrides(
            style=style,
            paper=paper,
            layout=layout,
            font_size=font_size,
            fps=fps,
            wpm=wpm,
            cursor=cursor,
        )

        slots = self._layout_slots(text, cfg)
        frames = self._compose_frames(text, slots, cfg)
        if cfg.cursor:
            positions = self._cursor_positions(slots, frames, cfg)
            frames = overlay_cursor_on_frames(
                frames,
                positions,
                radius=cfg.cursor_radius,
                halo=cfg.cursor_halo,
            )
        return frames

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _merge_overrides(self, **overrides: object) -> NoteAnimationConfig:
        cfg = self.config
        merged = NoteAnimationConfig(
            style=overrides.get("style") or cfg.style,
            paper=overrides.get("paper") or cfg.paper,
            layout=overrides.get("layout") or cfg.layout,
            font_size=int(overrides.get("font_size") or cfg.font_size),
            fps=int(overrides.get("fps") or cfg.fps),
            wpm=int(overrides.get("wpm") or cfg.wpm),
            cursor=cfg.cursor if overrides.get("cursor") is None else bool(overrides.get("cursor")),
            pacing_strategy=cfg.pacing_strategy,
            page_size=cfg.page_size,
            margins=cfg.margins,
            prototype_pack=cfg.prototype_pack,
            cursor_radius=cfg.cursor_radius,
            cursor_halo=cfg.cursor_halo,
        )
        return merged

    def _grid(self, cfg: NoteAnimationConfig) -> tuple[int, int, int]:
        """Return (max_columns, max_rows, line_height) for the page."""
        top, right, bottom, left = cfg.margins
        page_width, page_height = cfg.page_size
        char_gap, line_gap = _layout_spacing(cfg.font_size, cfg.layout)
        line_height = cfg.font_size + line_gap
        column_step = (
            line_height if cfg.paper in _GRID_PAPERS else cfg.font_size + char_gap
        )
        first_column_x = _aligned_origin(left, column_step) if cfg.paper in _GRID_PAPERS else left
        first_row_y = _aligned_origin(top, line_height) if cfg.paper in _GUIDED_PAPERS else top

        max_columns = _count_slots(
            total_extent=page_width,
            start=first_column_x,
            end_margin=right,
            item_extent=cfg.font_size,
            step=column_step,
        )
        max_rows = _count_slots(
            total_extent=page_height,
            start=first_row_y,
            end_margin=bottom,
            item_extent=cfg.font_size,
            step=line_height,
        )
        return max(1, max_columns), max(1, max_rows), line_height

    def _layout_slots(
        self,
        text: str,
        cfg: NoteAnimationConfig,
    ) -> list[_Slot]:
        max_columns, _max_rows, line_height = self._grid(cfg)
        top, _right, _bottom, left = cfg.margins
        char_gap, _line_gap = _layout_spacing(cfg.font_size, cfg.layout)
        column_step = (
            line_height if cfg.paper in _GRID_PAPERS else cfg.font_size + char_gap
        )
        first_column_x = _aligned_origin(left, column_step) if cfg.paper in _GRID_PAPERS else left
        first_row_y = _aligned_origin(top, line_height) if cfg.paper in _GUIDED_PAPERS else top

        budgets = plan_frame_budget(
            text,
            base_fps=cfg.fps,
            wpm=cfg.wpm,
            strategy=cfg.pacing_strategy,
        )

        slots: list[_Slot] = []
        column = 0
        row = 0
        for index, char in enumerate(text):
            if char == "\n":
                slots.append(
                    _Slot(
                        char=char,
                        x=first_column_x + column * column_step,
                        y=first_row_y + row * line_height,
                        is_drawable=False,
                        frame_budget=budgets[index],
                    )
                )
                row += 1
                column = 0
                continue

            if column >= max_columns:
                row += 1
                column = 0

            x = first_column_x + column * column_step
            y = first_row_y + row * line_height
            slot = _Slot(
                char=char,
                x=x,
                y=y,
                is_drawable=not char.isspace(),
                frame_budget=budgets[index],
            )
            slots.append(slot)
            column += 1

        # Attach glyph images for drawable slots.
        if any(slot.is_drawable for slot in slots):
            self._attach_glyphs(slots, cfg)
        return slots

    def _attach_glyphs(self, slots: list[_Slot], cfg: NoteAnimationConfig) -> None:
        # Defer engine import so the live_note module can be imported even
        # when the heavyweight StyleEngine has not been initialised yet.
        from handwrite import BUILTIN_STYLES, _get_engine  # noqa: WPS433

        style_id = BUILTIN_STYLES.get(cfg.style, 0)
        engine = _get_engine(prototype_pack=cfg.prototype_pack)
        cache: dict[str, Image.Image] = {}
        for slot in slots:
            if not slot.is_drawable:
                continue
            cached = cache.get(slot.char)
            if cached is None:
                cached = engine.generate_char(slot.char, style_id)
                if cached.size != (cfg.font_size, cfg.font_size):
                    cached = cached.resize(
                        (cfg.font_size, cfg.font_size), Image.Resampling.LANCZOS
                    )
                cache[slot.char] = cached
            slot.glyph_image = cached

    # ------------------------------------------------------------------
    # Frame composition
    # ------------------------------------------------------------------

    def _compose_frames(
        self,
        text: str,
        slots: list[_Slot],
        cfg: NoteAnimationConfig,
    ) -> list[Image.Image]:
        page_size = cfg.page_size
        background = create_paper(page_size, cfg.paper, line_spacing=cfg.font_size + 8)

        # Build a fully-written final canvas first. We then reveal slots by
        # cross-fading between the previous fill state and the next.
        final_canvas = background.copy()
        for slot in slots:
            if slot.glyph_image is None or not slot.is_drawable:
                continue
            _paste_glyph(final_canvas, slot.glyph_image, (slot.x, slot.y))

        frames: list[Image.Image] = []
        current_canvas = background.copy()
        # Always include at least one frame of blank paper for the intro.
        frames.append(current_canvas.copy())

        for slot in slots:
            budget = max(1, slot.frame_budget)
            if not slot.is_drawable or slot.glyph_image is None:
                # Non-drawable slot (whitespace / newline): hold the current
                # frame for the requested budget.
                for _ in range(budget):
                    frames.append(current_canvas.copy())
                continue

            next_canvas = current_canvas.copy()
            _paste_glyph(next_canvas, slot.glyph_image, (slot.x, slot.y))
            for step in range(1, budget + 1):
                alpha = step / budget
                blended = Image.blend(current_canvas, next_canvas, alpha)
                frames.append(blended)
            current_canvas = next_canvas

        if not frames:
            frames.append(background)
        return frames

    # ------------------------------------------------------------------
    # Cursor positioning
    # ------------------------------------------------------------------

    def _cursor_positions(
        self,
        slots: list[_Slot],
        frames: list[Image.Image],
        cfg: NoteAnimationConfig,
    ) -> list[tuple[int, int]]:
        """One cursor position per frame, tracking the active glyph."""
        positions: list[tuple[int, int]] = []
        if not slots:
            return [(cfg.margins[3], cfg.margins[0])] * len(frames)

        # Frame 0 is the intro blank; cursor sits at the first glyph slot.
        first = slots[0]
        positions.append(_glyph_anchor(first, cfg.font_size))

        for slot in slots:
            anchor = _glyph_anchor(slot, cfg.font_size)
            budget = max(1, slot.frame_budget)
            for _ in range(budget):
                positions.append(anchor)

        if len(positions) < len(frames):
            positions.extend([positions[-1]] * (len(frames) - len(positions)))
        return positions[: len(frames)]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def live_note_video(
    text: str,
    output_path: PathLike,
    style: str = _DEFAULT_STYLE,
    paper: str = RULED_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    fps: int = 24,
    wpm: int = 80,
    cursor: bool = True,
    format: str = "gif",
) -> dict[str, object]:
    """Render ``text`` to a live-note video file (GIF or MP4).

    Returns a metadata dictionary describing the produced file.
    """
    if text is None or not text.strip():
        raise ValueError("text must not be empty")

    config = NoteAnimationConfig(
        style=style,
        paper=paper,
        layout=layout,
        font_size=int(font_size),
        fps=int(fps),
        wpm=int(wpm),
        cursor=bool(cursor),
    )
    engine = LiveNoteEngine(config=config)
    frames = engine.render(text)

    # Defer the import so the live_note module remains importable in
    # contexts where the exporter is not yet ready.
    from handwrite.animation.animation_engine import export_animation

    path = Path(output_path)
    export_animation(frames, path, format=format, fps=config.fps)

    duration_s = len(frames) / float(config.fps) if config.fps > 0 else 0.0
    return {
        "frame_count": len(frames),
        "duration_s": duration_s,
        "output_path": str(path),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _layout_spacing(font_size: int, layout: str) -> tuple[int, int]:
    if layout == NEAT_LAYOUT:
        return max(4, font_size // 12), max(8, font_size // 4)
    if layout == CURSIVE_LAYOUT:
        return max(8, font_size // 8), max(14, font_size // 3)
    return max(6, font_size // 10), max(12, font_size // 3)


def _aligned_origin(margin: int, step: int) -> int:
    if step <= 0:
        return int(margin)
    return ((int(margin) + step - 1) // step) * step


def _count_slots(
    *,
    total_extent: int,
    start: int,
    end_margin: int,
    item_extent: int,
    step: int,
) -> int:
    available_extent = total_extent - end_margin - start
    if available_extent < item_extent:
        return 0
    if step <= 0:
        return 1
    return 1 + max(0, (available_extent - item_extent) // step)


def _glyph_anchor(slot: _Slot, font_size: int) -> tuple[int, int]:
    """Return a point near the glyph's lower-right where the pen-tip sits."""
    return (slot.x + max(2, font_size // 2), slot.y + max(2, font_size // 2))


def _paste_glyph(
    page: Image.Image,
    glyph: Image.Image,
    origin: tuple[int, int],
) -> None:
    from PIL import ImageChops, ImageOps  # local import to keep top clean

    rgba = glyph.convert("RGBA")
    grayscale = ImageOps.grayscale(rgba)
    alpha = rgba.getchannel("A")
    mask = ImageChops.multiply(ImageOps.invert(grayscale), alpha)
    bbox = mask.getbbox()
    if bbox is None:
        return
    page.paste(0, origin, mask)


__all__ = ["LiveNoteEngine", "NoteAnimationConfig", "live_note_video"]
