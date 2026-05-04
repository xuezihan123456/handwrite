"""Summary renderer: draws mind map or outline layouts onto PIL images.

Renders structured layout data into handwriting-style page images,
using the project's existing compose_page pipeline for character rendering.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from handwrite.composer import WHITE_PAPER, create_paper
from handwrite.summary.mind_map_layout import MapNode, MindMapLayout
from handwrite.summary.outline_layout import OutlineItem, OutlineLayout


# Default rendering parameters
_DEFAULT_PAGE_SIZE = (2480, 3508)
_DEFAULT_FONT_SIZE = 36
_DEFAULT_MARGINS = (120, 120, 120, 120)

# Color palette for branches (grayscale-friendly, visually distinct)
_BRANCH_COLORS: list[int] = [
    30,  # Dark charcoal
    80,  # Medium dark
    120,  # Medium
    160,  # Medium light
    50,  # Dark gray
    100,  # Gray
    140,  # Light gray
    70,  # Slate
]

# Node background colors (light tinted)
_NODE_BG_COLORS: list[int] = [
    245,  # Almost white
    240,  # Very light gray
    235,  # Light gray
    230,  # Slightly darker
]

# Font size multipliers by level
_FONT_SCALE = {0: 1.4, 1: 1.1, 2: 0.95, 3: 0.85}

# Outline indent in pixels by level
_OUTLINE_INDENT_PX = {0: 0, 1: 60, 2: 120, 3: 180}

# Outline bullet symbols (Unicode)
_BULLET_SYMBOLS = {0: "", 1: "\u25cf ", 2: "\u25cb ", 3: "\u25aa "}


def render_mind_map_image(
    layout: MindMapLayout,
    *,
    page_size: tuple[int, int] = _DEFAULT_PAGE_SIZE,
    font_size: int = _DEFAULT_FONT_SIZE,
    margins: tuple[int, int, int, int] = _DEFAULT_MARGINS,
    paper: str = WHITE_PAPER,
    generate_char_fn=None,
) -> Image.Image:
    """Render a mind map layout onto a page image.

    Args:
        layout: Computed mind map layout.
        page_size: Output page dimensions (width, height).
        font_size: Base font size.
        margins: Page margins (top, right, bottom, left).
        paper: Paper type for background.
        generate_char_fn: Optional callable(char, font_size) -> PIL.Image
            for handwriting character rendering. If None, uses default font.

    Returns:
        PIL Image of the rendered mind map.
    """
    top, right, bottom, left = margins
    page_width, page_height = page_size

    # Scale layout to fit page
    scale = _compute_fit_scale(
        layout.width,
        layout.height,
        page_width - left - right,
        page_height - top - bottom,
    )

    # Create background
    page = create_paper(page_size, paper, line_spacing=font_size * 2)
    draw = ImageDraw.Draw(page)

    # Get font
    try:
        base_font = _get_font(font_size)
    except (OSError, IOError):
        base_font = None

    # Draw edges first (behind nodes)
    node_map = {n.id: n for n in layout.nodes}
    for edge in layout.edges:
        source = node_map.get(edge.source_id)
        target = node_map.get(edge.target_id)
        if source and target:
            sx = left + source.x * scale
            sy = top + source.y * scale
            tx = left + target.x * scale
            ty = top + target.y * scale
            color = _BRANCH_COLORS[target.color_index % len(_BRANCH_COLORS)]
            _draw_curved_edge(draw, sx, sy, tx, ty, color=color, width=2)

    # Draw nodes
    for node in layout.nodes:
        nx = left + node.x * scale
        ny = top + node.y * scale
        nw = node.width * scale
        nh = node.height * scale
        _draw_node(
            draw,
            page,
            node,
            nx,
            ny,
            nw,
            nh,
            font_size=font_size,
            base_font=base_font,
            generate_char_fn=generate_char_fn,
        )

    return page


def render_outline_image(
    layout: OutlineLayout,
    *,
    page_size: tuple[int, int] = _DEFAULT_PAGE_SIZE,
    font_size: int = _DEFAULT_FONT_SIZE,
    margins: tuple[int, int, int, int] = _DEFAULT_MARGINS,
    paper: str = WHITE_PAPER,
    generate_char_fn=None,
) -> Image.Image:
    """Render an outline layout onto a page image.

    Args:
        layout: Computed outline layout.
        page_size: Output page dimensions (width, height).
        font_size: Base font size.
        margins: Page margins (top, right, bottom, left).
        paper: Paper type for background.
        generate_char_fn: Optional callable(char, font_size) -> PIL.Image
            for handwriting character rendering. If None, uses default font.

    Returns:
        PIL Image of the rendered outline.
    """
    top, right, bottom, left = margins
    page_width, page_height = page_size

    # Create background
    line_spacing = int(font_size * 2.2)
    page = create_paper(page_size, paper, line_spacing=line_spacing)
    draw = ImageDraw.Draw(page)

    # Get font
    try:
        base_font = _get_font(font_size)
    except (OSError, IOError):
        base_font = None

    y = top
    line_height = int(font_size * 1.8)

    for item in layout.items:
        if y + font_size > page_height - bottom:
            break

        indent = _OUTLINE_INDENT_PX.get(item.level, 0)
        x = left + indent

        # Scale font by level
        scale = _FONT_SCALE.get(item.level, 1.0)
        item_font_size = int(font_size * scale)

        # Draw bullet symbol
        bullet = item.bullet
        if bullet:
            bullet_color = _BRANCH_COLORS[item.level % len(_BRANCH_COLORS)]
            if base_font:
                small_font = _get_font(item_font_size)
                draw.text((x, y), bullet, fill=bullet_color, font=small_font)
            x += int(item_font_size * 1.5)

        # Draw text
        _draw_outline_text(
            draw,
            page,
            item,
            x,
            y,
            font_size=item_font_size,
            base_font=base_font,
            generate_char_fn=generate_char_fn,
        )

        y += line_height

    return page


def _compute_fit_scale(
    content_width: float,
    content_height: float,
    available_width: int,
    available_height: int,
) -> float:
    """Compute scale factor to fit content within available space."""
    if content_width <= 0 or content_height <= 0:
        return 1.0

    scale_x = available_width / content_width
    scale_y = available_height / content_height
    return min(scale_x, scale_y, 1.0)  # Don't upscale


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font, falling back to default if needed."""
    # Try common Chinese-capable fonts on Windows
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",  # SimHei
        "C:/Windows/Fonts/simsun.ttc",  # SimSun
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_curved_edge(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: int,
    width: int,
) -> None:
    """Draw a slightly curved edge between two points."""
    # Simple straight line with slight curve simulation
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0

    # Add a slight perpendicular offset for curve effect
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return

    # Perpendicular offset (10% of length)
    offset = length * 0.08
    nx = -dy / length * offset
    ny = dx / length * offset

    ctrl_x = mid_x + nx
    ctrl_y = mid_y + ny

    # Draw as a series of line segments (bezier approximation)
    steps = 20
    prev_x, prev_y = x1, y1
    for i in range(1, steps + 1):
        t = i / steps
        it = 1 - t
        px = it * it * x1 + 2 * it * t * ctrl_x + t * t * x2
        py = it * it * y1 + 2 * it * t * ctrl_y + t * t * y2
        draw.line(
            (int(prev_x), int(prev_y), int(px), int(py)),
            fill=color,
            width=width,
        )
        prev_x, prev_y = px, py


def _draw_node(
    draw: ImageDraw.ImageDraw,
    page: Image.Image,
    node: MapNode,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    font_size: int,
    base_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    generate_char_fn=None,
) -> None:
    """Draw a single mind map node."""
    # Node background
    bg_color = _NODE_BG_COLORS[node.level % len(_NODE_BG_COLORS)]
    padding = 8
    bbox = (
        int(x - width / 2 - padding),
        int(y - height / 2 - padding),
        int(x + width / 2 + padding),
        int(y + height / 2 + padding),
    )

    # Draw rounded rectangle background
    radius = 12 if node.level == 0 else 8
    _draw_rounded_rect(draw, bbox, radius, fill=bg_color)

    # Draw border
    border_color = _BRANCH_COLORS[node.color_index % len(_BRANCH_COLORS)]
    _draw_rounded_rect(draw, bbox, radius, outline=border_color, width=2)

    # Draw text
    scale = _FONT_SCALE.get(node.level, 1.0)
    text_font_size = int(font_size * scale)

    if generate_char_fn is not None:
        # Use handwriting character rendering
        _draw_handwritten_text(
            page,
            node.text,
            int(x),
            int(y),
            font_size=text_font_size,
            generate_char_fn=generate_char_fn,
            color=border_color,
        )
    else:
        if base_font:
            text_font = _get_font(text_font_size)
            bbox_text = draw.textbbox((0, 0), node.text, font=text_font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            draw.text(
                (int(x - tw / 2), int(y - th / 2)),
                node.text,
                fill=border_color,
                font=text_font,
            )


def _draw_outline_text(
    draw: ImageDraw.ImageDraw,
    page: Image.Image,
    item: OutlineItem,
    x: int,
    y: int,
    *,
    font_size: int,
    base_font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
    generate_char_fn=None,
) -> None:
    """Draw text for an outline item."""
    color = _BRANCH_COLORS[item.level % len(_BRANCH_COLORS)]

    if generate_char_fn is not None:
        _draw_handwritten_text(
            page,
            item.text,
            x,
            y,
            font_size=font_size,
            generate_char_fn=generate_char_fn,
            color=color,
        )
    else:
        if base_font:
            font = _get_font(font_size)
            draw.text((x, y), item.text, fill=color, font=font)


def _draw_handwritten_text(
    page: Image.Image,
    text: str,
    x: int,
    y: int,
    *,
    font_size: int,
    generate_char_fn,
    color: int,
) -> None:
    """Draw text using handwriting character images."""
    offset_x = x
    for char in text:
        if char.isspace():
            offset_x += font_size // 3
            continue
        try:
            char_img = generate_char_fn(char, font_size)
            if char_img is not None:
                # Convert to grayscale and paste
                glyph = _prepare_char_glyph(char_img, font_size, color=color)
                if glyph is not None:
                    page.paste(color if color < 128 else 0, (offset_x, y), glyph)
                    offset_x += font_size
                else:
                    offset_x += font_size // 2
        except Exception:
            offset_x += font_size // 2


def _prepare_char_glyph(
    char_image: Image.Image,
    font_size: int,
    *,
    color: int = 0,
) -> Image.Image | None:
    """Prepare a character image glyph for pasting."""
    from PIL import ImageChops, ImageOps

    rgba = char_image.convert("RGBA")
    grayscale = ImageOps.grayscale(rgba)
    alpha = rgba.getchannel("A")
    glyph = ImageChops.multiply(ImageOps.invert(grayscale), alpha)
    bbox = glyph.getbbox()
    if bbox is None:
        return None

    cropped = glyph.crop(bbox)
    resized = ImageOps.contain(
        cropped, (font_size, font_size), Image.Resampling.LANCZOS
    )
    mask = Image.new("L", (font_size, font_size), color=0)
    x_offset = (font_size - resized.width) // 2
    y_offset = (font_size - resized.height) // 2
    mask.paste(resized, (x_offset, y_offset))
    return mask


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    radius: int,
    fill: int | None = None,
    outline: int | None = None,
    width: int = 1,
) -> None:
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = bbox
    if fill is not None:
        draw.rounded_rectangle(bbox, radius=radius, fill=fill)
    if outline is not None:
        draw.rounded_rectangle(bbox, radius=radius, outline=outline, width=width)
