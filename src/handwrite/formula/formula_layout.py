"""Formula layout engine.

Computes bounding boxes and positions for each node in the parsed LaTeX AST.
Fractions stack vertically, superscripts/subscripts shrink and offset,
square roots wrap content with a radical sign, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from handwrite.formula.latex_parser import (
    FractionNode,
    GreekNode,
    IntegralNode,
    MatrixNode,
    ParseNode,
    SqrtNode,
    SubscriptNode,
    SumNode,
    SuperscriptNode,
    TextNode,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutConfig:
    """Tunable parameters for formula layout."""

    base_font_size: int = 40
    """Font size (pixels) for the main body text."""

    script_scale: float = 0.65
    """Scale factor for superscripts / subscripts."""

    limit_scale: float = 0.6
    """Scale factor for integral / sum limits."""

    fraction_spacing: int = 4
    """Vertical gap between fraction bar and numerator/denominator."""

    radical_gap: int = 3
    """Gap between radical sign and content."""

    matrix_cell_padding: int = 6
    """Padding inside each matrix cell."""

    matrix_col_gap: int = 12
    """Horizontal gap between matrix columns."""

    matrix_row_gap: int = 8
    """Vertical gap between matrix rows."""

    operator_spacing: int = 6
    """Horizontal spacing around binary operators."""


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box for a laid-out element.

    Attributes:
        x: Left edge (pixels).
        y: Top edge (pixels).
        width: Box width.
        height: Box height.
        baseline_y: Y coordinate of the text baseline within the box.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    baseline_y: int = 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass
class LayoutItem:
    """A single renderable element after layout."""

    kind: str
    """Item type: 'text', 'fraction_line', 'radical_sign', 'integral_sign',
    'sum_sign', 'matrix_bracket'."""

    bbox: BBox
    text: str = ""
    nodes: list[LayoutItem] | None = None
    font_size: int = 0
    extra: dict = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


# ---------------------------------------------------------------------------
# Layout engine
# ---------------------------------------------------------------------------

class FormulaLayout:
    """Computes layout for a parsed LaTeX AST.

    Usage::

        layout = FormulaLayout(config)
        items = layout.layout(nodes)
    """

    def __init__(self, config: LayoutConfig | None = None) -> None:
        self._cfg = config or LayoutConfig()

    # -- public --

    def layout(self, nodes: list[ParseNode]) -> list[LayoutItem]:
        """Layout a list of parse nodes and return renderable items."""
        items, _ = self._layout_sequence(nodes, font_size=self._cfg.base_font_size)
        return items

    # -- internal layout methods --

    def _layout_sequence(
        self, nodes: list[ParseNode], *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        """Layout a horizontal sequence of nodes."""
        items: list[LayoutItem] = []
        cursor_x = 0
        max_height = 0
        max_baseline = 0

        for node in nodes:
            node_items, node_bbox = self._layout_node(node, font_size=font_size)
            # Offset to current cursor position.
            for item in node_items:
                item.bbox.x += cursor_x
            items.extend(node_items)
            cursor_x += node_bbox.width + self._operator_gap(node, nodes)
            if node_bbox.height > max_height:
                max_height = node_bbox.height
            if node_bbox.baseline_y > max_baseline:
                max_baseline = node_bbox.baseline_y

        seq_bbox = BBox(x=0, y=0, width=cursor_x, height=max_height, baseline_y=max_baseline)
        return items, seq_bbox

    def _layout_node(
        self, node: ParseNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        if isinstance(node, TextNode):
            return self._layout_text(node, font_size=font_size)
        if isinstance(node, FractionNode):
            return self._layout_fraction(node, font_size=font_size)
        if isinstance(node, SuperscriptNode):
            return self._layout_superscript(node, font_size=font_size)
        if isinstance(node, SubscriptNode):
            return self._layout_subscript(node, font_size=font_size)
        if isinstance(node, SqrtNode):
            return self._layout_sqrt(node, font_size=font_size)
        if isinstance(node, IntegralNode):
            return self._layout_integral(node, font_size=font_size)
        if isinstance(node, SumNode):
            return self._layout_sum(node, font_size=font_size)
        if isinstance(node, MatrixNode):
            return self._layout_matrix(node, font_size=font_size)
        if isinstance(node, GreekNode):
            return self._layout_greek(node, font_size=font_size)
        # Fallback: treat as text.
        return self._layout_text(TextNode(str(node)), font_size=font_size)

    # -- text --

    def _layout_text(self, node: TextNode, *, font_size: int) -> tuple[list[LayoutItem], BBox]:
        width = max(1, int(len(node.text) * font_size * 0.65))
        bbox = BBox(x=0, y=0, width=width, height=font_size, baseline_y=int(font_size * 0.75))
        item = LayoutItem(kind="text", bbox=bbox, text=node.text, font_size=font_size)
        return [item], bbox

    def _layout_greek(self, node: GreekNode, *, font_size: int) -> tuple[list[LayoutItem], BBox]:
        width = max(1, int(font_size * 0.7))
        bbox = BBox(x=0, y=0, width=width, height=font_size, baseline_y=int(font_size * 0.75))
        item = LayoutItem(kind="text", bbox=bbox, text=node.name, font_size=font_size)
        return [item], bbox

    # -- fraction --

    def _layout_fraction(
        self, node: FractionNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        cfg = self._cfg
        num_items, num_bbox = self._layout_sequence(node.numerator, font_size=font_size)
        den_items, den_bbox = self._layout_sequence(node.denominator, font_size=font_size)

        child_width = max(num_bbox.width, den_bbox.width)
        line_y = num_bbox.height + cfg.fraction_spacing

        # Reposition numerator to be centered over line.
        num_offset_x = (child_width - num_bbox.width) // 2
        for item in num_items:
            item.bbox.x += num_offset_x

        # Reposition denominator below line.
        den_offset_y = line_y + cfg.fraction_spacing
        den_offset_x = (child_width - den_bbox.width) // 2
        for item in den_items:
            item.bbox.x += den_offset_x
            item.bbox.y += den_offset_y

        total_height = den_offset_y + den_bbox.height
        baseline_y = line_y

        # Fraction line item.
        line_bbox = BBox(x=0, y=line_y, width=child_width, height=2, baseline_y=line_y)
        line_item = LayoutItem(kind="fraction_line", bbox=line_bbox, font_size=font_size)

        all_items = [line_item] + num_items + den_items
        result_bbox = BBox(x=0, y=0, width=child_width, height=total_height, baseline_y=baseline_y)
        return all_items, result_bbox

    # -- superscript / subscript --

    def _layout_superscript(
        self, node: SuperscriptNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        script_size = max(10, int(font_size * self._cfg.script_scale))
        items, bbox = self._layout_sequence(node.content, font_size=script_size)
        # Shift up so that the baseline of the script sits above the parent baseline.
        offset_y = -int(font_size * 0.4)
        for item in items:
            item.bbox.y += offset_y
        result_bbox = BBox(
            x=0, y=offset_y, width=bbox.width, height=bbox.height, baseline_y=bbox.baseline_y + offset_y,
        )
        return items, result_bbox

    def _layout_subscript(
        self, node: SubscriptNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        script_size = max(10, int(font_size * self._cfg.script_scale))
        items, bbox = self._layout_sequence(node.content, font_size=script_size)
        # Shift down so script sits below the parent baseline.
        offset_y = int(font_size * 0.2)
        for item in items:
            item.bbox.y += offset_y
        result_bbox = BBox(
            x=0, y=offset_y, width=bbox.width, height=bbox.height + offset_y, baseline_y=bbox.baseline_y + offset_y,
        )
        return items, result_bbox

    # -- square root --

    def _layout_sqrt(
        self, node: SqrtNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        cfg = self._cfg
        items, content_bbox = self._layout_sequence(node.content, font_size=font_size)

        # Offset content to make room for the radical sign.
        radical_width = max(int(font_size * 0.45), 10)
        content_offset_x = radical_width + cfg.radical_gap
        for item in items:
            item.bbox.x += content_offset_x

        # Radical sign bounding box (covers left side + overbar).
        total_width = content_offset_x + content_bbox.width + 2
        overbar_y = 0
        radical_bbox = BBox(x=0, y=overbar_y, width=total_width, height=content_bbox.height + 2, baseline_y=content_bbox.baseline_y)
        radical_item = LayoutItem(
            kind="radical_sign",
            bbox=radical_bbox,
            font_size=font_size,
            extra={"content_width": content_bbox.width, "content_height": content_bbox.height},
        )

        all_items = [radical_item] + items
        return all_items, radical_bbox

    # -- integral --

    def _layout_integral(
        self, node: IntegralNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        cfg = self._cfg
        sign_width = max(int(font_size * 0.55), 14)
        limit_size = max(10, int(font_size * cfg.limit_scale))

        all_items: list[LayoutItem] = []

        # Integral sign.
        sign_bbox = BBox(x=0, y=0, width=sign_width, height=font_size, baseline_y=int(font_size * 0.75))
        all_items.append(LayoutItem(kind="integral_sign", bbox=sign_bbox, font_size=font_size))

        cursor_x = sign_width

        # Upper limit.
        if node.upper:
            up_items, up_bbox = self._layout_sequence(node.upper, font_size=limit_size)
            offset_y = -int(font_size * 0.35)
            for item in up_items:
                item.bbox.x += cursor_x + (sign_width - up_bbox.width) // 2
                item.bbox.y += offset_y
            all_items.extend(up_items)

        # Lower limit.
        if node.lower:
            lo_items, lo_bbox = self._layout_sequence(node.lower, font_size=limit_size)
            offset_y = int(font_size * 0.5)
            for item in lo_items:
                item.bbox.x += cursor_x + (sign_width - lo_bbox.width) // 2
                item.bbox.y += offset_y
            all_items.extend(lo_items)
            cursor_x += max(sign_width, lo_bbox.width)

        result_bbox = BBox(x=0, y=0, width=cursor_x, height=font_size, baseline_y=int(font_size * 0.75))
        return all_items, result_bbox

    # -- sum --

    def _layout_sum(
        self, node: SumNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        cfg = self._cfg
        sign_size = int(font_size * 0.9)
        sign_width = max(int(font_size * 0.8), 16)
        limit_size = max(10, int(font_size * cfg.limit_scale))

        all_items: list[LayoutItem] = []

        # Sum sign.
        sign_bbox = BBox(x=0, y=0, width=sign_width, height=sign_size, baseline_y=int(sign_size * 0.75))
        all_items.append(LayoutItem(kind="sum_sign", bbox=sign_bbox, font_size=font_size))

        cursor_x = sign_width

        # Upper limit.
        if node.upper:
            up_items, up_bbox = self._layout_sequence(node.upper, font_size=limit_size)
            offset_y = -int(font_size * 0.25)
            for item in up_items:
                item.bbox.x += cursor_x + (sign_width - up_bbox.width) // 2
                item.bbox.y += offset_y
            all_items.extend(up_items)

        # Lower limit.
        if node.lower:
            lo_items, lo_bbox = self._layout_sequence(node.lower, font_size=limit_size)
            offset_y = int(sign_size * 0.6)
            for item in lo_items:
                item.bbox.x += cursor_x + (sign_width - lo_bbox.width) // 2
                item.bbox.y += offset_y
            all_items.extend(lo_items)
            cursor_x += max(sign_width, lo_bbox.width)

        result_bbox = BBox(x=0, y=0, width=cursor_x, height=sign_size, baseline_y=int(sign_size * 0.75))
        return all_items, result_bbox

    # -- matrix --

    def _layout_matrix(
        self, node: MatrixNode, *, font_size: int,
    ) -> tuple[list[LayoutItem], BBox]:
        cfg = self._cfg
        if not node.rows:
            return [], BBox()

        # Layout every cell.
        cell_items_grid: list[list[tuple[list[LayoutItem], BBox]]] = []
        col_widths: list[int] = []
        row_heights: list[int] = []

        for row in node.rows:
            row_cells: list[tuple[list[LayoutItem], BBox]] = []
            for cell in row:
                items, bbox = self._layout_sequence(cell, font_size=font_size)
                row_cells.append((items, bbox))
            cell_items_grid.append(row_cells)

            row_max_h = max((bbox.height for _, bbox in row_cells), default=font_size)
            row_heights.append(row_max_h)

        # Determine column widths.
        max_cols = max((len(row) for row in cell_items_grid), default=0)
        for col_idx in range(max_cols):
            col_max_w = 0
            for row_cells in cell_items_grid:
                if col_idx < len(row_cells):
                    col_max_w = max(col_max_w, row_cells[col_idx][1].width)
            col_widths.append(col_max_w)

        # Compute cell positions and collect items.
        all_items: list[LayoutItem] = []
        bracket_width = max(int(font_size * 0.2), 4)
        content_offset_x = bracket_width + cfg.matrix_cell_padding

        y_cursor = 0
        for row_idx, row_cells in enumerate(cell_items_grid):
            x_cursor = content_offset_x
            for col_idx, (cell_items, cell_bbox) in enumerate(row_cells):
                # Center cell vertically within the row.
                y_offset = y_cursor + (row_heights[row_idx] - cell_bbox.height) // 2
                for item in cell_items:
                    item.bbox.x += x_cursor
                    item.bbox.y += y_offset
                all_items.extend(cell_items)
                if col_idx < len(col_widths):
                    x_cursor += col_widths[col_idx] + cfg.matrix_col_gap
            y_cursor += row_heights[row_idx] + cfg.matrix_row_gap

        # Remove trailing row gap.
        y_cursor = max(0, y_cursor - cfg.matrix_row_gap)
        total_width = content_offset_x * 2
        if col_widths:
            total_width += sum(col_widths) + cfg.matrix_col_gap * (len(col_widths) - 1)
        total_height = y_cursor

        # Matrix bracket items.
        left_bracket = BBox(x=0, y=0, width=bracket_width, height=total_height, baseline_y=total_height // 2)
        right_bracket = BBox(x=total_width - bracket_width, y=0, width=bracket_width, height=total_height, baseline_y=total_height // 2)
        all_items.append(LayoutItem(kind="matrix_bracket", bbox=left_bracket, font_size=font_size, extra={"side": "left", "kind": node.kind}))
        all_items.append(LayoutItem(kind="matrix_bracket", bbox=right_bracket, font_size=font_size, extra={"side": "right", "kind": node.kind}))

        result_bbox = BBox(x=0, y=0, width=total_width, height=total_height, baseline_y=total_height // 2)
        return all_items, result_bbox

    # -- helpers --

    def _operator_gap(self, node: ParseNode, siblings: list[ParseNode]) -> int:
        """Extra horizontal gap around binary operators."""
        if isinstance(node, TextNode) and node.text.strip() in {"+", "-", "=", ">", "<", "\u00b1", "\u00d7", "\u00b7"}:
            return self._cfg.operator_spacing
        return 0
