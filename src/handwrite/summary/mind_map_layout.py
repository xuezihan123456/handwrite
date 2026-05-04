"""Mind map layout engine with simple force-directed positioning.

Places a central topic node with radiating branches, using a repulsion
force to prevent node overlaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from handwrite.summary.text_summarizer import SummaryResult


@dataclass
class MapNode:
    """A node in the mind map."""

    id: int
    text: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    level: int = 0  # 0 = center, 1 = branch, 2 = leaf
    parent_id: int | None = None
    children_ids: list[int] = field(default_factory=list)
    color_index: int = 0


@dataclass
class MapEdge:
    """An edge connecting two nodes."""

    source_id: int
    target_id: int


@dataclass
class MindMapLayout:
    """Complete mind map layout with nodes and edges."""

    nodes: list[MapNode]
    edges: list[MapEdge]
    width: float
    height: float


# Character width estimates for layout calculation
_CHAR_WIDTH_ZH: float = 1.0  # Relative to font_size
_CHAR_WIDTH_EN: float = 0.6  # Relative to font_size
_NODE_PADDING: float = 20.0
_MIN_NODE_WIDTH: float = 80.0
_NODE_HEIGHT_RATIO: float = 2.0  # height = font_size * this ratio


def compute_mind_map_layout(
    summary: SummaryResult,
    *,
    font_size: int = 24,
    canvas_width: int = 1600,
    canvas_height: int = 1200,
    branch_spacing: float = 60.0,
    node_gap: float = 40.0,
) -> MindMapLayout:
    """Compute positions for a mind map from a summary.

    Args:
        summary: The extracted summary result.
        font_size: Base font size for text rendering.
        canvas_width: Target canvas width.
        canvas_height: Target canvas height.
        branch_spacing: Minimum spacing between branch nodes.
        node_gap: Minimum gap between any two nodes.

    Returns:
        MindMapLayout with positioned nodes and edges.
    """
    nodes: list[MapNode] = []
    edges: list[MapEdge] = []
    next_id = 0

    # Create center node (title)
    center_text = summary.title or "摘要"
    center_w, center_h = _estimate_text_size(center_text, font_size, level=0)
    center = MapNode(
        id=next_id,
        text=center_text,
        x=canvas_width / 2.0,
        y=canvas_height / 2.0,
        width=center_w,
        height=center_h,
        level=0,
        color_index=0,
    )
    nodes.append(center)
    next_id += 1

    # Collect branch content from sections, keywords, and key sentences
    branches = _collect_branches(summary)
    if not branches:
        return MindMapLayout(nodes=nodes, edges=edges, width=canvas_width, height=canvas_height)

    # Place branch nodes in a circle around center
    branch_count = len(branches)
    radius = max(
        center_w / 2.0 + branch_spacing * 2,
        min(canvas_width, canvas_height) * 0.3,
    )

    for branch_idx, (branch_text, sub_items) in enumerate(branches):
        angle = (2 * math.pi * branch_idx) / branch_count - math.pi / 2
        bx = center.x + radius * math.cos(angle)
        by = center.y + radius * math.sin(angle)

        bw, bh = _estimate_text_size(branch_text, font_size, level=1)
        branch_node = MapNode(
            id=next_id,
            text=branch_text,
            x=bx,
            y=by,
            width=bw,
            height=bh,
            level=1,
            parent_id=center.id,
            color_index=branch_idx % 8,
        )
        nodes.append(branch_node)
        center.children_ids.append(branch_node.id)
        edges.append(MapEdge(source_id=center.id, target_id=branch_node.id))
        next_id += 1

        # Place sub-items around the branch node
        if sub_items:
            sub_radius = max(bw / 2.0 + branch_spacing, 100.0)
            sub_count = len(sub_items)
            for sub_idx, sub_text in enumerate(sub_items[:6]):  # Limit sub-items
                sub_angle = angle + (sub_idx - (sub_count - 1) / 2) * 0.4
                sx = bx + sub_radius * math.cos(sub_angle)
                sy = by + sub_radius * math.sin(sub_angle)

                sw, sh = _estimate_text_size(sub_text, font_size, level=2)
                sub_node = MapNode(
                    id=next_id,
                    text=sub_text,
                    x=sx,
                    y=sy,
                    width=sw,
                    height=sh,
                    level=2,
                    parent_id=branch_node.id,
                    color_index=branch_node.color_index,
                )
                nodes.append(sub_node)
                branch_node.children_ids.append(sub_node.id)
                edges.append(MapEdge(source_id=branch_node.id, target_id=sub_node.id))
                next_id += 1

    # Apply force-directed adjustments to reduce overlaps
    _apply_force_directed(nodes, iterations=80, node_gap=node_gap)

    # Recompute canvas bounds
    final_width, final_height = _compute_bounds(nodes, canvas_width, canvas_height)

    return MindMapLayout(
        nodes=nodes, edges=edges, width=final_width, height=final_height
    )


def _collect_branches(
    summary: SummaryResult,
) -> list[tuple[str, list[str]]]:
    """Collect branch labels and their sub-items from the summary."""
    branches: list[tuple[str, list[str]]] = []

    # From sections
    for section in summary.sections[:8]:
        sub_items = section.items[:6]
        branches.append((section.heading, sub_items))

    # If no sections, use keywords and key sentences
    if not branches:
        # Keywords as branches
        for kw in summary.keywords[:6]:
            branches.append((kw, []))

        # Key sentences as additional branches
        for sentence in summary.key_sentences[:4]:
            truncated = sentence[:30] + ("..." if len(sentence) > 30 else "")
            branches.append((truncated, []))

    # If still few branches, add bullet points
    if len(branches) < 3 and summary.bullet_points:
        for bp in summary.bullet_points[:4]:
            truncated = bp[:30] + ("..." if len(bp) > 30 else "")
            branches.append((truncated, []))

    return branches[:12]  # Cap at 12 branches


def _estimate_text_size(text: str, font_size: int, level: int) -> tuple[float, float]:
    """Estimate the pixel dimensions of a text label."""
    effective_font = font_size * {0: 1.2, 1: 1.0, 2: 0.85}.get(level, 1.0)
    width = 0.0
    for ch in text:
        if ord(ch) > 0x2E7F:
            width += effective_font * _CHAR_WIDTH_ZH
        else:
            width += effective_font * _CHAR_WIDTH_EN
    width = max(width + _NODE_PADDING * 2, _MIN_NODE_WIDTH)
    height = effective_font * _NODE_HEIGHT_RATIO
    return width, height


def _apply_force_directed(
    nodes: list[MapNode],
    *,
    iterations: int,
    node_gap: float,
) -> None:
    """Apply a simple force-directed layout to reduce node overlaps.

    Uses repulsion between nodes and attraction toward original positions.
    """
    if len(nodes) < 2:
        return

    center = nodes[0]
    repulsion_strength = 5000.0
    damping = 0.85

    # Velocities
    vx: dict[int, float] = {n.id: 0.0 for n in nodes}
    vy: dict[int, float] = {n.id: 0.0 for n in nodes}

    for _ in range(iterations):
        # Compute repulsion forces between all pairs
        for i, ni in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                dx = ni.x - nj.x
                dy = ni.y - nj.y
                dist_sq = max(dx * dx + dy * dy, 1.0)
                dist = math.sqrt(dist_sq)

                # Minimum distance based on node sizes
                min_dist = (ni.width + nj.width) / 2.0 + node_gap

                if dist < min_dist:
                    # Strong repulsion when overlapping
                    force = repulsion_strength * (min_dist - dist) / min_dist
                else:
                    # Weak repulsion at distance
                    force = repulsion_strength * 0.1 / dist_sq

                fx = force * dx / dist
                fy = force * dy / dist

                vx[ni.id] += fx
                vy[ni.id] += fy
                vx[nj.id] -= fx
                vy[nj.id] -= fy

        # Apply spring force toward original radial position (keep structure)
        for node in nodes:
            if node.id == center.id:
                # Keep center roughly centered
                dx = center.x - (center.x)
                dy = center.y - (center.y)
                continue

            if node.parent_id is not None:
                parent = next((n for n in nodes if n.id == node.parent_id), None)
                if parent is not None:
                    # Spring toward parent
                    dx = node.x - parent.x
                    dy = node.y - parent.y
                    dist = math.sqrt(dx * dx + dy * dy) or 1.0
                    # Desired distance based on level
                    desired = {1: 200.0, 2: 120.0}.get(node.level, 150.0)
                    spring_force = (dist - desired) * 0.02
                    vx[node.id] -= spring_force * dx / dist
                    vy[node.id] -= spring_force * dy / dist

        # Update positions
        for node in nodes:
            if node.id == center.id:
                continue
            node.x += vx[node.id] * damping
            node.y += vy[node.id] * damping
            vx[node.id] *= damping * 0.5
            vy[node.id] *= damping * 0.5


def _compute_bounds(
    nodes: list[MapNode],
    min_width: float,
    min_height: float,
) -> tuple[float, float]:
    """Compute the bounding box of all nodes."""
    if not nodes:
        return min_width, min_height

    max_x = max(n.x + n.width / 2.0 for n in nodes)
    max_y = max(n.y + n.height / 2.0 for n in nodes)
    min_x = min(n.x - n.width / 2.0 for n in nodes)
    min_y = min(n.y - n.height / 2.0 for n in nodes)

    # Add padding
    padding = 80.0
    width = max(max_x - min_x + padding * 2, min_width)
    height = max(max_y - min_y + padding * 2, min_height)

    return width, height
