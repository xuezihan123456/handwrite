"""Built-in paper template definitions.

These definitions mirror the JSON files in data/papers/ and serve as fallbacks
when the JSON directory is unavailable.
"""

from __future__ import annotations

from typing import Any

# A4 at 300 DPI
_A4_SIZE = [2480, 3508]

BUILTIN_PAPER_DEFS: dict[str, dict[str, Any]] = {
    "康奈尔笔记": {
        "name": "康奈尔笔记",
        "description": "Cornell note-taking system with cue, note, and summary regions",
        "size": _A4_SIZE,
        "regions": [
            {"type": "line", "x1": 700, "y1": 0, "x2": 700, "y2": 3508, "color": 200, "width": 2},
            {"type": "line", "x1": 0, "y1": 780, "x2": 2480, "y2": 780, "color": 200, "width": 2},
            {"type": "line", "x1": 0, "y1": 2800, "x2": 2480, "y2": 2800, "color": 200, "width": 2},
            {"type": "text", "x": 200, "y": 300, "text": "线索栏", "size": 60, "color": 160},
            {"type": "text", "x": 1200, "y": 300, "text": "笔记栏", "size": 60, "color": 160},
            {"type": "text", "x": 900, "y": 2900, "text": "总结栏", "size": 60, "color": 160},
            {
                "type": "hline_group",
                "y_start": 900,
                "y_end": 2750,
                "spacing": 100,
                "color": 230,
                "width": 1,
                "x1": 720,
                "x2": 2460,
            },
        ],
    },
    "作文稿纸": {
        "name": "作文稿纸",
        "description": "Chinese essay writing grid paper (800 cells)",
        "size": _A4_SIZE,
        "regions": [
            {
                "type": "hline_group",
                "y_start": 300,
                "y_end": 3300,
                "spacing": 120,
                "color": 214,
                "width": 1,
                "x1": 200,
                "x2": 2280,
            },
            {
                "type": "vline_group",
                "x_start": 200,
                "x_end": 2280,
                "spacing": 120,
                "color": 214,
                "width": 1,
                "y1": 300,
                "y2": 3300,
            },
            {"type": "text", "x": 1050, "y": 120, "text": "作文稿纸", "size": 70, "color": 160},
            {"type": "line", "x1": 200, "y1": 300, "x2": 2280, "y2": 300, "color": 180, "width": 2},
            {"type": "line", "x1": 200, "y1": 3300, "x2": 2280, "y2": 3300, "color": 180, "width": 2},
            {"type": "line", "x1": 200, "y1": 300, "x2": 200, "y2": 3300, "color": 180, "width": 2},
            {"type": "line", "x1": 2280, "y1": 300, "x2": 2280, "y2": 3300, "color": 180, "width": 2},
        ],
    },
    "五线谱": {
        "name": "五线谱",
        "description": "Music staff paper for notation",
        "size": _A4_SIZE,
        "regions": [
            {
                "type": "staff_group",
                "y_start": 400,
                "staff_count": 8,
                "line_spacing": 24,
                "staff_gap": 260,
                "color": 180,
                "width": 1,
                "x1": 200,
                "x2": 2280,
            },
            {"type": "text", "x": 1000, "y": 120, "text": "五线谱", "size": 70, "color": 160},
        ],
    },
    "错题本": {
        "name": "错题本",
        "description": "Error correction notebook with question, wrong answer, and correction regions",
        "size": _A4_SIZE,
        "regions": [
            {
                "type": "hline_group",
                "y_start": 300,
                "y_end": 3400,
                "spacing": 120,
                "color": 230,
                "width": 1,
                "x1": 200,
                "x2": 2280,
            },
            {"type": "line", "x1": 200, "y1": 600, "x2": 2280, "y2": 600, "color": 200, "width": 2},
            {"type": "line", "x1": 200, "y1": 1600, "x2": 2280, "y2": 1600, "color": 200, "width": 2},
            {"type": "line", "x1": 200, "y1": 2600, "x2": 2280, "y2": 2600, "color": 200, "width": 2},
            {"type": "text", "x": 250, "y": 400, "text": "题目", "size": 55, "color": 140},
            {"type": "text", "x": 250, "y": 1400, "text": "错误解法", "size": 55, "color": 140},
            {"type": "text", "x": 250, "y": 2400, "text": "正确解法", "size": 55, "color": 140},
            {"type": "text", "x": 250, "y": 3100, "text": "总结与反思", "size": 55, "color": 140},
        ],
    },
    "思维导图": {
        "name": "思维导图",
        "description": "Mind map paper with central node and radiating branches",
        "size": _A4_SIZE,
        "regions": [
            {"type": "ellipse", "cx": 1240, "cy": 1754, "rx": 260, "ry": 120, "color": 200, "width": 2},
            {"type": "line", "x1": 1500, "y1": 1754, "x2": 2100, "y2": 1100, "color": 210, "width": 1},
            {"type": "line", "x1": 1500, "y1": 1754, "x2": 2100, "y2": 1754, "color": 210, "width": 1},
            {"type": "line", "x1": 1500, "y1": 1754, "x2": 2100, "y2": 2400, "color": 210, "width": 1},
            {"type": "line", "x1": 980, "y1": 1754, "x2": 380, "y2": 1100, "color": 210, "width": 1},
            {"type": "line", "x1": 980, "y1": 1754, "x2": 380, "y2": 1754, "color": 210, "width": 1},
            {"type": "line", "x1": 980, "y1": 1754, "x2": 380, "y2": 2400, "color": 210, "width": 1},
            {"type": "ellipse", "cx": 2100, "cy": 1100, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "ellipse", "cx": 2100, "cy": 1754, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "ellipse", "cx": 2100, "cy": 2400, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "ellipse", "cx": 380, "cy": 1100, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "ellipse", "cx": 380, "cy": 1754, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "ellipse", "cx": 380, "cy": 2400, "rx": 180, "ry": 70, "color": 220, "width": 1},
            {"type": "text", "x": 1100, "y": 1720, "text": "中心主题", "size": 55, "color": 120},
        ],
    },
    "英语练习纸": {
        "name": "英语练习纸",
        "description": "English practice paper with four-line three-grid layout for handwriting",
        "size": _A4_SIZE,
        "regions": [
            {"type": "text", "x": 1000, "y": 120, "text": "English Practice", "size": 70, "color": 160},
            {
                "type": "four_line_group",
                "y_start": 350,
                "y_end": 3300,
                "group_height": 200,
                "color": 200,
                "width": 1,
                "x1": 200,
                "x2": 2280,
                "dash_middle": True,
                "dash_color": 220,
            },
        ],
    },
}


def builtin_paper_names() -> list[str]:
    """Return the names of all built-in paper definitions."""
    return list(BUILTIN_PAPER_DEFS.keys())
