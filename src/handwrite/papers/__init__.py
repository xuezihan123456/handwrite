"""Paper template ecosystem for HandWrite.

Provides a registry of paper templates (Cornell notes, essay grids, music staff,
error notebooks, mind maps, English practice) that can be loaded from JSON
definitions or built-in defaults and rendered to PIL images.
"""

from handwrite.papers.paper_registry import PaperRegistry, get_paper, list_papers
from handwrite.papers.paper_renderer import render_paper

__all__ = [
    "PaperRegistry",
    "get_paper",
    "list_papers",
    "render_paper",
]
