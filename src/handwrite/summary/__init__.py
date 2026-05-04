"""Smart summary handwriting module.

Extract key information from long text and render as handwriting notes
with mind map or outline layout support.
"""

from handwrite.summary.summary_engine import (
    extract_summary,
    render_mind_map,
    render_outline,
)

__all__ = [
    "extract_summary",
    "render_mind_map",
    "render_outline",
]
