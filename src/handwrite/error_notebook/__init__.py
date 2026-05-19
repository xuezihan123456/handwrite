"""Error-notebook module — render student mistakes as polished review pages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from handwrite.error_notebook.annotation import (
    red_correction,
    strike_through,
    underline,
)
from handwrite.error_notebook.builder import ErrorEntry, ErrorNotebookBuilder
from handwrite.error_notebook.diff import (
    DIFF_DELETE,
    DIFF_EQUAL,
    DIFF_INSERT,
    DIFF_REPLACE,
    DiffSegment,
    diff_answers,
    tokenize_answer,
)

__all__ = [
    "ErrorEntry",
    "ErrorNotebookBuilder",
    "build_error_notebook",
    "DiffSegment",
    "diff_answers",
    "tokenize_answer",
    "DIFF_EQUAL",
    "DIFF_DELETE",
    "DIFF_INSERT",
    "DIFF_REPLACE",
    "strike_through",
    "underline",
    "red_correction",
]


def build_error_notebook(
    entries: Sequence[ErrorEntry | dict],
    output_path: str | Path,
    *,
    style: str = "工整楷书",
    paper: str = "横线纸",
    layout: str = "自然",
    font_size: int = 64,
    title: str = "错题本",
    dpi: int = 300,
) -> dict[str, object]:
    """Render *entries* into a multi-page error-notebook PDF."""
    if not entries:
        raise ValueError("entries must contain at least one ErrorEntry or dict")

    builder = ErrorNotebookBuilder(title=title)
    builder.extend(entries)
    return builder.export_pdf(
        output_path,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
        dpi=dpi,
    )
