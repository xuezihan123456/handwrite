"""Data classes describing a single exam-sheet problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Figure:
    """One geometric figure entry attached to a problem.

    ``image`` is a transparent overlay produced by
    :mod:`handwrite.geometry_sheet.figures`; ``size`` describes the requested
    canvas size of that overlay.  ``caption`` is rendered underneath the
    figure when the problem is composed onto a page.
    """

    image: Any
    size: tuple[int, int] = (320, 240)
    caption: str = ""

    def __post_init__(self) -> None:
        try:
            from PIL import Image  # local import to avoid hard import cycles
        except ImportError as exc:  # pragma: no cover - PIL is a hard dep
            raise RuntimeError("Pillow is required for geometry sheets") from exc

        if not isinstance(self.image, Image.Image):
            raise TypeError("Figure.image must be a PIL.Image.Image")
        width, height = self.image.size
        # Always normalise to a tuple of ints; tolerate user-provided sizes
        # smaller than the actual image by widening to the image's bounds.
        target_w = max(int(self.size[0]), width)
        target_h = max(int(self.size[1]), height)
        self.size = (target_w, target_h)


@dataclass
class Problem:
    """A single exam problem with figures, solution steps, and formulas."""

    question_text: str
    figures: list[Figure] = field(default_factory=list)
    solution_steps: list[str] = field(default_factory=list)
    answer: str = ""
    formula_latex: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question_text, str):
            raise TypeError("question_text must be a string")
        if self.figures is None:
            self.figures = []
        if self.solution_steps is None:
            self.solution_steps = []
        if not isinstance(self.figures, list):
            self.figures = list(self.figures)
        if not isinstance(self.solution_steps, list):
            self.solution_steps = list(self.solution_steps)
        for fig in self.figures:
            if not isinstance(fig, Figure):
                raise TypeError("figures must be a list of Figure instances")
        for step in self.solution_steps:
            if not isinstance(step, str):
                raise TypeError("solution_steps must be strings")

    @property
    def is_empty(self) -> bool:
        """True when there is no rendered content for the problem."""
        return (
            not self.question_text.strip()
            and not self.figures
            and not any(step.strip() for step in self.solution_steps)
            and not self.answer.strip()
            and not (self.formula_latex or "").strip()
        )

    @classmethod
    def from_text(
        cls,
        question_text: str,
        steps: Sequence[str] | None = None,
        answer: str = "",
        formula_latex: str | None = None,
    ) -> "Problem":
        return cls(
            question_text=question_text,
            figures=[],
            solution_steps=list(steps or []),
            answer=answer,
            formula_latex=formula_latex,
        )
