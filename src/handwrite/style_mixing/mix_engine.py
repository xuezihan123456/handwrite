"""Main style mixing engine.

Orchestrates style vector mixing and image-level style transfer into a
single unified API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .style_vector import StyleVector
from .style_mixer import mix_multi, mix_styles
from .style_transfer import TransferResult, transfer_style


@dataclass(frozen=True)
class MixRecipe:
    """A named mixture of styles with per-style weights."""

    name: str
    styles: list[StyleVector]
    weights: list[float]
    method: str = "linear"

    def __post_init__(self) -> None:
        if len(self.styles) != len(self.weights):
            raise ValueError("styles and weights must have the same length")
        if not self.styles:
            raise ValueError("At least one style is required")

    @property
    def blended(self) -> StyleVector:
        """Compute the blended style vector for this recipe."""
        if len(self.styles) == 1:
            return self.styles[0]
        if len(self.styles) == 2:
            total = self.weights[0] + self.weights[1]
            ratio = self.weights[1] / total if total else 0.5
            return mix_styles(self.styles[0], self.styles[1], ratio, method=self.method)
        return mix_multi(self.styles, self.weights, method=self.method)


@dataclass
class MixEngine:
    """High-level engine for style mixing and transfer.

    Usage::

        engine = MixEngine()
        result = engine.apply(image, recipe)
        # or
        mixed = engine.blend(style_a, style_b, ratio=0.3)
        result = engine.transfer(image, mixed)
    """

    def blend(
        self,
        style_a: StyleVector,
        style_b: StyleVector,
        ratio: float = 0.5,
        *,
        method: str = "linear",
    ) -> StyleVector:
        """Blend two style vectors and return the resulting style."""
        return mix_styles(style_a, style_b, ratio, method=method)

    def blend_multi(
        self,
        styles: list[StyleVector],
        weights: list[float],
        *,
        method: str = "linear",
    ) -> StyleVector:
        """Blend multiple style vectors with explicit weights."""
        return mix_multi(styles, weights, method=method)

    def transfer(
        self,
        image: Image.Image,
        target_style: StyleVector,
        source_style: StyleVector | None = None,
    ) -> TransferResult:
        """Apply style transfer to an image."""
        return transfer_style(image, target_style, source_style=source_style)

    def apply(
        self,
        image: Image.Image,
        recipe: MixRecipe,
        source_style: StyleVector | None = None,
    ) -> TransferResult:
        """Blend styles from a recipe and apply the result to an image.

        This is the one-call convenience method.
        """
        target = recipe.blended
        return transfer_style(image, target, source_style=source_style)

    # -- Preset recipes ------------------------------------------------------

    @staticmethod
    def preset_neat_with_touch_of_cursive() -> MixRecipe:
        """80% neat, 20% cursive -- clean with a hint of flow."""
        return MixRecipe(
            name="neat_with_touch_of_cursive",
            styles=[StyleVector.neat(), StyleVector.cursive()],
            weights=[0.8, 0.2],
        )

    @staticmethod
    def preset_balanced_mix() -> MixRecipe:
        """Equal blend of neat, cursive, and messy."""
        return MixRecipe(
            name="balanced_mix",
            styles=[StyleVector.neat(), StyleVector.cursive(), StyleVector.messy()],
            weights=[0.34, 0.33, 0.33],
        )

    @staticmethod
    def preset_casual_cursive() -> MixRecipe:
        """60% cursive, 40% messy -- relaxed flowing style."""
        return MixRecipe(
            name="casual_cursive",
            styles=[StyleVector.cursive(), StyleVector.messy()],
            weights=[0.6, 0.4],
        )


__all__ = [
    "MixRecipe",
    "MixEngine",
]
