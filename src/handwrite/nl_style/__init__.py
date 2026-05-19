"""Natural-language style picker for HandWrite."""

from .keywords import (
    EMOTION_TO_PARAMS,
    INTENSITY_MODIFIERS,
    KEYWORD_TO_STYLE,
)
from .parser import NaturalLanguageStyler, parse_description
from .style_vector import (
    CURSIVE_LAYOUT,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    StyleVector,
)

__all__ = [
    "NaturalLanguageStyler",
    "parse_description",
    "StyleVector",
    "KEYWORD_TO_STYLE",
    "INTENSITY_MODIFIERS",
    "EMOTION_TO_PARAMS",
    "NEAT_LAYOUT",
    "NATURAL_LAYOUT",
    "CURSIVE_LAYOUT",
]
