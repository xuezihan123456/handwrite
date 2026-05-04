"""Handwriting animation generation module.

Provides stroke extraction, trajectory generation, frame rendering,
and animation export for Chinese handwriting animations.
"""

from .animation_engine import (
    export_animation,
    generate_char_animation,
    generate_text_animation,
)

__all__ = [
    "generate_char_animation",
    "generate_text_animation",
    "export_animation",
]
