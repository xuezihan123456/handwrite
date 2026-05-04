"""Handwrite Style Mixing & Transfer module.

Supports multi-style blending (e.g. 70% neat + 30% messy) and
image-level style transfer via geometric / photometric transforms.

Quick start::

    from handwrite.style_mixing import StyleVector, MixEngine

    engine = MixEngine()
    mixed = engine.blend(StyleVector.neat(), StyleVector.messy(), ratio=0.3)
    result = engine.transfer(some_image, mixed)
"""

from .style_vector import StyleVector, cosine_similarity, euclidean_distance
from .style_mixer import describe_mixture, mix_multi, mix_styles
from .style_transfer import TransferResult, transfer_style
from .interpolation_engine import bezier, lerp, slerp, weighted_blend
from .mix_engine import MixEngine, MixRecipe

__all__ = [
    # style_vector
    "StyleVector",
    "euclidean_distance",
    "cosine_similarity",
    # style_mixer
    "mix_styles",
    "mix_multi",
    "describe_mixture",
    # style_transfer
    "transfer_style",
    "TransferResult",
    # interpolation_engine
    "lerp",
    "slerp",
    "bezier",
    "weighted_blend",
    # mix_engine
    "MixEngine",
    "MixRecipe",
]
