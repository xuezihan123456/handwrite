"""AR handwriting overlay module.

Provides functionality to overlay generated handwriting onto real paper
photographs, including paper detection, perspective transformation,
lighting adjustment, and texture blending.
"""

from handwrite.ar.paper_detector import PaperDetector, PaperDetectionResult
from handwrite.ar.perspective_transform import PerspectiveTransformer
from handwrite.ar.lighting_adjuster import LightingAdjuster
from handwrite.ar.texture_blender import TextureBlender
from handwrite.ar.ar_engine import AREngine, AROverlayOptions, AROverlayResult, overlay_on_paper, detect_paper_edges

__all__ = [
    "PaperDetector",
    "PaperDetectionResult",
    "PerspectiveTransformer",
    "LightingAdjuster",
    "TextureBlender",
    "AREngine",
    "AROverlayOptions",
    "AROverlayResult",
    "overlay_on_paper",
    "detect_paper_edges",
]
