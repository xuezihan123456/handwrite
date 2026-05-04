"""Collaborative handwriting document generation.

Multiple contributors each write different segments of a document,
with smooth style transitions at paragraph boundaries.
"""

from handwrite.collaboration.collab_engine import generate_collaborative_document
from handwrite.collaboration.collaborative_composer import CollaborativeComposer
from handwrite.collaboration.contributor import Contributor
from handwrite.collaboration.segment_assigner import (
    assign_segments,
    assign_segments_round_robin,
)
from handwrite.collaboration.style_blender import StyleBlender

__all__ = [
    "Contributor",
    "CollaborativeComposer",
    "StyleBlender",
    "assign_segments",
    "assign_segments_round_robin",
    "generate_collaborative_document",
]
