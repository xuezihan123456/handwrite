"""Annotation renderer -- draw red teacher-style annotations on images.

Supports:
    STRIKETHROUGH  -- red line through erroneous text
    CIRCLE         -- red circle around erroneous text
    WAVE_UNDERLINE -- red wavy underline beneath text
    MARGIN_NOTE    -- red text note in the margin
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from handwrite.grading.error_detector import ErrorInfo, ErrorType


# Standard red for teacher annotations
ANNOTATION_RED: tuple[int, int, int] = (220, 50, 50)
ANNOTATION_RED_RGBA: tuple[int, int, int, int] = (220, 50, 50, 200)


class AnnotationType(str, Enum):
    """Types of visual annotation."""

    STRIKETHROUGH = "strikethrough"
    CIRCLE = "circle"
    WAVE_UNDERLINE = "wave_underline"
    MARGIN_NOTE = "margin_note"


@dataclass(frozen=True)
class Annotation:
    """A single annotation to be drawn on an image canvas.

    ``bbox`` is (x, y, width, height) of the target text region.
    """

    annotation_type: AnnotationType
    bbox: tuple[int, int, int, int]
    message: str = ""
    color: tuple[int, int, int] = ANNOTATION_RED
    line_width: int = 2
    font_size: int = 16

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]


def _error_type_to_annotation_type(error_type: ErrorType) -> AnnotationType:
    """Map an error type to the conventional annotation style."""
    mapping = {
        ErrorType.TYPO: AnnotationType.STRIKETHROUGH,
        ErrorType.GRAMMAR: AnnotationType.WAVE_UNDERLINE,
        ErrorType.PUNCTUATION: AnnotationType.CIRCLE,
        ErrorType.FORMAT: AnnotationType.MARGIN_NOTE,
    }
    return mapping.get(error_type, AnnotationType.STRIKETHROUGH)


@dataclass
class AnnotationRenderer:
    """Render red annotations onto a PIL Image.

    Example::

        renderer = AnnotationRenderer()
        img = Image.new("RGB", (800, 600), "white")
        annotations = renderer.from_errors(errors, char_positions)
        renderer.render(img, annotations)
    """

    color: tuple[int, int, int] = ANNOTATION_RED
    line_width: int = 3
    font_size: int = 18
    margin_right: int = 20  # pixels from right edge for margin notes
    _font: Optional[ImageFont.FreeTypeFont] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._font = self._load_font()

    def _load_font(self) -> Optional[ImageFont.FreeTypeFont]:
        """Try to load a suitable font for margin notes."""
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",     # Microsoft YaHei
            "C:/Windows/Fonts/simsun.ttc",    # SimSun
            "C:/Windows/Fonts/simhei.ttf",    # SimHei
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        for fp in font_paths:
            try:
                return ImageFont.truetype(fp, self.font_size)
            except (OSError, IOError):
                continue
        return None

    # ------------------------------------------------------------------
    # Annotation construction helpers
    # ------------------------------------------------------------------

    def from_errors(
        self,
        errors: Sequence[ErrorInfo],
        char_bboxes: Sequence[tuple[int, int, int, int]],
        *,
        default_annotation_type: Optional[AnnotationType] = None,
    ) -> list[Annotation]:
        """Build ``Annotation`` objects from detected errors.

        ``char_bboxes`` maps each character index to its (x, y, w, h) bounding box.
        Characters beyond the length of ``char_bboxes`` are skipped.
        """
        annotations: list[Annotation] = []
        n_chars = len(char_bboxes)

        for error in errors:
            start = error.position
            end = error.end_position()
            if start >= n_chars:
                continue
            end = min(end, n_chars)

            # Merge bounding boxes of the erroneous span
            boxes = char_bboxes[start:end]
            if not boxes:
                continue

            min_x = min(b[0] for b in boxes)
            min_y = min(b[1] for b in boxes)
            max_x = max(b[0] + b[2] for b in boxes)
            max_y = max(b[1] + b[3] for b in boxes)

            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            ann_type = (
                default_annotation_type
                if default_annotation_type is not None
                else _error_type_to_annotation_type(error.error_type)
            )

            message = error.message
            if error.suggestion:
                message = f"{message} -> {error.suggestion}"

            annotations.append(
                Annotation(
                    annotation_type=ann_type,
                    bbox=bbox,
                    message=message,
                    color=self.color,
                    line_width=self.line_width,
                    font_size=self.font_size,
                )
            )

        return annotations

    def make_annotation(
        self,
        annotation_type: AnnotationType,
        bbox: tuple[int, int, int, int],
        message: str = "",
    ) -> Annotation:
        """Create a single annotation manually."""
        return Annotation(
            annotation_type=annotation_type,
            bbox=bbox,
            message=message,
            color=self.color,
            line_width=self.line_width,
            font_size=self.font_size,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        image: "Image.Image",
        annotations: Sequence[Annotation],
    ) -> "Image.Image":
        """Draw all annotations onto *image* (in-place) and return it.

        Requires Pillow.
        """
        if Image is None:
            raise ImportError("Pillow is required for annotation rendering")

        draw = ImageDraw.Draw(image)

        for ann in annotations:
            if ann.annotation_type == AnnotationType.STRIKETHROUGH:
                self._draw_strikethrough(draw, ann)
            elif ann.annotation_type == AnnotationType.CIRCLE:
                self._draw_circle(draw, ann)
            elif ann.annotation_type == AnnotationType.WAVE_UNDERLINE:
                self._draw_wave_underline(draw, ann)
            elif ann.annotation_type == AnnotationType.MARGIN_NOTE:
                self._draw_margin_note(draw, ann, image.width)

        return image

    def render_to_new(
        self,
        size: tuple[int, int],
        annotations: Sequence[Annotation],
        background: str = "white",
    ) -> "Image.Image":
        """Create a new image and render annotations onto it."""
        if Image is None:
            raise ImportError("Pillow is required for annotation rendering")
        img = Image.new("RGB", size, background)
        return self.render(img, annotations)

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_strikethrough(
        self, draw: "ImageDraw.ImageDraw", ann: Annotation
    ) -> None:
        """Draw a horizontal red line through the center of the bbox."""
        y_mid = ann.y + ann.height // 2
        draw.line(
            [(ann.x, y_mid), (ann.x + ann.width, y_mid)],
            fill=ann.color,
            width=ann.line_width,
        )

    def _draw_circle(self, draw: "ImageDraw.ImageDraw", ann: Annotation) -> None:
        """Draw an ellipse around the bbox."""
        pad = 4
        draw.ellipse(
            [
                ann.x - pad,
                ann.y - pad,
                ann.x + ann.width + pad,
                ann.y + ann.height + pad,
            ],
            outline=ann.color,
            width=ann.line_width,
        )

    def _draw_wave_underline(
        self, draw: "ImageDraw.ImageDraw", ann: Annotation
    ) -> None:
        """Draw a wavy line beneath the bbox."""
        y_base = ann.y + ann.height + 2
        amplitude = 3
        wavelength = 10
        points: list[tuple[int, int]] = []

        x = ann.x
        while x <= ann.x + ann.width:
            phase = (x - ann.x) / wavelength * 2 * math.pi
            y = y_base + int(amplitude * math.sin(phase))
            points.append((x, y))
            x += 2

        if len(points) >= 2:
            draw.line(points, fill=ann.color, width=ann.line_width)

    def _draw_margin_note(
        self,
        draw: "ImageDraw.ImageDraw",
        ann: Annotation,
        image_width: int,
    ) -> None:
        """Draw a small red text note in the right margin."""
        if not ann.message:
            return

        font = self._font
        # Truncate long messages
        display_text = ann.message if len(ann.message) <= 30 else ann.message[:28] + "..."
        text_x = image_width - self.margin_right
        text_y = ann.y

        # Use textbbox for accurate size measurement
        bbox = draw.textbbox((text_x, text_y), display_text, font=font)
        text_w = bbox[2] - bbox[0]
        # Right-align the text
        actual_x = text_x - text_w

        draw.text(
            (actual_x, text_y),
            display_text,
            fill=ann.color,
            font=font,
        )

        # Draw a thin line from the text region to the margin note
        draw.line(
            [(ann.x + ann.width, ann.y + ann.height // 2), (actual_x, text_y + 8)],
            fill=ann.color,
            width=1,
        )
