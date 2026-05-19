"""End-to-end pipeline turning a handwriting photograph into a prototype pack.

The pipeline is intentionally composed from existing HandWrite primitives:

* :class:`handwrite.ocr_style.ImagePreprocessor` for skew/perspective fixes.
* :class:`handwrite.ocr_style.CharacterSegmenter` for component-based
  character cropping (this works even without an OCR backend installed).
* :class:`handwrite.digitization.HandwritingRecognizer` for an optional
  OCR labelling pass when ``pytesseract`` (or its binary) is present.
* :class:`handwrite.photo_style.GlyphBuilder` to emit the prototype pack.

The OCR pass is optional - when it fails (missing binary, ``LookupError``,
runtime error) the pipeline falls back to positional pseudo-labels so the
pack remains usable for prototype-based generation by character index.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from handwrite.ocr_style.character_segmenter import CharBox, CharacterSegmenter
from handwrite.ocr_style.image_preprocessor import (
    ImagePreprocessor,
    PreprocessResult,
)
from handwrite.photo_style.glyph_builder import (
    GlyphBuilder,
    GlyphPackResult,
    SegmentedChar,
)
from handwrite.photo_style.quality_check import (
    PhotoQualityReport,
    assess_photo_quality,
)


_LOGGER = logging.getLogger(__name__)
_LabelFn = Callable[[np.ndarray, Sequence[CharBox]], Sequence[Optional[str]]]
_ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass(frozen=True)
class PipelineConfig:
    """Tunable parameters for :class:`PhotoStylePipeline`."""

    min_char_area: int = 80
    min_char_height: int = 12
    min_char_width: int = 8
    glyph_size: int = 256
    ocr_languages: tuple[str, ...] = ("chi_sim", "eng")
    writer_id: str = "photo_capture"


@dataclass(frozen=True)
class PhotoIngestResult:
    """Intermediate result from ingesting a single photograph."""

    source: str
    quality: PhotoQualityReport
    chars: tuple[SegmentedChar, ...] = field(default_factory=tuple)
    used: bool = True
    note: str = ""


class PhotoStylePipeline:
    """One-shot photo to personal handwriting prototype pack."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        label_fn: Optional[_LabelFn] = None,
        require_quality: bool = False,
    ) -> None:
        self._config = config or PipelineConfig()
        self._label_fn = label_fn
        self._require_quality = require_quality
        self._preprocessor = ImagePreprocessor()
        self._segmenter = CharacterSegmenter(
            min_char_area=self._config.min_char_area,
            min_char_height=self._config.min_char_height,
            min_char_width=self._config.min_char_width,
        )
        self._builder = GlyphBuilder(
            glyph_size=self._config.glyph_size,
            writer_id=self._config.writer_id,
        )

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def run(
        self,
        images: Union[_ImageInput, Iterable[_ImageInput]],
        output_dir: Union[str, Path],
        pack_name: str = "my_handwriting",
    ) -> dict[str, object]:
        start_ms = time.perf_counter()
        normalised_inputs = _coerce_image_list(images)
        if not normalised_inputs:
            raise ValueError("At least one input image must be provided")

        ingestions: list[PhotoIngestResult] = []
        for entry in normalised_inputs:
            ingestions.append(self._ingest_one(entry))

        accumulated: list[SegmentedChar] = []
        for ingestion in ingestions:
            if ingestion.used:
                accumulated.extend(ingestion.chars)

        if not accumulated:
            raise ValueError(
                "No characters could be extracted from the provided photos. "
                "Try clearer photographs with darker ink on a light page."
            )

        pack: GlyphPackResult = self._builder.build(
            accumulated,
            output_dir,
            pack_name=pack_name,
            source="photo_style",
        )
        elapsed_ms = (time.perf_counter() - start_ms) * 1000.0

        report = ingestions[0].quality
        return {
            "pack_path": str(pack.manifest_path),
            "manifest_path": str(pack.manifest_path),
            "pack_name": pack.pack_name,
            "glyph_count": pack.glyph_count,
            "chars": list(pack.chars),
            "quality": report,
            "quality_reports": [item.quality for item in ingestions],
            "photo_count": len(ingestions),
            "used_photo_count": sum(1 for item in ingestions if item.used),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def _ingest_one(self, image: _ImageInput) -> PhotoIngestResult:
        source_label = _describe_source(image)
        quality = assess_photo_quality(image)
        if self._require_quality and not quality.is_usable:
            return PhotoIngestResult(
                source=source_label,
                quality=quality,
                chars=tuple(),
                used=False,
                note="skipped: quality below threshold",
            )

        preprocessed = self._preprocess(image)
        char_boxes = self._segmenter.segment(preprocessed.image)
        if not char_boxes:
            return PhotoIngestResult(
                source=source_label,
                quality=quality,
                chars=tuple(),
                used=False,
                note="no character components detected",
            )

        labels = self._maybe_label(preprocessed.image, char_boxes)
        chars: list[SegmentedChar] = []
        for index, box in enumerate(char_boxes):
            label = labels[index] if index < len(labels) else None
            chars.append(
                SegmentedChar(
                    image=box.image,
                    bbox=(box.x, box.y, box.w, box.h),
                    label=label,
                    confidence=float(80.0 if label else 0.0),
                    ink_is_white=True,
                    writer_id=self._config.writer_id,
                )
            )
        return PhotoIngestResult(
            source=source_label,
            quality=quality,
            chars=tuple(chars),
            used=True,
            note=f"extracted {len(chars)} characters",
        )

    def _preprocess(self, image: _ImageInput) -> PreprocessResult:
        if isinstance(image, (str, Path)):
            return self._preprocessor.preprocess(str(image))
        arr = _to_grayscale_array(image)
        corrected, perspective_applied = self._preprocessor._perspective_correction(arr)
        deskewed, skew_angle = self._preprocessor._skew_correction(corrected)
        binary = self._preprocessor._binarize(deskewed)
        denoised = self._preprocessor._denoise(binary)
        return PreprocessResult(
            image=denoised,
            original_shape=arr.shape[:2],
            skew_angle=float(skew_angle),
            perspective_corrected=bool(perspective_applied),
        )

    def _maybe_label(
        self, preprocessed: np.ndarray, char_boxes: Sequence[CharBox]
    ) -> list[Optional[str]]:
        labeller = self._label_fn or _default_tesseract_labeller(
            self._config.ocr_languages
        )
        try:
            labels = list(labeller(preprocessed, char_boxes))
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.info("OCR labelling skipped (%s)", exc)
            return [None] * len(char_boxes)
        if len(labels) < len(char_boxes):
            labels.extend([None] * (len(char_boxes) - len(labels)))
        return labels[: len(char_boxes)]


def photo_to_style(
    images: Union[_ImageInput, Iterable[_ImageInput]],
    output_dir: Union[str, Path],
    pack_name: str = "my_handwriting",
    *,
    config: Optional[PipelineConfig] = None,
    label_fn: Optional[_LabelFn] = None,
) -> dict[str, object]:
    """Shortcut wrapper around :class:`PhotoStylePipeline`."""
    pipeline = PhotoStylePipeline(config=config, label_fn=label_fn)
    return pipeline.run(images, output_dir=output_dir, pack_name=pack_name)


def _coerce_image_list(
    images: Union[_ImageInput, Iterable[_ImageInput]]
) -> list[_ImageInput]:
    if isinstance(images, (str, Path, Image.Image, np.ndarray)):
        return [images]
    return [item for item in images]


def _describe_source(image: _ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, Image.Image):
        return f"PIL.Image<{image.size}>"
    if isinstance(image, np.ndarray):
        return f"ndarray<{image.shape}>"
    return repr(image)


def _to_grayscale_array(image: _ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)
    if isinstance(image, Image.Image):
        return np.array(image.convert("L"), dtype=np.uint8)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _default_tesseract_labeller(
    languages: tuple[str, ...],
) -> _LabelFn:
    """Build the default OCR labeller using pytesseract."""

    def _labeller(
        preprocessed: np.ndarray, char_boxes: Sequence[CharBox]
    ) -> list[Optional[str]]:
        try:
            import pytesseract  # noqa: WPS433 - optional dependency
        except ImportError:
            return [None] * len(char_boxes)

        lang = "+".join(languages)
        inverted_full = cv2.bitwise_not(preprocessed)
        labels: list[Optional[str]] = []
        for box in char_boxes:
            x0 = max(0, box.x - 2)
            y0 = max(0, box.y - 2)
            x1 = min(inverted_full.shape[1], box.x + box.w + 2)
            y1 = min(inverted_full.shape[0], box.y + box.h + 2)
            crop = inverted_full[y0:y1, x0:x1]
            if crop.size == 0:
                labels.append(None)
                continue
            try:
                text = pytesseract.image_to_string(
                    Image.fromarray(crop), lang=lang, config="--psm 10"
                )
            except Exception:  # pragma: no cover - defensive
                labels.append(None)
                continue
            cleaned = "".join(text.split())
            labels.append(cleaned[0] if cleaned else None)
        return labels

    return _labeller


__all__ = [
    "PhotoStylePipeline",
    "PipelineConfig",
    "PhotoIngestResult",
    "photo_to_style",
]
