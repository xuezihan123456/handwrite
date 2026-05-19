"""High-level :class:`TimelineModel` for the handwriting timeline.

Combines :class:`HandwritingFeatureExtractor` and :func:`fit_curve` to
expose a tiny API that callers can drive without touching the building
blocks directly:

    >>> from handwrite.timeline import fit_timeline, generate_at_age
    >>> model = fit_timeline([(7, "child.png"), (15, "teen.png"), (25, "adult.png")])
    >>> glyph = generate_at_age(model, "好", age=10)

The model fits a curve per :class:`AgeFeatures` field, predicts features at
any target age (including modest extrapolation), and synthesises a glyph
PNG using PIL fonts so that no external rendering dependency is required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .curve_fitting import Curve, fit_curve
from .feature_extractor import AgeFeatures, HandwritingFeatureExtractor

ImageSource = Union[str, Path, Image.Image]
SamplePair = tuple[int, ImageSource]

_FEATURE_NAMES: tuple[str, ...] = (
    "stroke_width",
    "slant",
    "connectivity",
    "legibility",
    "density",
)

_FEATURE_CLAMPS: dict[str, tuple[float, float]] = {
    "stroke_width": (0.0, 50.0),
    "slant": (-30.0, 30.0),
    "connectivity": (0.0, 1.0),
    "legibility": (0.0, 1.0),
    "density": (0.0, 1.0),
}

_DEFAULT_GLYPH_SIZE = 256


@dataclass(frozen=True)
class _Sample:
    """Internal record of a fitted sample."""

    age: float
    features: AgeFeatures


class TimelineModel:
    """A fitted age-to-features evolution model.

    The model stores one :class:`Curve` per feature (stroke_width, slant,
    connectivity, legibility, density) plus the original training samples
    so the model can be inspected and serialized to JSON.
    """

    def __init__(
        self,
        curves: dict[str, Curve],
        samples: Sequence[_Sample],
        *,
        method: str = "polynomial",
        glyph_size: int = _DEFAULT_GLYPH_SIZE,
    ) -> None:
        self._curves = dict(curves)
        self._samples = list(samples)
        self._method = method
        self._glyph_size = int(glyph_size)
        self._extractor = HandwritingFeatureExtractor()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        samples: Sequence[SamplePair],
        method: str = "polynomial",
        glyph_size: int = _DEFAULT_GLYPH_SIZE,
    ) -> "TimelineModel":
        """Fit a :class:`TimelineModel` from a list of (age, sample) pairs.

        Args:
            samples: Iterable of ``(age, image_or_path)`` pairs.
            method: Curve fitting method (see :func:`fit_curve`).
            glyph_size: Output glyph size when synthesising new ages.

        Returns:
            A fitted :class:`TimelineModel`.

        Raises:
            ValueError: If fewer than two samples are supplied.
        """
        sample_list = list(samples)
        if len(sample_list) < 2:
            raise ValueError(
                "TimelineModel.fit requires at least two (age, sample) pairs"
            )

        extractor = HandwritingFeatureExtractor()
        records: list[_Sample] = []
        for age, source in sample_list:
            features = extractor.extract(source)
            records.append(_Sample(age=float(age), features=features))

        ages = [record.age for record in records]
        curves: dict[str, Curve] = {}
        for feature_name in _FEATURE_NAMES:
            values = [getattr(record.features, feature_name) for record in records]
            curves[feature_name] = fit_curve(
                ages,
                values,
                method=method,
                clamp=_FEATURE_CLAMPS[feature_name],
            )

        return cls(
            curves=curves,
            samples=records,
            method=method,
            glyph_size=glyph_size,
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Return the curve fitting method used by this model."""
        return self._method

    @property
    def glyph_size(self) -> int:
        """Return the glyph image side length used by :meth:`generate_glyph`."""
        return self._glyph_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered tuple of fitted feature names."""
        return _FEATURE_NAMES

    @property
    def training_ages(self) -> list[float]:
        """Return the ages of the original training samples."""
        return [record.age for record in self._samples]

    def curve(self, feature_name: str) -> Curve:
        """Return the fitted :class:`Curve` for a single feature."""
        if feature_name not in self._curves:
            raise KeyError(f"Unknown feature: {feature_name!r}")
        return self._curves[feature_name]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_features(self, age: float) -> AgeFeatures:
        """Predict :class:`AgeFeatures` at any age.

        The result is automatically clamped to the per-feature valid range
        so even extreme extrapolations stay physically meaningful.
        """
        values = {
            name: self._curves[name].predict(float(age)) for name in _FEATURE_NAMES
        }
        return AgeFeatures(
            stroke_width=values["stroke_width"],
            slant=values["slant"],
            connectivity=values["connectivity"],
            legibility=values["legibility"],
            density=values["density"],
        )

    def generate_glyph(
        self,
        char: str,
        age: float,
        base_style: str | None = None,
        *,
        glyph_size: int | None = None,
        font_path: str | None = None,
    ) -> Image.Image:
        """Render a single glyph image that reflects the predicted features.

        Args:
            char: The character to render. The first character of the
                string is used when more than one is supplied.
            age: Target age (in years). May be inside or outside the
                training range.
            base_style: Optional style name reserved for future style
                mixing - currently retained for API compatibility.
            glyph_size: Optional override for the output square size.
            font_path: Optional path to a TTF/OTF font. When omitted, a
                bundled-default fallback is used.

        Returns:
            A grayscale :class:`PIL.Image.Image` whose width and height
            equal ``glyph_size`` (or the model's configured ``glyph_size``).
        """
        if not char:
            raise ValueError("char must be a non-empty string")

        size = int(glyph_size or self._glyph_size)
        symbol = char[0]
        features = self.predict_features(age)

        # Step 1: render a clean base glyph at the requested size.
        base = _render_base_glyph(symbol, size, font_path=font_path)

        # Step 2: morph stroke width by blurring then re-thresholding.
        adjusted = _apply_stroke_width(base, features.stroke_width, size)

        # Step 3: apply slant via affine shear.
        adjusted = _apply_slant(adjusted, features.slant)

        # Step 4: simulate connectivity with a gentle blur (cursive = softer).
        if features.connectivity > 0.05:
            radius = 0.3 + features.connectivity * 1.4
            adjusted = adjusted.filter(ImageFilter.GaussianBlur(radius=radius))

        # Step 5: legibility controls contrast; lower legibility => softer ink.
        contrast = 0.6 + 0.8 * float(max(0.0, min(1.0, features.legibility)))
        adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)

        # Step 6: density acts as overall darkness multiplier.
        darkness = 0.55 + 0.6 * float(max(0.0, min(1.0, features.density)))
        adjusted = ImageEnhance.Brightness(adjusted).enhance(1.05 - darkness * 0.35)

        # Ensure final shape exactly matches the caller's expectation.
        if adjusted.size != (size, size):
            adjusted = adjusted.resize((size, size), Image.Resampling.LANCZOS)
        return adjusted.convert("L")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the model."""
        return {
            "method": self._method,
            "glyph_size": self._glyph_size,
            "curves": {name: curve.to_dict() for name, curve in self._curves.items()},
            "samples": [
                {"age": record.age, "features": record.features.to_dict()}
                for record in self._samples
            ],
        }

    def to_json(self) -> str:
        """Return a JSON string with the model parameters."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TimelineModel":
        """Reconstruct a :class:`TimelineModel` from :meth:`to_dict` output."""
        curves_raw = data.get("curves") or {}
        if not isinstance(curves_raw, dict):
            raise ValueError("curves payload must be a dict")
        curves = {
            name: Curve.from_dict(payload)
            for name, payload in curves_raw.items()
            if isinstance(payload, dict)
        }
        for name in _FEATURE_NAMES:
            if name not in curves:
                raise ValueError(f"Missing curve for feature {name!r}")

        samples_raw = data.get("samples") or []
        samples: list[_Sample] = []
        if isinstance(samples_raw, list):
            for entry in samples_raw:
                if not isinstance(entry, dict):
                    continue
                features_payload = entry.get("features")
                if not isinstance(features_payload, dict):
                    continue
                samples.append(
                    _Sample(
                        age=float(entry.get("age", 0.0)),
                        features=AgeFeatures.from_dict(features_payload),
                    )
                )

        method = str(data.get("method", "polynomial"))
        glyph_size = int(data.get("glyph_size", _DEFAULT_GLYPH_SIZE))
        return cls(curves=curves, samples=samples, method=method, glyph_size=glyph_size)

    @classmethod
    def from_json(cls, payload: str) -> "TimelineModel":
        """Reconstruct a :class:`TimelineModel` from a JSON string."""
        return cls.from_dict(json.loads(payload))


# ----------------------------------------------------------------------
# Module-level convenience API
# ----------------------------------------------------------------------


def fit_timeline(
    samples: Sequence[SamplePair],
    method: str = "polynomial",
    glyph_size: int = _DEFAULT_GLYPH_SIZE,
) -> TimelineModel:
    """Convenience constructor matching the documented public API."""
    return TimelineModel.fit(samples, method=method, glyph_size=glyph_size)


def generate_at_age(
    model: TimelineModel,
    char: str,
    age: float,
    *,
    glyph_size: int | None = None,
    font_path: str | None = None,
) -> Image.Image:
    """Synthesize a glyph at the requested age from a fitted model."""
    return model.generate_glyph(
        char,
        age,
        glyph_size=glyph_size,
        font_path=font_path,
    )


# ----------------------------------------------------------------------
# Internal rendering helpers
# ----------------------------------------------------------------------


def _resolve_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    """Resolve a usable :class:`ImageFont` instance.

    Tries the explicit ``font_path`` first, then a handful of well-known
    Windows/Linux/macOS handwriting fonts, finally falling back to PIL's
    default bitmap font so the rendering pipeline never crashes.
    """
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            "C:/Windows/Fonts/STKAITI.TTF",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/STXINGKA.TTF",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
    )
    target_size = max(8, int(size * 0.72))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=target_size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _render_base_glyph(char: str, size: int, font_path: str | None = None) -> Image.Image:
    """Render the base character as a centred grayscale glyph."""
    img = Image.new("L", (size, size), 255)
    font = _resolve_font(font_path, size)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
    except (AttributeError, TypeError):
        bbox = (0, 0, size, size)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) / 2 - bbox[0]
    y = (size - text_h) / 2 - bbox[1]
    draw.text((int(x), int(y)), char, font=font, fill=0)
    return img


def _apply_stroke_width(
    image: Image.Image,
    stroke_width: float,
    size: int,
) -> Image.Image:
    """Approximate stroke-width changes via blur + thresholding."""
    sw = max(0.0, float(stroke_width))
    # Map stroke width to a blur radius around an empirical baseline of 4 px.
    delta = sw - 4.0
    radius = max(0.0, min(4.0, 0.18 * delta + 0.4))
    if radius < 0.05:
        return image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.asarray(blurred, dtype=np.float32)
    threshold = 200 if sw < 6 else 220
    arr = np.where(arr < threshold, arr * 0.7, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _apply_slant(image: Image.Image, slant_deg: float) -> Image.Image:
    """Apply an affine shear to simulate slant in degrees."""
    angle = float(slant_deg)
    if abs(angle) < 0.5:
        return image
    shear = math.tan(math.radians(angle))
    w, h = image.size
    # Use Pillow's affine transform: matrix is (a, b, c, d, e, f).
    matrix = (1, shear, -shear * h / 2, 0, 1, 0)
    return image.transform(
        (w, h),
        Image.AFFINE,
        matrix,
        resample=Image.BICUBIC,
        fillcolor=255,
    )


__all__ = [
    "TimelineModel",
    "fit_timeline",
    "generate_at_age",
]
