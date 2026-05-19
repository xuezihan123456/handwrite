# 笔迹时光机 (Handwriting Timeline) - Innovation 07

`handwrite.timeline` turns a sparse set of dated handwriting samples into a fitted *evolution model* that can synthesize a glyph at **any** target age, including ages that fall well outside the training range.

## Pipeline

1. **Feature extraction.** `HandwritingFeatureExtractor.extract(image)` returns an `AgeFeatures` dataclass with five normalized fields: `stroke_width`, `slant`, `connectivity`, `legibility`, and `density`.
2. **Per-feature curve fitting.** `fit_curve(ages, values, method=...)` builds a tiny `Curve` object for each feature. Three methods are supported: `"linear"`, `"polynomial"`, `"smoothing"`.
3. **TimelineModel.** `TimelineModel.fit(samples)` stores one `Curve` per feature plus the training records.

## Public API

- `TimelineModel` (class with `.fit`, `.predict_features`, `.generate_glyph`, `.to_json`, `.from_json`)
- `fit_timeline(samples)` -> `TimelineModel`
- `generate_at_age(model, char, age)` -> `PIL.Image.Image`
- `AgeFeatures` dataclass
