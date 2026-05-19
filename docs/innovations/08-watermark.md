# Innovation #08 — Invisible Handwriting Watermark (防伪手写水印)

## Why

Generated handwriting needs an invisible, machine-verifiable provenance tag so the team can audit which images came from HandWrite and plug into a future StegMark cross-project pipeline.

## What ships

| Component | File | Role |
|-----------|------|------|
| Public API | `src/handwrite/watermark/__init__.py` | `embed_watermark` / `extract_watermark` with `method="lsb"\|"dct"` |
| LSB scheme | `src/handwrite/watermark/lsb.py` | High-capacity, bit-perfect, fragile |
| DCT scheme | `src/handwrite/watermark/dct.py` | Mid-frequency, JPEG-tolerant |
| Adapter ABC | `src/handwrite/watermark/adapter.py` | `StegMarkAdapter` + `NoopStegMarkAdapter` fallback |
| Tests | `tests/test_watermark.py` | 18 tests covering round-trip, robustness, contracts |

## LSB vs DCT tradeoffs

**LSB**: writes one payload bit into the low-order bit of each grayscale pixel. Capacity `W*H` bits. Visual diff stays below `2/255`. Fragile to JPEG.

**DCT**: embeds one bit per 8x8 block by quantizing a mid-frequency coefficient (`(4,3)`, step=24). Survives JPEG q=85.

## StegMark integration plan

`StegMarkAdapter` is an `ABC` with `embed` / `extract` abstract methods. Default `NoopStegMarkAdapter` returns the image unchanged.
