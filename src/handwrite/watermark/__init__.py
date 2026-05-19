"""Invisible handwriting watermark module."""

from __future__ import annotations

from PIL import Image

from handwrite.watermark.adapter import NoopStegMarkAdapter, StegMarkAdapter
from handwrite.watermark.dct import DCTWatermark
from handwrite.watermark.lsb import LSBWatermark

_LSB_METHOD = "lsb"
_DCT_METHOD = "dct"
_SUPPORTED_METHODS = {_LSB_METHOD, _DCT_METHOD}


def embed_watermark(
    image: Image.Image,
    payload: bytes,
    method: str = _LSB_METHOD,
) -> Image.Image:
    method_normalized = method.lower()
    if method_normalized not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported watermark method: {method!r}")
    if method_normalized == _LSB_METHOD:
        return LSBWatermark().embed(image, payload)
    return DCTWatermark().embed(image, payload)


def extract_watermark(
    image: Image.Image,
    payload_length: int | None = None,
    method: str = _LSB_METHOD,
) -> bytes:
    method_normalized = method.lower()
    if method_normalized not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported watermark method: {method!r}")
    if method_normalized == _LSB_METHOD:
        if payload_length is None:
            raise ValueError("payload_length is required for LSB extraction")
        return LSBWatermark().extract(image, payload_length)
    return DCTWatermark().extract(image)


__all__ = [
    "embed_watermark",
    "extract_watermark",
    "LSBWatermark",
    "DCTWatermark",
    "StegMarkAdapter",
    "NoopStegMarkAdapter",
]
