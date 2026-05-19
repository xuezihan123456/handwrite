"""Least Significant Bit (LSB) watermark.

Embeds payload bytes into the low-order bit of each grayscale pixel.
High capacity (1 bit/pixel) and visually imperceptible (max +/-1 grayscale
change), but fragile to recompression or color conversion.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class LSBWatermark:
    """LSB watermark embed/extract on grayscale images."""

    def embed(self, image: Image.Image, payload: bytes) -> Image.Image:
        """Embed payload bytes into the LSB plane of the image.

        Args:
            image: Source image (will be converted to grayscale).
            payload: Bytes to embed. Must fit in width*height bits.

        Returns:
            New grayscale PIL Image with payload encoded in LSBs.
        """
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")

        grayscale = image.convert("L")
        pixels = np.array(grayscale, dtype=np.uint8)
        flat = pixels.flatten()

        payload_bytes = bytes(payload)
        bit_count = len(payload_bytes) * 8

        if bit_count > flat.size:
            raise ValueError(
                f"Payload too large: requires {bit_count} bits, "
                f"image has {flat.size} pixels"
            )

        # Unpack payload bytes into a bit stream (MSB-first per byte)
        payload_array = np.frombuffer(payload_bytes, dtype=np.uint8)
        bits = np.unpackbits(payload_array)

        # Clear LSB on the slots we will write and OR in the payload bits
        flat[:bit_count] = (flat[:bit_count] & np.uint8(0xFE)) | bits

        watermarked = flat.reshape(pixels.shape)
        return Image.fromarray(watermarked, mode="L")

    def extract(self, image: Image.Image, payload_length: int) -> bytes:
        """Extract payload bytes from the LSB plane.

        Args:
            image: Image containing an LSB watermark.
            payload_length: Number of bytes to recover.

        Returns:
            Recovered payload bytes.
        """
        if payload_length < 0:
            raise ValueError("payload_length must be non-negative")

        grayscale = image.convert("L")
        pixels = np.array(grayscale, dtype=np.uint8).flatten()

        bit_count = payload_length * 8
        if bit_count > pixels.size:
            raise ValueError(
                f"payload_length too large for image: "
                f"need {bit_count} bits, image has {pixels.size} pixels"
            )

        bits = pixels[:bit_count] & np.uint8(0x01)
        recovered = np.packbits(bits)
        return bytes(recovered[:payload_length])


__all__ = ["LSBWatermark"]
