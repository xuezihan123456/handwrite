"""DCT-domain watermark.

Embeds binary payload bits into a mid-frequency coefficient of each 8x8
block via quantization-style sign coding. Robust to mild JPEG
recompression compared with the LSB scheme; uses a manual 2D DCT-II so we
do not introduce a scipy dependency.

Payload layout (header + body):
    bytes 0..3 : big-endian uint32 payload length (body bytes)
    bytes 4.. : payload body
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_BLOCK_SIZE = 8
# Mid-frequency coefficient location (avoid DC, very-high freq stripped by JPEG).
_COEF_ROW = 4
_COEF_COL = 3
# Quantization step; large enough to survive JPEG q=85 round-trips.
_QUANT_STEP = 24.0
_HEADER_BYTES = 4


class DCTWatermark:
    """DCT-domain watermark embedding using 8x8 block coefficient quantization."""

    def embed(self, image: Image.Image, payload: bytes) -> Image.Image:
        """Embed payload into mid-frequency DCT coefficients.

        Args:
            image: Source image (converted to grayscale).
            payload: Bytes to embed; capacity is (num_blocks // 8) - header.

        Returns:
            New grayscale image with the DCT watermark applied.
        """
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")

        payload_bytes = bytes(payload)
        header = len(payload_bytes).to_bytes(_HEADER_BYTES, byteorder="big")
        full_payload = header + payload_bytes

        grayscale = image.convert("L")
        pixels = np.array(grayscale, dtype=np.float32)
        height, width = pixels.shape

        usable_height = (height // _BLOCK_SIZE) * _BLOCK_SIZE
        usable_width = (width // _BLOCK_SIZE) * _BLOCK_SIZE
        if usable_height == 0 or usable_width == 0:
            raise ValueError("Image too small for 8x8 DCT blocks")

        num_blocks = (usable_height // _BLOCK_SIZE) * (usable_width // _BLOCK_SIZE)
        capacity_bits = num_blocks
        required_bits = len(full_payload) * 8
        if required_bits > capacity_bits:
            raise ValueError(
                f"Payload too large: needs {required_bits} bits, "
                f"image capacity is {capacity_bits} bits"
            )

        bit_stream = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
        output = pixels.copy()

        block_index = 0
        for top in range(0, usable_height, _BLOCK_SIZE):
            for left in range(0, usable_width, _BLOCK_SIZE):
                if block_index >= bit_stream.size:
                    break

                block = output[top : top + _BLOCK_SIZE, left : left + _BLOCK_SIZE]
                coefficients = _dct2(block)
                coefficients[_COEF_ROW, _COEF_COL] = _quantize_coefficient(
                    coefficients[_COEF_ROW, _COEF_COL],
                    int(bit_stream[block_index]),
                )
                output[top : top + _BLOCK_SIZE, left : left + _BLOCK_SIZE] = _idct2(
                    coefficients
                )
                block_index += 1
            if block_index >= bit_stream.size:
                break

        clipped = np.clip(output, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(clipped, mode="L")

    def extract(self, image: Image.Image) -> bytes:
        """Extract payload bytes from a DCT-watermarked image.

        Reads a 32-bit length header first, then that many body bytes.
        """
        grayscale = image.convert("L")
        pixels = np.array(grayscale, dtype=np.float32)
        height, width = pixels.shape

        usable_height = (height // _BLOCK_SIZE) * _BLOCK_SIZE
        usable_width = (width // _BLOCK_SIZE) * _BLOCK_SIZE
        if usable_height == 0 or usable_width == 0:
            raise ValueError("Image too small for 8x8 DCT blocks")

        num_blocks = (usable_height // _BLOCK_SIZE) * (usable_width // _BLOCK_SIZE)
        header_bits_needed = _HEADER_BYTES * 8
        if num_blocks < header_bits_needed:
            raise ValueError("Image too small to contain watermark header")

        # First pass: read just enough bits to decode the header.
        header_bits = _read_bits(pixels, usable_height, usable_width, header_bits_needed)
        header_bytes = np.packbits(header_bits).tobytes()
        payload_length = int.from_bytes(header_bytes[:_HEADER_BYTES], byteorder="big")

        if payload_length < 0:
            raise ValueError("Decoded payload length is negative")

        total_bits_needed = (_HEADER_BYTES + payload_length) * 8
        if total_bits_needed > num_blocks:
            raise ValueError(
                f"Decoded payload_length={payload_length} exceeds image capacity"
            )

        all_bits = _read_bits(pixels, usable_height, usable_width, total_bits_needed)
        all_bytes = np.packbits(all_bits).tobytes()
        return all_bytes[_HEADER_BYTES : _HEADER_BYTES + payload_length]


def _read_bits(
    pixels: np.ndarray,
    usable_height: int,
    usable_width: int,
    bit_count: int,
) -> np.ndarray:
    """Read `bit_count` payload bits from 8x8 DCT mid-frequency coefficients."""
    bits = np.zeros(bit_count, dtype=np.uint8)
    index = 0
    for top in range(0, usable_height, _BLOCK_SIZE):
        for left in range(0, usable_width, _BLOCK_SIZE):
            if index >= bit_count:
                return bits
            block = pixels[top : top + _BLOCK_SIZE, left : left + _BLOCK_SIZE]
            coefficients = _dct2(block)
            bits[index] = _decode_bit(coefficients[_COEF_ROW, _COEF_COL])
            index += 1
    return bits


def _quantize_coefficient(value: float, bit: int) -> float:
    """Snap a coefficient to a quantization centroid that encodes `bit`.

    Even multiples of QUANT_STEP encode 0; odd multiples encode 1.
    """
    nearest = round(value / _QUANT_STEP)
    if (nearest % 2 + 2) % 2 != bit:
        # Move to the closest centroid of the correct parity.
        candidates = (nearest - 1, nearest + 1)
        nearest = min(candidates, key=lambda c: abs(c * _QUANT_STEP - value))
    return float(nearest) * _QUANT_STEP


def _decode_bit(value: float) -> int:
    """Recover the embedded bit from a (possibly perturbed) coefficient."""
    nearest = round(value / _QUANT_STEP)
    return int((nearest % 2 + 2) % 2)


def _dct2(block: np.ndarray) -> np.ndarray:
    """Compute a 2D DCT-II for an 8x8 block via matrix multiplication."""
    matrix = _dct_matrix(_BLOCK_SIZE)
    return matrix @ block @ matrix.T


def _idct2(block: np.ndarray) -> np.ndarray:
    """Compute the inverse 2D DCT-II for an 8x8 block."""
    matrix = _dct_matrix(_BLOCK_SIZE)
    return matrix.T @ block @ matrix


_DCT_MATRIX_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(size: int) -> np.ndarray:
    """Return (and cache) the orthonormal DCT-II basis matrix of given size."""
    cached = _DCT_MATRIX_CACHE.get(size)
    if cached is not None:
        return cached

    indices = np.arange(size, dtype=np.float32)
    matrix = np.cos(
        np.pi * (2 * indices[np.newaxis, :] + 1) * indices[:, np.newaxis] / (2 * size)
    ).astype(np.float32)
    matrix *= np.sqrt(2.0 / size)
    matrix[0, :] *= 1.0 / np.sqrt(2.0)
    _DCT_MATRIX_CACHE[size] = matrix
    return matrix


__all__ = ["DCTWatermark"]
