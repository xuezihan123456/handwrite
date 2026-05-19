"""StegMark adapter interface.

Plug-in contract for future cross-project StegMark integration. Provides
an abstract base class plus a no-op default that simply round-trips the
image and payload unchanged. Concrete adapters (e.g., for external
stegmark.io APIs) should subclass StegMarkAdapter and implement embed /
extract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class StegMarkAdapter(ABC):
    """Abstract adapter for external steganographic watermark backends.

    Subclasses implement embed/extract using the StegMark service (or any
    compatible backend). The default :class:`NoopStegMarkAdapter` is used
    when no external dependency is available.
    """

    @abstractmethod
    def embed(self, image: Image.Image, payload: bytes) -> Image.Image:
        """Embed a payload into the image using the backend."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, image: Image.Image, payload_length: int | None = None) -> bytes:
        """Extract a previously embedded payload."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-readable adapter name (defaults to class name)."""
        return type(self).__name__


class NoopStegMarkAdapter(StegMarkAdapter):
    """Default fallback adapter: returns the image untouched.

    Used when no real StegMark backend is configured; payload is stored on
    the instance so :meth:`extract` can echo it back for tests and graceful
    fallback paths. This is **not** a real watermark.
    """

    def __init__(self) -> None:
        self._last_payload: bytes = b""

    def embed(self, image: Image.Image, payload: bytes) -> Image.Image:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        self._last_payload = bytes(payload)
        return image.copy()

    def extract(self, image: Image.Image, payload_length: int | None = None) -> bytes:
        if payload_length is None:
            return self._last_payload
        return self._last_payload[:payload_length]


__all__ = ["StegMarkAdapter", "NoopStegMarkAdapter"]
