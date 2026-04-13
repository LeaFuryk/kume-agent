from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class _ImageEntry:
    images: list[bytes]
    mime_types: list[str]


class ImageStore:
    """Request-scoped storage for image bytes with MIME types.

    Uses a contextvar to track the current request_id so tools can
    retrieve images without needing the request_id passed explicitly.
    """

    def __init__(self) -> None:
        self._data: dict[str, _ImageEntry] = {}
        self._current_request: ContextVar[str] = ContextVar("image_store_request", default="")

    def set_images(self, request_id: str, images: list[bytes], mime_types: list[str] | None = None) -> None:
        """Store image bytes (with optional MIME types) and set as current request."""
        if mime_types is None:
            mime_types = ["image/jpeg"] * len(images)
        self._data[request_id] = _ImageEntry(images=images, mime_types=mime_types)
        self._current_request.set(request_id)

    def get_image(self, request_id: str, index: int) -> bytes | None:
        """Get image by 1-based index. Returns None if not found."""
        entry = self._data.get(request_id)
        if entry and 1 <= index <= len(entry.images):
            return entry.images[index - 1]
        return None

    def get_mime_type(self, request_id: str, index: int) -> str:
        """Get MIME type for an image by 1-based index. Defaults to image/jpeg."""
        entry = self._data.get(request_id)
        if entry and 1 <= index <= len(entry.mime_types):
            return entry.mime_types[index - 1]
        return "image/jpeg"

    @property
    def current_request_id(self) -> str:
        """Get the current request_id (set by the orchestrator)."""
        return self._current_request.get()

    def clear(self, request_id: str) -> None:
        """Remove images after request completes."""
        self._data.pop(request_id, None)
