from __future__ import annotations


class ImageStore:
    """Request-scoped storage for image bytes."""

    def __init__(self) -> None:
        self._images: dict[str, list[bytes]] = {}

    def set_images(self, request_id: str, images: list[bytes]) -> None:
        """Store image bytes for a request."""
        self._images[request_id] = images

    def get_image(self, request_id: str, index: int) -> bytes | None:
        """Get image by 1-based index. Returns None if not found."""
        images = self._images.get(request_id, [])
        if 1 <= index <= len(images):
            return images[index - 1]
        return None

    def clear(self, request_id: str) -> None:
        """Remove images after request completes."""
        self._images.pop(request_id, None)
