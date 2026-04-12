from __future__ import annotations

from kume.infrastructure.image_store import ImageStore


class TestSetAndGetImages:
    def test_set_images_and_get_by_index(self) -> None:
        store = ImageStore()
        images = [b"img1", b"img2", b"img3"]
        store.set_images("req-1", images)

        assert store.get_image("req-1", 1) == b"img1"
        assert store.get_image("req-1", 2) == b"img2"
        assert store.get_image("req-1", 3) == b"img3"


class TestGetImageInvalidIndex:
    def test_index_zero_returns_none(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img1"])
        assert store.get_image("req-1", 0) is None

    def test_negative_index_returns_none(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img1"])
        assert store.get_image("req-1", -1) is None

    def test_index_beyond_length_returns_none(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img1", b"img2"])
        assert store.get_image("req-1", 3) is None


class TestGetImageUnknownRequest:
    def test_unknown_request_id_returns_none(self) -> None:
        store = ImageStore()
        assert store.get_image("unknown", 1) is None


class TestClear:
    def test_clear_removes_images(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img1"])
        assert store.get_image("req-1", 1) == b"img1"

        store.clear("req-1")
        assert store.get_image("req-1", 1) is None

    def test_clear_nonexistent_request_does_not_error(self) -> None:
        store = ImageStore()
        store.clear("nonexistent")  # Should not raise


class TestMultipleRequests:
    def test_multiple_requests_dont_interfere(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img-a"])
        store.set_images("req-2", [b"img-b"])

        assert store.get_image("req-1", 1) == b"img-a"
        assert store.get_image("req-2", 1) == b"img-b"

        store.clear("req-1")
        assert store.get_image("req-1", 1) is None
        assert store.get_image("req-2", 1) == b"img-b"
