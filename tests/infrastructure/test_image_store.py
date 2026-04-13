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


class TestCurrentRequestId:
    def test_set_images_updates_current_request_id(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img"])
        assert store.current_request_id == "req-1"

    def test_latest_set_images_wins(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"a"])
        store.set_images("req-2", [b"b"])
        assert store.current_request_id == "req-2"

    def test_default_is_empty_string(self) -> None:
        store = ImageStore()
        assert store.current_request_id == ""


class TestMimeTypes:
    def test_stores_and_retrieves_mime_type(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img"], ["image/png"])
        assert store.get_mime_type("req-1", 1) == "image/png"

    def test_defaults_to_jpeg(self) -> None:
        store = ImageStore()
        store.set_images("req-1", [b"img"])
        assert store.get_mime_type("req-1", 1) == "image/jpeg"

    def test_unknown_returns_jpeg(self) -> None:
        store = ImageStore()
        assert store.get_mime_type("unknown", 1) == "image/jpeg"
