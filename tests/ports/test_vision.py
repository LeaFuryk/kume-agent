import pytest

from kume.ports.output.vision import VisionPort


def test_vision_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        VisionPort()  # type: ignore[abstract]


class _FakeVision(VisionPort):
    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_image_bytes = image_bytes
        self.last_mime_type = mime_type
        return "A plate of pasta with tomato sauce."


async def test_concrete_subclass_can_analyze_image() -> None:
    adapter = _FakeVision()
    result = await adapter.analyze_image(
        system_prompt="You are a nutrition analyst.",
        user_prompt="Describe this meal.",
        image_bytes=b"\x89PNG",
        mime_type="image/png",
    )
    assert result == "A plate of pasta with tomato sauce."
    assert adapter.last_system_prompt == "You are a nutrition analyst."
    assert adapter.last_user_prompt == "Describe this meal."
    assert adapter.last_image_bytes == b"\x89PNG"
    assert adapter.last_mime_type == "image/png"


def test_vision_port_importable_from_ports_package() -> None:
    from kume.ports import VisionPort as FromPorts
    from kume.ports.output import VisionPort as FromOutput

    assert FromPorts is FromOutput
