from typing import Any

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

    async def analyze_image_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: dict[str, Any],
    ) -> str:
        self.last_json_schema = json_schema
        return '{"calories": 450}'


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


async def test_concrete_subclass_can_analyze_image_json() -> None:
    adapter = _FakeVision()
    schema = {"type": "object", "properties": {"calories": {"type": "number"}}}
    result = await adapter.analyze_image_json(
        system_prompt="Extract nutrition data.",
        user_prompt="Analyze this meal.",
        image_bytes=b"\x89PNG",
        mime_type="image/png",
        json_schema=schema,
    )
    assert result == '{"calories": 450}'
    assert adapter.last_json_schema is schema


def test_analyze_image_json_is_abstract() -> None:
    """Verify analyze_image_json is listed as an abstract method."""
    assert "analyze_image_json" in VisionPort.__abstractmethods__


def test_partial_implementation_cannot_be_instantiated() -> None:
    """A subclass implementing only analyze_image (not analyze_image_json) cannot be instantiated."""

    class _PartialVision(VisionPort):
        async def analyze_image(self, system_prompt: str, user_prompt: str, image_bytes: bytes, mime_type: str) -> str:
            return ""

    with pytest.raises(TypeError):
        _PartialVision()  # type: ignore[abstract]


def test_vision_port_importable_from_ports_package() -> None:
    from kume.ports import VisionPort as FromPorts
    from kume.ports.output import VisionPort as FromOutput

    assert FromPorts is FromOutput
