import pytest

from kume.ports.output.resource_processor import ResourceProcessorPort


def test_resource_processor_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ResourceProcessorPort()  # type: ignore[abstract]


class _FakeResourceProcessor(ResourceProcessorPort):
    async def process(self, raw_bytes: bytes, *, mime_type: str | None = None) -> str:
        return f"processed:{len(raw_bytes)} bytes"


async def test_concrete_subclass_can_process() -> None:
    processor = _FakeResourceProcessor()
    result = await processor.process(b"hello world")
    assert result == "processed:11 bytes"


def test_resource_processor_port_importable_from_ports_package() -> None:
    from kume.ports import ResourceProcessorPort as FromPorts
    from kume.ports.output import ResourceProcessorPort as FromOutput

    assert FromPorts is FromOutput
