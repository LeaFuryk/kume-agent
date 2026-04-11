import pytest

from kume.ports.output.messaging import MessagingPort


def test_messaging_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MessagingPort()  # type: ignore[abstract]


class _FakeMessaging(MessagingPort):
    async def send_message(self, chat_id: int, text: str) -> None:
        self.last_chat_id = chat_id
        self.last_text = text


async def test_concrete_subclass_can_send_message() -> None:
    adapter = _FakeMessaging()
    await adapter.send_message(chat_id=42, text="hello")
    assert adapter.last_chat_id == 42
    assert adapter.last_text == "hello"


def test_messaging_port_importable_from_ports_package() -> None:
    from kume.ports import MessagingPort as FromPorts
    from kume.ports.output import MessagingPort as FromOutput

    assert FromPorts is FromOutput
