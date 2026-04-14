import pytest

from kume.ports.output.messaging import MessagingPort


def test_messaging_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MessagingPort()  # type: ignore[abstract]


class _FakeMessaging(MessagingPort):
    async def send_message(self, chat_id: int, text: str) -> None:
        self.last_chat_id = chat_id
        self.last_text = text

    async def send_and_get_id(self, chat_id: int, text: str) -> int:
        self.last_chat_id = chat_id
        self.last_text = text
        return 999

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        self.last_chat_id = chat_id
        self.last_message_id = message_id
        self.last_text = text


async def test_concrete_subclass_can_send_message() -> None:
    adapter = _FakeMessaging()
    await adapter.send_message(chat_id=42, text="hello")
    assert adapter.last_chat_id == 42
    assert adapter.last_text == "hello"


async def test_concrete_subclass_can_send_and_get_id() -> None:
    adapter = _FakeMessaging()
    msg_id = await adapter.send_and_get_id(chat_id=42, text="hello")
    assert msg_id == 999
    assert adapter.last_chat_id == 42
    assert adapter.last_text == "hello"


async def test_concrete_subclass_can_edit_message() -> None:
    adapter = _FakeMessaging()
    await adapter.edit_message(chat_id=42, message_id=7, text="updated")
    assert adapter.last_chat_id == 42
    assert adapter.last_message_id == 7
    assert adapter.last_text == "updated"


def test_partial_implementation_raises_type_error() -> None:
    """A subclass that only implements send_message cannot be instantiated."""

    class _Incomplete(MessagingPort):
        async def send_message(self, chat_id: int, text: str) -> None: ...

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_messaging_port_importable_from_ports_package() -> None:
    from kume.ports import MessagingPort as FromPorts
    from kume.ports.output import MessagingPort as FromOutput

    assert FromPorts is FromOutput
