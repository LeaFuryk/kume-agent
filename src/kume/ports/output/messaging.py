from abc import ABC, abstractmethod


class MessagingPort(ABC):
    @abstractmethod
    async def send_message(self, chat_id: int, text: str) -> None: ...

    @abstractmethod
    async def send_and_get_id(self, chat_id: int, text: str) -> int:
        """Send a message and return the message_id for later editing."""
        ...

    @abstractmethod
    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        """Edit an existing message in-place."""
        ...
