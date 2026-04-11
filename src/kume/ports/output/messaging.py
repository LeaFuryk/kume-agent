from abc import ABC, abstractmethod


class MessagingPort(ABC):
    @abstractmethod
    async def send_message(self, chat_id: int, text: str) -> None: ...
