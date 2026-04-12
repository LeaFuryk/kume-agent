from abc import ABC, abstractmethod


class ResourceProcessorPort(ABC):
    @abstractmethod
    async def process(self, raw_bytes: bytes) -> str: ...
