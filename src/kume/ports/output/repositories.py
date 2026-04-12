from abc import ABC, abstractmethod
from datetime import datetime

from kume.domain.entities import Document, Goal, LabMarker, Restriction, User


class UserRepository(ABC):
    @abstractmethod
    async def get_or_create(self, telegram_id: int, name: str | None = None, language: str = "en") -> User: ...

    @abstractmethod
    async def update(self, user: User) -> None: ...


class GoalRepository(ABC):
    @abstractmethod
    async def save(self, goal: Goal) -> None: ...

    @abstractmethod
    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Goal]: ...


class RestrictionRepository(ABC):
    @abstractmethod
    async def save(self, restriction: Restriction) -> None: ...

    @abstractmethod
    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Restriction]: ...


class DocumentRepository(ABC):
    @abstractmethod
    async def save(self, doc: Document) -> None: ...


class EmbeddingRepository(ABC):
    @abstractmethod
    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None: ...

    @abstractmethod
    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]: ...


class LabMarkerRepository(ABC):
    @abstractmethod
    async def save_many(self, markers: list[LabMarker]) -> None: ...

    @abstractmethod
    async def get_by_user(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]: ...
