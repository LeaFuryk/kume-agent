from abc import ABC, abstractmethod
from datetime import datetime

from kume.domain.entities import Document, Goal, LabMarker, Restriction, User


class DatabasePort(ABC):
    @abstractmethod
    async def get_or_create_user(self, telegram_id: int, name: str | None = None, language: str = "en") -> User: ...

    @abstractmethod
    async def update_user(self, user: User) -> None: ...

    @abstractmethod
    async def save_document(self, doc: Document) -> None: ...

    @abstractmethod
    async def save_lab_markers(self, markers: list[LabMarker]) -> None: ...

    @abstractmethod
    async def get_lab_markers(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]: ...

    @abstractmethod
    async def save_goal(self, goal: Goal) -> None: ...

    @abstractmethod
    async def get_goals(self, user_id: str, active_only: bool = True) -> list[Goal]: ...

    @abstractmethod
    async def save_restriction(self, restriction: Restriction) -> None: ...

    @abstractmethod
    async def get_restrictions(self, user_id: str, active_only: bool = True) -> list[Restriction]: ...
