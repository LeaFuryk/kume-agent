from __future__ import annotations

from datetime import datetime

from kume.domain.entities import Document, Goal, LabMarker, Meal, Restriction, User
from kume.ports.output.llm import LLMPort
from kume.ports.output.repositories import (
    DocumentRepository,
    EmbeddingRepository,
    GoalRepository,
    LabMarkerRepository,
    MealRepository,
    RestrictionRepository,
    UserRepository,
)
from kume.ports.output.vision import VisionPort


class FakeLLMPort(LLMPort):
    """A minimal LLMPort implementation for testing.

    Supports multiple responses: pass a list to return different values per call.
    """

    def __init__(self, response_text: str | list[str] = "fake response") -> None:
        if isinstance(response_text, list):
            self._responses = list(response_text)
        else:
            self._responses = [response_text]
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        return self._next_response()

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> str:  # type: ignore[override]
        return self._next_response()

    def _next_response(self) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class FakeUserRepository(UserRepository):
    """A minimal UserRepository implementation for testing."""

    def __init__(self) -> None:
        self.updated_users: list[User] = []

    async def get_or_create(self, telegram_id: int, name: str | None = None, language: str = "en") -> User:
        return User(id="fake-user", telegram_id=telegram_id, name=name, language=language)

    async def update(self, user: User) -> None:
        self.updated_users.append(user)


class FakeGoalRepository(GoalRepository):
    """A minimal GoalRepository implementation for testing."""

    def __init__(self) -> None:
        self.saved_goals: list[Goal] = []

    async def save(self, goal: Goal) -> None:
        self.saved_goals.append(goal)

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Goal]:
        return []


class FakeRestrictionRepository(RestrictionRepository):
    """A minimal RestrictionRepository implementation for testing."""

    def __init__(self) -> None:
        self.saved_restrictions: list[Restriction] = []

    async def save(self, restriction: Restriction) -> None:
        self.saved_restrictions.append(restriction)

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Restriction]:
        return []


class FakeDocumentRepository(DocumentRepository):
    """A minimal DocumentRepository implementation for testing."""

    def __init__(self) -> None:
        self.saved_docs: list[Document] = []

    async def save(self, doc: Document) -> None:
        self.saved_docs.append(doc)


class FakeEmbeddingRepository(EmbeddingRepository):
    """A minimal EmbeddingRepository implementation for testing."""

    def __init__(self) -> None:
        self.embedded: list[tuple[str, str, list[str]]] = []

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        self.embedded.append((user_id, document_id, chunks))

    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]:
        return []


class FakeLabMarkerRepository(LabMarkerRepository):
    """A minimal LabMarkerRepository implementation for testing."""

    def __init__(self) -> None:
        self.saved_markers: list[list[LabMarker]] = []

    async def save_many(self, markers: list[LabMarker]) -> None:
        self.saved_markers.append(markers)

    async def get_by_user(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]:
        all_markers = [m for batch in self.saved_markers for m in batch if m.user_id == user_id]
        if name:
            all_markers = [m for m in all_markers if name.lower() in m.name.lower()]
        return all_markers


class FakeVisionPort(VisionPort):
    """A minimal VisionPort implementation for testing."""

    def __init__(self, response_text: str = "fake vision response") -> None:
        self.response_text = response_text
        self.last_call: dict[str, object] | None = None

    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        self.last_call = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_bytes": image_bytes,
            "mime_type": mime_type,
        }
        return self.response_text

    async def analyze_image_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: dict,
    ) -> str:
        self.last_call = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "json_schema": json_schema,
        }
        return self.response_text


class FakeMealRepository(MealRepository):
    """A minimal MealRepository implementation for testing."""

    def __init__(self) -> None:
        self.saved_meals: list[Meal] = []

    async def save(self, meal: Meal) -> None:
        self.saved_meals.append(meal)

    async def get_by_user(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Meal]:
        return [m for m in self.saved_meals if m.user_id == user_id]
