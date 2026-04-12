"""PostgreSQL repository implementations sharing a common session factory."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kume.adapters.output.postgres_models import (
    DocumentModel,
    GoalModel,
    LabMarkerModel,
    RestrictionModel,
    UserModel,
)
from kume.domain.entities import Document, Goal, LabMarker, Restriction, User
from kume.ports.output.repositories import (
    DocumentRepository,
    GoalRepository,
    LabMarkerRepository,
    RestrictionRepository,
    UserRepository,
)


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create a shared session factory from a database URL."""
    engine = create_async_engine(database_url)
    return async_sessionmaker(engine, expire_on_commit=False)


class PostgresUserRepository(UserRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def get_or_create(self, telegram_id: int, name: str | None = None, language: str = "en") -> User:
        async with self._sf() as session:
            stmt = select(UserModel).where(UserModel.telegram_id == telegram_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                return _to_user(row)
            model = UserModel(
                id=str(uuid.uuid4()), telegram_id=telegram_id, name=name, language=language, timezone="UTC"
            )
            session.add(model)
            try:
                await session.commit()
            except IntegrityError:
                # Another request inserted the same telegram_id concurrently
                await session.rollback()
                result = await session.execute(stmt)
                row = result.scalar_one()
                return _to_user(row)
            return _to_user(model)

    async def update(self, user: User) -> None:
        async with self._sf() as session:
            stmt = select(UserModel).where(UserModel.id == user.id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return
            row.name = user.name
            row.language = user.language
            row.timezone = user.timezone
            await session.commit()


class PostgresGoalRepository(GoalRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(self, goal: Goal) -> None:
        async with self._sf() as session:
            model = GoalModel(
                id=goal.id,
                user_id=goal.user_id,
                description=goal.description,
                created_at=goal.created_at,
                completed_at=goal.completed_at,
            )
            session.add(model)
            await session.commit()

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Goal]:
        async with self._sf() as session:
            stmt = select(GoalModel).where(GoalModel.user_id == user_id)
            if active_only:
                stmt = stmt.where(GoalModel.completed_at.is_(None))
            result = await session.execute(stmt)
            return [_to_goal(r) for r in result.scalars().all()]


class PostgresRestrictionRepository(RestrictionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(self, restriction: Restriction) -> None:
        async with self._sf() as session:
            model = RestrictionModel(
                id=restriction.id,
                user_id=restriction.user_id,
                type=restriction.type,
                description=restriction.description,
                created_at=restriction.created_at,
                completed_at=restriction.completed_at,
            )
            session.add(model)
            await session.commit()

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Restriction]:
        async with self._sf() as session:
            stmt = select(RestrictionModel).where(RestrictionModel.user_id == user_id)
            if active_only:
                stmt = stmt.where(RestrictionModel.completed_at.is_(None))
            result = await session.execute(stmt)
            return [_to_restriction(r) for r in result.scalars().all()]


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(self, doc: Document) -> None:
        async with self._sf() as session:
            model = DocumentModel(
                id=doc.id,
                user_id=doc.user_id,
                type=doc.type,
                filename=doc.filename,
                summary=doc.summary,
                ingested_at=doc.ingested_at,
            )
            session.add(model)
            await session.commit()


class PostgresLabMarkerRepository(LabMarkerRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save_many(self, markers: list[LabMarker]) -> None:
        async with self._sf() as session:
            for marker in markers:
                model = LabMarkerModel(
                    id=marker.id,
                    document_id=marker.document_id,
                    user_id=marker.user_id,
                    name=marker.name,
                    value=marker.value,
                    unit=marker.unit,
                    reference_range=marker.reference_range,
                    date=marker.date,
                )
                session.add(model)
            await session.commit()

    async def get_by_user(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]:
        async with self._sf() as session:
            stmt = select(LabMarkerModel).where(LabMarkerModel.user_id == user_id)
            if name is not None:
                stmt = stmt.where(LabMarkerModel.name == name)
            if since is not None:
                stmt = stmt.where(LabMarkerModel.date >= since)
            result = await session.execute(stmt)
            return [_to_lab_marker(r) for r in result.scalars().all()]


# --- Model-to-entity mappers ---


def _to_user(m: UserModel) -> User:
    return User(id=m.id, telegram_id=m.telegram_id, name=m.name, language=m.language, timezone=m.timezone)


def _to_goal(m: GoalModel) -> Goal:
    return Goal(
        id=m.id, user_id=m.user_id, description=m.description, created_at=m.created_at, completed_at=m.completed_at
    )


def _to_restriction(m: RestrictionModel) -> Restriction:
    return Restriction(
        id=m.id,
        user_id=m.user_id,
        type=m.type,
        description=m.description,
        created_at=m.created_at,
        completed_at=m.completed_at,
    )


def _to_lab_marker(m: LabMarkerModel) -> LabMarker:
    return LabMarker(
        id=m.id,
        document_id=m.document_id,
        user_id=m.user_id,
        name=m.name,
        value=m.value,
        unit=m.unit,
        reference_range=m.reference_range,
        date=m.date,
    )
