"""PostgreSQL adapter implementing the DatabasePort interface."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kume.adapters.output.postgres_models import (
    DocumentModel,
    GoalModel,
    LabMarkerModel,
    RestrictionModel,
    UserModel,
)
from kume.domain.entities import Document, Goal, LabMarker, Restriction, User
from kume.ports.output.database import DatabasePort


class PostgresAdapter(DatabasePort):
    """Async PostgreSQL adapter using SQLAlchemy ORM."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def get_or_create_user(
        self,
        telegram_id: int,
        name: str | None = None,
        language: str = "en",
    ) -> User:
        async with self._session_factory() as session:
            stmt = select(UserModel).where(UserModel.telegram_id == telegram_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                return _user_model_to_entity(row)

            model = UserModel(
                id=str(uuid.uuid4()),
                telegram_id=telegram_id,
                name=name,
                language=language,
                timezone="UTC",
            )
            session.add(model)
            await session.commit()
            return _user_model_to_entity(model)

    async def update_user(self, user: User) -> None:
        async with self._session_factory() as session:
            stmt = select(UserModel).where(UserModel.id == user.id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return
            row.name = user.name
            row.language = user.language
            row.timezone = user.timezone
            await session.commit()

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def save_document(self, doc: Document) -> None:
        async with self._session_factory() as session:
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

    # ------------------------------------------------------------------
    # Lab markers
    # ------------------------------------------------------------------

    async def save_lab_markers(self, markers: list[LabMarker]) -> None:
        async with self._session_factory() as session:
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

    async def get_lab_markers(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]:
        async with self._session_factory() as session:
            stmt = select(LabMarkerModel).where(LabMarkerModel.user_id == user_id)
            if name is not None:
                stmt = stmt.where(LabMarkerModel.name == name)
            if since is not None:
                stmt = stmt.where(LabMarkerModel.date >= since)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_lab_marker_model_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    async def save_goal(self, goal: Goal) -> None:
        async with self._session_factory() as session:
            model = GoalModel(
                id=goal.id,
                user_id=goal.user_id,
                description=goal.description,
                created_at=goal.created_at,
                completed_at=goal.completed_at,
            )
            session.add(model)
            await session.commit()

    async def get_goals(self, user_id: str, active_only: bool = True) -> list[Goal]:
        async with self._session_factory() as session:
            stmt = select(GoalModel).where(GoalModel.user_id == user_id)
            if active_only:
                stmt = stmt.where(GoalModel.completed_at.is_(None))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_goal_model_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Restrictions
    # ------------------------------------------------------------------

    async def save_restriction(self, restriction: Restriction) -> None:
        async with self._session_factory() as session:
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

    async def get_restrictions(self, user_id: str, active_only: bool = True) -> list[Restriction]:
        async with self._session_factory() as session:
            stmt = select(RestrictionModel).where(RestrictionModel.user_id == user_id)
            if active_only:
                stmt = stmt.where(RestrictionModel.completed_at.is_(None))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_restriction_model_to_entity(r) for r in rows]


# ------------------------------------------------------------------
# Model-to-entity mappers
# ------------------------------------------------------------------


def _user_model_to_entity(model: UserModel) -> User:
    return User(
        id=model.id,
        telegram_id=model.telegram_id,
        name=model.name,
        language=model.language,
        timezone=model.timezone,
    )


def _goal_model_to_entity(model: GoalModel) -> Goal:
    return Goal(
        id=model.id,
        user_id=model.user_id,
        description=model.description,
        created_at=model.created_at,
        completed_at=model.completed_at,
    )


def _restriction_model_to_entity(model: RestrictionModel) -> Restriction:
    return Restriction(
        id=model.id,
        user_id=model.user_id,
        type=model.type,
        description=model.description,
        created_at=model.created_at,
        completed_at=model.completed_at,
    )


def _lab_marker_model_to_entity(model: LabMarkerModel) -> LabMarker:
    return LabMarker(
        id=model.id,
        document_id=model.document_id,
        user_id=model.user_id,
        name=model.name,
        value=model.value,
        unit=model.unit,
        reference_range=model.reference_range,
        date=model.date,
    )
