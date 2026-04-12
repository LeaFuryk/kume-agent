"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("telegram_id", sa.BIGINT, unique=True, nullable=False),
        sa.Column("name", sa.VARCHAR, nullable=True),
        sa.Column("language", sa.VARCHAR, nullable=False, server_default="en"),
        sa.Column("timezone", sa.VARCHAR, nullable=False, server_default="UTC"),
    )

    op.create_table(
        "goals",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("user_id", sa.VARCHAR, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("description", sa.TEXT, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    op.create_table(
        "restrictions",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("user_id", sa.VARCHAR, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("type", sa.VARCHAR, nullable=False),
        sa.Column("description", sa.TEXT, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("user_id", sa.VARCHAR, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("type", sa.VARCHAR, nullable=False),
        sa.Column("filename", sa.VARCHAR, nullable=False),
        sa.Column("summary", sa.TEXT, nullable=False),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )

    op.create_table(
        "lab_markers",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("document_id", sa.VARCHAR, sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("user_id", sa.VARCHAR, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.VARCHAR, nullable=False),
        sa.Column("value", sa.FLOAT, nullable=False),
        sa.Column("unit", sa.VARCHAR, nullable=False),
        sa.Column("reference_range", sa.VARCHAR, nullable=False),
        sa.Column("date", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("lab_markers")
    op.drop_table("documents")
    op.drop_table("restrictions")
    op.drop_table("goals")
    op.drop_table("users")
