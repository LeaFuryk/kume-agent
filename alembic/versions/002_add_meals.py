"""Add meals table

Revision ID: 002
Revises: 001
Create Date: 2026-04-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "meals",
        sa.Column("id", sa.VARCHAR, primary_key=True),
        sa.Column("user_id", sa.VARCHAR, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("description", sa.TEXT, nullable=False),
        sa.Column("calories", sa.FLOAT, nullable=False),
        sa.Column("protein_g", sa.FLOAT, nullable=False),
        sa.Column("carbs_g", sa.FLOAT, nullable=False),
        sa.Column("fat_g", sa.FLOAT, nullable=False),
        sa.Column("fiber_g", sa.FLOAT, nullable=False),
        sa.Column("sodium_mg", sa.FLOAT, nullable=False),
        sa.Column("sugar_g", sa.FLOAT, nullable=False),
        sa.Column("saturated_fat_g", sa.FLOAT, nullable=False),
        sa.Column("cholesterol_mg", sa.FLOAT, nullable=False),
        sa.Column("confidence", sa.FLOAT, nullable=False),
        sa.Column("image_present", sa.BOOLEAN, nullable=False, server_default="false"),
        sa.Column("logged_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("meals")
