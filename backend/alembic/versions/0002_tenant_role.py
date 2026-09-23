"""add tenants.role for minimal RBAC (admin/member)

Revision ID: 0002_tenant_role
Revises: 0001_initial
Create Date: 2026-08-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002_tenant_role"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("role", sa.String(length=50), nullable=False, server_default="member"),
    )


def downgrade() -> None:
    op.drop_column("tenants", "role")
