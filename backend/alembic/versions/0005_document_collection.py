"""add documents.collection for named document collections ("doc sets")

Revision ID: 0005_document_collection
Revises: 0004_chat_session_title
Create Date: 2026-08-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0005_document_collection"
down_revision: Union[str, None] = "0004_chat_session_title"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("collection", sa.String(length=255), nullable=True),
    )
    op.create_index(
        op.f("ix_documents_collection"), "documents", ["collection"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_collection"), table_name="documents")
    op.drop_column("documents", "collection")
