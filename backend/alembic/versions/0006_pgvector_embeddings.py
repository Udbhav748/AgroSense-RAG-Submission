"""create document_embeddings table for the pgvector vector store

Revision ID: 0006_pgvector_embeddings
Revises: 0005_document_collection
Create Date: 2026-08-13

Creates the pgvector extension (if the DB's postgres user has rights; on
managed Postgres the extension is usually preinstalled — PgvectorVectorStore
attempts CREATE EXTENSION IF NOT EXISTS at create_index() time as well) and
the document_embeddings table the PGVECTOR_ENABLED store reads/writes.

The table mirrors the FAISS metadata record shape (chunk_id/document_id/
tenant_id/chunk_index/text/metadata), plus a vector(N) column. One table for
every tenant, with tenant_id a nullable column (NULL = legacy/untagged
chunks, same semantics FAISS metadata uses) — the app's standard
multi-tenancy pattern, applied to vectors.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0006_pgvector_embeddings"
down_revision: Union[str, None] = "0005_document_collection"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match Settings.pgvector_dimensions (default 384 for
# all-MiniLM-L6-v2). The store's create_index() validates the column's
# actual dimension against the configured one and raises a clear error on
# mismatch rather than a Postgres-level failure.
_DIMENSION = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_embeddings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("chunk_id", sa.Text(), nullable=False, unique=True),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding", Vector(_DIMENSION), nullable=False),
    )
    op.create_index(
        op.f("ix_document_embeddings_document_id"),
        "document_embeddings",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_embeddings_tenant_id"),
        "document_embeddings",
        ["tenant_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_document_embeddings_tenant_id"), table_name="document_embeddings")
    op.drop_index(op.f("ix_document_embeddings_document_id"), table_name="document_embeddings")
    op.drop_table("document_embeddings")