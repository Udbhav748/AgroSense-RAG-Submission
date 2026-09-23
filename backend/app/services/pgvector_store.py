"""PostgreSQL pgvector-backed implementation of the VectorStore interface.

Stores chunk embeddings in a Postgres table with a pgvector column
(see alembic migration 0006 / app/core/database.py), giving two things the
FAISS index can't scale to:

- native remove-by-id delete_document() — a single DELETE ... WHERE
  document_id, instead of FAISS's rebuild-from-reconstructed-vectors;
- tenant-scoped vector filtering as a SQL WHERE clause, the same
  multi-tenant pattern every other relational model in this app uses.

Use in place of FAISSVectorStore by setting PGVECTOR_ENABLED=true (and
DATABASE_URL, since pgvector lives in Postgres) — get_vector_store() in
query.py makes the swap. FAISS remains the default; this is the opt-in
migration path.

Same contract as FAISSVectorStore: callers must pass L2-normalized
embeddings (see embedding_service) so inner product == cosine similarity.
Scores use pgvector's `<=>` cosine-distance operator, reported as
1 - distance (so higher is more similar, on the same rough 0-1 scale as
FAISS's inner-product scores, which callers already threshold against).

Vector construction uses Postgres's vector literal cast
(CAST(:vec AS vector)) rather than a pgvector-specific Python adapter —
that keeps the store driver-agnostic and avoids requiring pgvector's
SQLAlchemy type machinery; it only needs the pgvector extension to exist
in the target database (created by migration 0006).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from app.core.config import settings
from app.core.exceptions import (
    DatabaseNotConfiguredError,
    EmbeddingDimensionMismatchError,
)
from app.models.document import EmbeddedChunk, RetrievedChunk
from app.services.vector_store import VectorStore

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine, RowMapping

logger = logging.getLogger(__name__)

_TABLE = "document_embeddings"


class PgvectorVectorStore(VectorStore):
    def __init__(self, table_name: str | None = None):
        self.table_name = table_name or settings.pgvector_table_name
        # Engine is imported lazily inside methods (rather than at module
        # import) so importing this module never fails when DATABASE_URL is
        # empty — the store is only ever constructed when pgvector is
        # enabled, which already requires a DB.
        self._engine: Engine | None = None

    def _connect(self) -> Connection:
        from app.core import database

        if not database.db_enabled():
            raise DatabaseNotConfiguredError(
                "PGVECTOR_ENABLED requires DATABASE_URL to be set — there is "
                "nowhere to store vectors without a Postgres database."
            )
        if self._engine is None:
            # db_enabled() being True guarantees database.engine is set
            # (engine/SessionLocal are always assigned together — see
            # app/core/database.py), but mypy can't see that invariant
            # across the db_enabled() call, so it's re-checked explicitly.
            engine = database.engine
            if engine is None:
                raise DatabaseNotConfiguredError(
                    "PGVECTOR_ENABLED requires DATABASE_URL to be set — there is "
                    "nowhere to store vectors without a Postgres database."
                )
            self._engine = engine
        return self._engine.connect()

    def _query_vector(self, embedding: list[float]) -> str:
        return "[" + ",".join(repr(float(v)) for v in embedding) + "]"

    def _record_to_chunk(self, row: RowMapping, score: float) -> RetrievedChunk:
        metadata = dict(row["metadata"] or {})
        text_content = metadata.pop("text", row.get("text") or "")
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=text_content,
            score=score,
            metadata=metadata,
        )

    # --- VectorStore ABC -------------------------------------------------

    def create_index(self, dimension: int) -> None:
        """Ensure the embeddings table exists with a vector column of the
        right dimension. The table is created by Alembic migration 0006 at
        migration time (with Settings.pgvector_dimensions); this is a
        runtime check + graceful bootstrap for databases where the
        migration hasn't run (e.g. tests), matching FAISS's create_index()
        role of "establish the empty index before adding to it"."""
        with self._connect() as conn:
            # Verify pgvector extension availability.
            ext = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
            if not ext:
                # Attempt to create the extension (superuser may be needed;
                # on managed Postgres it's preinstalled). If it fails, raise
                # a clear dimension/store error rather than a raw SQL one.
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                except Exception as exc:
                    raise EmbeddingDimensionMismatchError(
                        "pgvector extension unavailable — is the `pgvector` "
                        f"Postgres extension installed in this database? ({exc})"
                    ) from exc

            table_exists = conn.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_name = :table"),
                {"table": self.table_name},
            ).scalar()
            if table_exists:
                # Verify the existing vector column's dimension matches.
                dim = conn.execute(
                    text(
                        "SELECT vector_dimensions(atttypmod) "
                        "FROM pg_attribute "
                        "WHERE attrelid = :table::regclass AND attname = 'embedding'"
                    ),
                    {"table": self.table_name},
                ).scalar()
                if dim is not None and dim != dimension:
                    raise EmbeddingDimensionMismatchError(
                        f"pgvector column 'embedding' in {self.table_name} has "
                        f"dimension {dim}, but {dimension} was expected. Re-run "
                        "the 0006 migration or alter the column to match the "
                        "embedding model's output size."
                    )
                return

            # Bootstrap: table missing (migration not run) — create it.
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                    "  id BIGSERIAL PRIMARY KEY,"
                    "  chunk_id TEXT NOT NULL UNIQUE,"
                    "  document_id TEXT NOT NULL,"
                    "  tenant_id INTEGER,"
                    "  chunk_index INTEGER NOT NULL DEFAULT 0,"
                    "  text TEXT NOT NULL,"
                    "  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,"
                    f"  embedding vector({dimension}) NOT NULL"
                    ")"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS ix_{self.table_name}_document_id "
                    f"ON {self.table_name} (document_id)"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS ix_{self.table_name}_tenant_id "
                    f"ON {self.table_name} (tenant_id)"
                )
            )
            conn.commit()
            logger.info(
                "pgvector_table_created",
                extra={"extra_fields": {"table": self.table_name, "dimension": dimension}},
            )

    def add_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        dimension = settings.pgvector_dimensions
        for chunk in embedded_chunks:
            if len(chunk.embedding) != dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Embedding for chunk {chunk.chunk_id} has dimension "
                    f"{len(chunk.embedding)}, but the pgvector column expects "
                    f"{dimension}."
                )

        start = time.perf_counter()
        tenant_id = None
        chunk_index = 0
        with self._connect() as conn:
            for chunk in embedded_chunks:
                tenant_id = chunk.metadata.get("tenant_id")
                chunk_index = chunk.metadata.get("chunk_index", 0)
                conn.execute(
                    text(
                        f"INSERT INTO {self.table_name} "
                        "(chunk_id, document_id, tenant_id, chunk_index, text, metadata, embedding) "
                        "VALUES (:chunk_id, :document_id, :tenant_id, :chunk_index, :text, :metadata, "
                        "CAST(:embedding AS vector)) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET "
                        "  document_id = EXCLUDED.document_id,"
                        "  tenant_id = EXCLUDED.tenant_id,"
                        "  chunk_index = EXCLUDED.chunk_index,"
                        "  text = EXCLUDED.text,"
                        "  metadata = EXCLUDED.metadata,"
                        "  embedding = EXCLUDED.embedding"
                    ),
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "tenant_id": tenant_id,
                        "chunk_index": chunk_index,
                        "text": chunk.metadata.get("text", ""),
                        "metadata": json.dumps(chunk.metadata),
                        "embedding": self._query_vector(chunk.embedding),
                    },
                )
            conn.commit()

        processing_duration = time.perf_counter() - start
        logger.info(
            "pgvector_vectors_added",
            extra={
                "extra_fields": {
                    "vectors_added": len(embedded_chunks),
                    "table": self.table_name,
                    "processing_duration": round(processing_duration, 4),
                }
            },
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if len(query_vector) != settings.pgvector_dimensions:
            raise EmbeddingDimensionMismatchError(
                f"Query vector has dimension {len(query_vector)}, but the "
                f"pgvector column expects {settings.pgvector_dimensions}."
            )
        if top_k <= 0:
            return []

        filters = []
        params: dict[str, Any] = {
            "query_vec": self._query_vector(query_vector),
            "top_k": top_k,
        }
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if document_ids is not None:
            filters.append("document_id = ANY(:document_ids)")
            params["document_ids"] = list(document_ids)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = text(
            f"SELECT chunk_id, document_id, metadata, "
            f"1 - (embedding <=> CAST(:query_vec AS vector)) AS score "
            f"FROM {self.table_name} {where_clause} "
            f"ORDER BY embedding <=> CAST(:query_vec AS vector) "
            f"LIMIT :top_k"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._record_to_chunk(row, float(row["score"])) for row in rows]

    def search_bm25(
        self,
        query: str,
        top_k: int,
        tenant_id: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Lexical counterpart to FAISS's BM25 search, backed by Postgres
        full-text search (tsvector) instead of a rank-bm25 index. Postgres
        FTS has no exact BM25 scoring, but it is a genuine lexical signal
        (tokenized, stemming-aware) that hybrid_search can fuse with the
        semantic side — the pgvector analog of search_bm25. Same filtering
        semantics as search()."""
        if not query.strip() or top_k <= 0:
            return []

        filters = ["to_tsvector('english', text) @@ plainto_tsquery('english', :qtext)"]
        params: dict[str, Any] = {"qtext": query, "top_k": top_k}
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if document_ids is not None:
            filters.append("document_id = ANY(:document_ids)")
            params["document_ids"] = list(document_ids)

        sql = text(
            f"SELECT chunk_id, document_id, metadata, "
            f"ts_rank(to_tsvector('english', text), plainto_tsquery('english', :qtext)) AS score "
            f"FROM {self.table_name} WHERE {' AND '.join(filters)} "
            f"ORDER BY score DESC LIMIT :top_k"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._record_to_chunk(row, float(row["score"])) for row in rows]

    def delete_document(self, document_id: str, tenant_id: int | None = None) -> int:
        filters = ["document_id = :document_id"]
        params: dict[str, Any] = {"document_id": document_id}
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        with self._connect() as conn:
            result = conn.execute(
                text(f"DELETE FROM {self.table_name} WHERE {' AND '.join(filters)}"),
                params,
            )
            conn.commit()
            removed_count = int(result.rowcount)
        if removed_count:
            logger.info(
                "pgvector_document_deleted",
                extra={
                    "extra_fields": {
                        "document_id": document_id,
                        "vectors_removed": removed_count,
                        "table": self.table_name,
                    }
                },
            )
        return removed_count

    def get_chunks_by_document(
        self, document_id: str, tenant_id: int | None = None
    ) -> list[RetrievedChunk]:
        filters = ["document_id = :document_id"]
        params: dict[str, Any] = {"document_id": document_id}
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        sql = text(
            f"SELECT chunk_id, document_id, metadata "
            f"FROM {self.table_name} WHERE {' AND '.join(filters)} "
            f"ORDER BY chunk_index"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._record_to_chunk(row, 1.0) for row in rows]

    def list_document_ids(self, tenant_id: int | None = None) -> list[str]:
        filters = []
        params: dict[str, Any] = {}
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = text(
            f"SELECT DISTINCT document_id FROM {self.table_name} {where_clause} "
            f"ORDER BY document_id"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).scalars().all()
        return list(rows)

    def first_chunk_vector(
        self, document_id: str, tenant_id: int | None = None
    ) -> list[float] | None:
        filters = ["document_id = :document_id"]
        params: dict[str, Any] = {"document_id": document_id}
        if tenant_id is not None:
            filters.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        sql = text(
            f"SELECT CAST(embedding AS float8[]) AS embedding FROM {self.table_name} "
            f"WHERE {' AND '.join(filters)} ORDER BY chunk_index LIMIT 1"
        )
        with self._connect() as conn:
            row = conn.execute(sql, params).mappings().first()
        if row is None:
            return None
        return list(row["embedding"])

    def save(self) -> None:
        """No-op for pgvector: every add/delete is already committed to
        Postgres, so there's no separate persistence step (FAISS's save()
        writes its files; here the DB is the persistence). Present to
        satisfy the VectorStore ABC; callers (upload/delete routes) keep
        working unchanged."""

    def load(self) -> None:
        """No-op for pgvector: the store reads live from Postgres on every
        query, so there's nothing to load into memory (FAISS's load() reads
        its files; here the DB is the persistence). Present to satisfy the
        VectorStore ABC; get_vector_store() calls it unconditionally."""

    def total_vectors(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar() or 0)
