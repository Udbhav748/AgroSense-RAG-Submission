"""SQLAlchemy ORM models for PostgreSQL persistence.

Only used when Settings.database_url is set (see app/core/database.py).
Column types target PostgreSQL; if the URL points elsewhere, these may
need adjusting. All tables are created by init_db() (create_all) and
tracked for real migrations via Alembic (backend/alembic/).
"""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Tenant(Base):
    """A workspace: an isolated unit of ownership for documents, sessions,
    and usage. Each authenticated client (API key) resolves to exactly one
    tenant; every row the client creates is scoped to that tenant."""

    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable external identifier (e.g. the client name from the API key
    # table), used to resolve a request to its tenant without coupling the
    # wire format to the DB's integer PK.
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # "admin" | "member". Defaults to "member" (the safe default — no
    # capability without being explicitly granted) rather than "admin",
    # unlike every other DB-backed field here. New tenants can be granted
    # admin without any DB access via Settings.admin_client_names (see
    # tenant_service.resolve_tenant) — checked live on every resolve, so
    # promoting/demoting a client is just an env var change, no migration
    # or manual SQL needed.
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="tenant")
    documents: Mapped[list["Document"]] = relationship(back_populates="tenant")
    users: Mapped[list["User"]] = relationship(back_populates="tenant")


class ApiKey(Base):
    """A hashed API key bound to a tenant. Only the SHA-256 hash is stored
    (matching the env-based auth in app/core/auth.py); the plaintext key is
    never persisted."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    client_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    tenant: Mapped["Tenant"] = relationship(back_populates="api_keys")


class User(Base):
    """An individual login-based account, distinct from a Tenant/ApiKey
    (a "client"). Each self-signup User gets its own personal Tenant
    (1:1, created alongside the User row at signup) — this is what makes
    "fully private per user" free: Document/ChatSession are already
    scoped by tenant_id, so a dedicated tenant per user gives full
    privacy through the exact same columns and ownership-check code
    that already exist for API-key clients, no schema change to either
    table needed. Role lives on Tenant, not here — resolve_tenant()'s
    existing admin/member logic (Settings.admin_client_names) applies
    unchanged to a JWT-authenticated user's tenant."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Null until the first POST /auth/logout (see user_service.
    # revoke_all_tokens). decode_access_token treats any JWT whose `iat`
    # predates this timestamp as invalid — a stateless JWT can't be
    # revoked any other way, and this needs no per-token bookkeeping
    # (one column, not a growing blocklist table): every token issued
    # before the user's most recent logout dies at once, which is the
    # simple, correct semantic for an app with no refresh-token/session
    # list.
    tokens_revoked_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    tenant: Mapped["Tenant"] = relationship(back_populates="users")


class Document(Base):
    """Metadata for a single uploaded document. The actual chunks/vectors
    live in the (shared) FAISS vector store; this row records who owns the
    document and what processing produced — the durable, queryable record
    that survives the ephemeral filesystem."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_embeddings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pages_ocred: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Optional user-assigned tag grouping documents into a named collection
    # ("doc set") that chats can be scoped to (Agent 3.1). Null = no group.
    collection: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    tenant: Mapped["Tenant"] = relationship(back_populates="documents")


class ChatSession(Base):
    """A chat session (conversation) owned by a tenant. Turns are stored
    separately in ChatTurn for bounded retrieval (max_turns per session,
    mirroring InMemorySessionStore's max_turns_per_session)."""

    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), nullable=False, index=True)
    # Populated from the first user turn (truncated) once one exists —
    # null for a session with no turns yet, or one created before this
    # column existed. Purely a display label for a history list; nothing
    # else derives meaning from it.
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    turns: Mapped[list["ChatTurn"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatTurn.id"
    )


class ChatTurn(Base):
    """A single user/assistant exchange within a chat session."""

    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="turns")


class UsageLog(Base):
    """One row per API request, written best-effort by the usage logging
    middleware (see app/main.py and app/services/usage_service.py). Tenant
    is null for requests that failed auth (no client resolved)."""

    __tablename__ = "usage_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    tenant_id: Mapped[int | None] = mapped_column(
        ForeignKey("tenants.id"), nullable=True, index=True
    )
    client_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
