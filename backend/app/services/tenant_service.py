"""Tenant resolution and API-key mirroring for PostgreSQL persistence.

Two responsibilities, both no-ops when the DB is disabled (db_enabled()
False — see app/core/database.py):

1. `resolve_tenant(client_name)` returns (tenant_id, role) for a client,
   creating the Tenant row on first sight (idempotent). Used by auth to
   scope every request to a tenant and to gate role-restricted actions
   (see app/core/auth.py, app/api/v1/routes/documents.py's delete route).
   role is "admin" whenever client_name is in
   Settings.admin_client_names_set — checked fresh on every call, not
   cached, so promoting/demoting a client via that env var takes effect
   on next request, no DB write needed — else whatever's stored on the
   Tenant row (default "member", see app/models/db_models.py).

2. `seed_keys_from_settings()` mirrors the env-configured API keys
   (Settings.api_key_table: hash -> client_name) into the api_keys table
   at startup, so a deployment that moves to Postgres keeps its existing
   keys working while gaining durable tenant rows. The env table remains
   the fast-path authority for key validity (no DB round-trip per
   request); the DB copy exists for durability/listing and for keys that
   are added directly to the DB later.
"""

import logging
import threading

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, db_enabled
from app.models.db_models import ApiKey, Tenant

logger = logging.getLogger(__name__)

# Client -> tenant_id cache (role deliberately excluded — see resolve_tenant's
# docstring on why it's always re-checked against admin_client_names_set
# rather than cached alongside the id). Tenants are created once and never
# renamed/deleted in this app, so caching the resolved id is safe and avoids
# a DB query on every authenticated request. Guarded by a lock because auth
# runs from the thread pool.
_tenant_cache: dict[str, int] = {}
_tenant_cache_lock = threading.Lock()


def _session() -> Session:
    if SessionLocal is None:
        raise RuntimeError("Database not configured")
    return SessionLocal()


def effective_role(client_name: str, stored_role: str) -> str:
    """Public (not module-private) because app/services/user_service.py
    also calls this for JWT-authenticated users — Settings.admin_client_names
    applies to a user's email exactly the same way it applies to an
    API-key client_name, one shared admin-override mechanism either way."""
    return "admin" if client_name in settings.admin_client_names_set else stored_role


def resolve_tenant(client_name: str) -> tuple[int, str] | None:
    """Return (tenant_id, role) for a client, creating the tenant row if
    needed. Returns None when the DB is disabled."""
    if not db_enabled():
        return None

    with _tenant_cache_lock:
        cached_id = _tenant_cache.get(client_name)

    if cached_id is not None:
        with _session() as db:
            tenant = db.query(Tenant).filter(Tenant.id == cached_id).first()
            stored_role = tenant.role if tenant is not None else "member"
        return cached_id, effective_role(client_name, stored_role)

    with _session() as db:
        tenant = db.query(Tenant).filter(Tenant.slug == client_name).first()
        if tenant is not None:
            tenant_id, stored_role = tenant.id, tenant.role
        else:
            tenant = Tenant(slug=client_name, name=client_name)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
            logger.info(
                "tenant_created",
                extra={
                    "extra_fields": {
                        "client_name": client_name,
                        "tenant_id": tenant.id,
                        "role": tenant.role,
                    }
                },
            )
            tenant_id, stored_role = tenant.id, tenant.role

    with _tenant_cache_lock:
        _tenant_cache[client_name] = tenant_id
    return tenant_id, effective_role(client_name, stored_role)


def find_api_key(key_hash: str) -> tuple[str, int, str] | None:
    """Return (client_name, tenant_id, role) for a key_hash if present in
    the DB, else None. Lets DB-added keys (not in the env table)
    authenticate. role follows the same admin_client_names_set override
    as resolve_tenant()."""
    if not db_enabled():
        return None
    with _session() as db:
        api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if api_key is None:
            return None
        tenant = db.query(Tenant).filter(Tenant.id == api_key.tenant_id).first()
        stored_role = tenant.role if tenant is not None else "member"
        return (
            api_key.client_name,
            api_key.tenant_id,
            effective_role(api_key.client_name, stored_role),
        )


def seed_keys_from_settings() -> None:
    """Mirror env-configured API keys into the DB (idempotent). No-op if
    the DB is disabled. Runs at startup so existing clients get durable
    tenant rows without manual migration."""
    if not db_enabled():
        return

    with _session() as db:
        for key_hash, client_name in settings.api_key_table.items():
            tenant = db.query(Tenant).filter(Tenant.slug == client_name).first()
            if tenant is None:
                tenant = Tenant(slug=client_name, name=client_name)
                db.add(tenant)
                db.flush()

            exists = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
            if exists is None:
                db.add(ApiKey(key_hash=key_hash, client_name=client_name, tenant_id=tenant.id))

        db.commit()

    logger.info(
        "api_keys_seeded",
        extra={"extra_fields": {"key_count": len(settings.api_key_table)}},
    )
