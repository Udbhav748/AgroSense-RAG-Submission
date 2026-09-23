"""Admin-only operational routes — currently just usage analytics.
Every route here must call permissions.check_permission first."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func

from app.core import permissions
from app.core.auth import require_auth
from app.core.database import SessionLocal, db_enabled
from app.models.db_models import UsageLog
from app.models.schemas import UsageSummaryResponse, UsageSummaryRow

router = APIRouter(tags=["Admin"], dependencies=[Depends(require_auth)])


@router.get("/admin/usage-summary", response_model=UsageSummaryResponse)
def get_usage_summary(request: Request) -> UsageSummaryResponse:
    permissions.check_permission(
        request,
        permissions.ANALYTICS_VIEW,
        message="Viewing usage analytics requires the admin role.",
    )
    if not db_enabled():
        return UsageSummaryResponse(rows=[])
    # db_enabled() guarantees SessionLocal is set, but that's not something
    # mypy can follow across the function call — narrow it explicitly here.
    assert SessionLocal is not None
    with SessionLocal() as db:
        results = (
            db.query(
                func.date(UsageLog.created_at).label("day"),
                func.count(UsageLog.id).label("request_count"),
                func.avg(UsageLog.latency_ms).label("avg_latency_ms"),
            )
            .group_by(func.date(UsageLog.created_at))
            .order_by(func.date(UsageLog.created_at).desc())
            .limit(30)
            .all()
        )
    rows = [
        UsageSummaryRow(
            day=str(r.day),
            request_count=r.request_count,
            avg_latency_ms=round(r.avg_latency_ms or 0, 1),
        )
        for r in results
    ]
    return UsageSummaryResponse(rows=rows)
