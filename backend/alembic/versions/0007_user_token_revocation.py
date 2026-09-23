"""add users.tokens_revoked_after for JWT revocation on logout

Revision ID: 0007_user_token_revocation
Revises: 0006_pgvector_embeddings
Create Date: 2026-08-14

Backs POST /auth/logout (app/api/v1/routes/auth.py) and
user_service.decode_access_token's revocation check: a JWT whose `iat`
predates the user's tokens_revoked_after is rejected, even though it
hasn't expired yet. Null (the default) means "never logged out" — no
revocation applied, every previously-issued token stays valid until its
own expiry, matching the app's pre-existing behavior for every user who
never calls the new endpoint.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0007_user_token_revocation"
down_revision: Union[str, None] = "0006_pgvector_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("tokens_revoked_after", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "tokens_revoked_after")
