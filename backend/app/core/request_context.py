"""Per-request context propagated via contextvars.

request_id_var holds the current request's X-Request-ID for the duration
of that request/response cycle, set by the request-ID middleware in
main.py. JSONFormatter (core/logging.py) reads it so every log line
emitted while handling a request carries the same request_id
automatically — call sites never pass it explicitly.

client_name_var holds the authenticated caller's name for the duration of
the request, set by require_auth/require_api_key (core/auth.py). It lets
deep layers that don't receive the Request object (e.g. the RAG service's
web-search approval gate) attribute a human-approval request to the
actual caller instead of "unknown".
"""

from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
client_name_var: ContextVar[str | None] = ContextVar("client_name", default=None)


def set_client_name(client_name: str | None) -> None:
    """Tag the current request with the authenticated client name."""
    client_name_var.set(client_name)


def get_client_name() -> str | None:
    """Authenticated client name for the current request, or None."""
    return client_name_var.get()
