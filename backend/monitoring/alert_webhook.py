"""Best-effort push-alert helper shared by uptime_check.py and
log_aggregate.py — closes the "no push notification path" gap noted in
monitoring/README.md's Known Limitations.

Posts a Slack-compatible JSON payload ({"text": ...}) via stdlib
urllib — no new dependency, matching this package's dependency-free
design (see README.md's opening line). Slack, Discord (with its
`/slack` webhook compatibility endpoint), and Microsoft Teams (via a
connector) all accept this same {"text": ...} shape, so one helper
covers the common cases without per-provider branching.

Never raises: a broken or unset webhook must never fail the check
that's trying to report through it. The check's own exit code remains
the real signal (CI failure / non-zero exit); this is strictly
additive on top of that, not a replacement for it.
"""

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10


def send_alert(webhook_url: str | None, text: str) -> bool:
    """POST {"text": text} to webhook_url. No-op (returns False) if
    webhook_url is falsy — the default, unconfigured state. Returns
    whether the POST got a 2xx response; any failure is logged and
    swallowed, never raised."""
    if not webhook_url:
        return False

    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("alert_webhook_failed", extra={"extra_fields": {"error": str(exc)}})
        return False
