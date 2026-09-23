"""Threshold-based alerting engine (Module 10 gap-closure: "automated
alerting" was previously ⚠️ -- inert without a deployment target).

Real, reusable, locally-testable infrastructure: rule definitions,
threshold evaluation, alert-event creation, debounce (no alert storm
from a metric that stays above threshold), a recovery event when the
metric returns below threshold, and a pluggable notification sink
(webhook adapter + an in-memory mock sink for tests).

Honest scope: this engine is fully implemented and testable locally. It
is NOT continuously invoked against a live production target, because
no persistent deployment exists for it to monitor -- see
`.github/workflows/health-monitor.yml` (already inert-by-design for the
same reason) and `docs/CHECKLIST.md`'s Alerting row. Wiring this engine
to a real periodic metrics poll against a live deployment is future
work, not claimed here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class Comparator(str, Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"


@dataclass(frozen=True)
class AlertRule:
    name: str
    metric: str
    threshold: float
    comparator: Comparator
    debounce_seconds: float = 0.0  # minimum time between repeated ALERT events for the same rule


@dataclass
class AlertEvent:
    rule_name: str
    metric: str
    value: float
    threshold: float
    kind: str  # "alert" | "recovery"
    timestamp: float


class NotificationSink(Protocol):
    def send(self, event: AlertEvent) -> None: ...


class MockNotificationSink:
    """In-memory sink for tests -- records every delivery attempt,
    including deliberately-injected failures, without hitting a real
    network endpoint."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.delivered: list[AlertEvent] = []
        self.failed: list[AlertEvent] = []

    def send(self, event: AlertEvent) -> None:
        if self.fail:
            self.failed.append(event)
            raise ConnectionError("mock notification sink: simulated delivery failure")
        self.delivered.append(event)


def make_webhook_sink(post_fn: Callable[[str, dict], None], url: str) -> NotificationSink:
    """Real webhook adapter: `post_fn` is typically `httpx.post`'s
    signature-compatible callable, injected so this module never imports
    httpx directly or hardcodes a URL/secret -- callers supply both,
    keeping this engine transport-agnostic and testable without network
    access."""

    class _WebhookSink:
        def send(self, event: AlertEvent) -> None:
            post_fn(
                url,
                {
                    "rule": event.rule_name,
                    "metric": event.metric,
                    "value": event.value,
                    "threshold": event.threshold,
                    "kind": event.kind,
                    "timestamp": event.timestamp,
                },
            )

    return _WebhookSink()


class AlertEngine:
    """Stateful threshold evaluator. Call `evaluate(metric, value)` once
    per observation; it looks up every rule registered for that metric,
    compares against the threshold, and emits an AlertEvent (via the
    sink) only on a state TRANSITION -- crossing into alert, or
    recovering back below threshold -- never repeatedly for a metric
    that stays above threshold within a rule's debounce window. A
    notification-sink failure is caught and logged, never allowed to
    propagate and take down the caller's request path.
    """

    def __init__(self, sink: NotificationSink) -> None:
        self._sink = sink
        self._rules: dict[str, list[AlertRule]] = {}
        self._active: dict[str, float] = {}  # rule_name -> time first alerted
        self._last_alert_at: dict[str, float] = {}
        self.audit_log: list[dict] = []

    def register(self, rule: AlertRule) -> None:
        self._rules.setdefault(rule.metric, []).append(rule)

    def _breaches(self, rule: AlertRule, value: float) -> bool:
        if rule.comparator == Comparator.GREATER_THAN:
            return value > rule.threshold
        return value < rule.threshold

    def evaluate(self, metric: str, value: float, *, now: float | None = None) -> list[AlertEvent]:
        now = now if now is not None else time.time()
        emitted: list[AlertEvent] = []
        for rule in self._rules.get(metric, []):
            breached = self._breaches(rule, value)
            is_active = rule.name in self._active

            if breached and not is_active:
                self._active[rule.name] = now
                self._last_alert_at[rule.name] = now
                emitted.append(self._emit(rule, value, "alert", now))
            elif breached and is_active:
                last = self._last_alert_at.get(rule.name, 0.0)
                if now - last >= rule.debounce_seconds:
                    self._last_alert_at[rule.name] = now
                    emitted.append(self._emit(rule, value, "alert", now))
                # else: still breached, within debounce window -- no repeat alert (no storm)
            elif not breached and is_active:
                del self._active[rule.name]
                emitted.append(self._emit(rule, value, "recovery", now))
            # not breached and not active: nothing to do

        return emitted

    def _emit(self, rule: AlertRule, value: float, kind: str, now: float) -> AlertEvent:
        event = AlertEvent(rule_name=rule.name, metric=rule.metric, value=value, threshold=rule.threshold, kind=kind, timestamp=now)
        self.audit_log.append(
            {"rule": rule.name, "metric": rule.metric, "value": value, "threshold": rule.threshold, "kind": kind, "timestamp": now}
        )
        try:
            self._sink.send(event)
        except Exception as exc:  # noqa: BLE001
            # A notification-sink failure must never break the caller's request path,
            # and must never be silently swallowed without a trace either.
            logger.warning(
                "alert_delivery_failed",
                extra={"extra_fields": {"rule": rule.name, "metric": rule.metric, "error_type": type(exc).__name__}},
            )
        return event


# Default rules matching the thresholds already documented/used elsewhere in this
# project (docs/RAG_BENCHMARK_REPORT.md's 3s latency target,
# .github/workflows/health-monitor.yml's own 3s threshold) -- not invented fresh here.
DEFAULT_RULES = [
    AlertRule(name="high_error_rate", metric="error_rate", threshold=0.05, comparator=Comparator.GREATER_THAN, debounce_seconds=60.0),
    AlertRule(name="high_p95_latency", metric="p95_latency_seconds", threshold=3.0, comparator=Comparator.GREATER_THAN, debounce_seconds=60.0),
    AlertRule(name="health_check_down", metric="health_check_success", threshold=0.5, comparator=Comparator.LESS_THAN, debounce_seconds=30.0),
    AlertRule(name="cost_threshold_exceeded", metric="estimated_cost_usd_per_request", threshold=0.01, comparator=Comparator.GREATER_THAN, debounce_seconds=300.0),
]


def build_default_engine(sink: NotificationSink) -> AlertEngine:
    engine = AlertEngine(sink)
    for rule in DEFAULT_RULES:
        engine.register(rule)
    return engine
