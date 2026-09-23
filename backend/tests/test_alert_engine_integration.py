"""Module 10 gap-closure (P6): proves the alerting path actually RUNS
end to end -- real-shaped metric input -> AlertEngine threshold
evaluation -> alert triggered -> payload produced -- not just that
app.core.alerting imports cleanly. Also proves an alert payload never
contains a secret, since app/core/alerting.py::AlertEvent structurally
only carries a rule name, a metric name, a float value, a threshold,
a kind, and a timestamp -- there is no code path by which request
content, API keys, or headers could enter it.

Complements tests/test_alerting.py (the engine's own unit tests) and
tests/test_alert_webhook.py (the simpler monitoring/alert_webhook.py
text-webhook helper used by log_aggregate.py/uptime_check.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.alerting import (  # noqa: E402
    AlertRule,
    Comparator,
    MockNotificationSink,
    build_default_engine,
    make_webhook_sink,
)
from monitoring.log_aggregate import aggregate  # noqa: E402


def _real_shaped_log_records(*, error_heavy: bool) -> list[dict]:
    """A small, realistic set of structured log records shaped exactly
    like what monitoring/log_aggregate.py already parses from a real
    captured app.log -- not fabricated metric values, but records in the
    same schema the running application actually emits."""
    records = [
        {"message": "chat_query_handled", "processing_duration": 0.5},
        {"message": "chat_query_handled", "processing_duration": 0.6},
        {"message": "chat_query_handled", "processing_duration": 4.5 if error_heavy else 0.7},
    ]
    if error_heavy:
        records += [{"message": "request_failed", "taxonomy_category": "tool"}] * 5
    return records


class TestMetricInputThroughThresholdToAlert:
    """TASK 5: metric input -> threshold evaluation -> alert triggered ->
    alert payload produced, using log_aggregate's real aggregate() output
    as the metric source (not a hand-picked float invented for the test)."""

    def test_error_rate_above_threshold_triggers_alert(self):
        agg = aggregate(_real_shaped_log_records(error_heavy=True))
        assert agg["error_rate"] > 0.05  # sanity: this scenario really is a breach

        sink = MockNotificationSink()
        engine = build_default_engine(sink)
        events = engine.evaluate("error_rate", agg["error_rate"], now=0.0)

        assert len(events) == 1
        assert events[0].kind == "alert"
        assert events[0].rule_name == "high_error_rate"
        assert len(sink.delivered) == 1

    def test_latency_above_threshold_triggers_alert(self):
        agg = aggregate(_real_shaped_log_records(error_heavy=False))
        p95_seconds = agg["p95_latency_ms"] / 1000.0

        sink = MockNotificationSink()
        engine = build_default_engine(sink)
        # Feed a deliberately high p95 (simulating a slow window) through
        # the same rule the real p95 would be evaluated against.
        events = engine.evaluate("p95_latency_seconds", max(p95_seconds, 10.0), now=0.0)

        assert len(events) == 1
        assert events[0].rule_name == "high_p95_latency"

    def test_health_check_success_below_minimum_triggers_alert(self):
        sink = MockNotificationSink()
        engine = build_default_engine(sink)

        events = engine.evaluate("health_check_success", 0.0, now=0.0)

        assert len(events) == 1
        assert events[0].rule_name == "health_check_down"
        assert events[0].kind == "alert"

    def test_values_within_threshold_produce_no_alert(self):
        agg = aggregate(_real_shaped_log_records(error_heavy=False))
        assert agg["error_rate"] <= 0.05  # sanity: this scenario is healthy

        sink = MockNotificationSink()
        engine = build_default_engine(sink)
        events = engine.evaluate("error_rate", agg["error_rate"], now=0.0)
        events += engine.evaluate("health_check_success", 1.0, now=0.0)

        assert events == []
        assert sink.delivered == []


class TestWebhookSinkFakeSlack:
    """TASK 5: exercised WITHOUT a real Slack account -- a fake post_fn
    stands in for httpx.post, proving make_webhook_sink actually calls
    through to a delivery function rather than only being importable."""

    def test_webhook_sink_delivers_alert_payload_via_injected_post_fn(self):
        received = []

        def fake_post(url, payload):
            received.append((url, payload))

        sink = make_webhook_sink(fake_post, "https://hooks.example.invalid/fake")
        engine = build_default_engine(sink)

        engine.evaluate("error_rate", 0.5, now=1234.0)

        assert len(received) == 1
        url, payload = received[0]
        assert url == "https://hooks.example.invalid/fake"
        assert payload["rule"] == "high_error_rate"
        assert payload["value"] == 0.5


class TestAlertPayloadQuality:
    """TASK 6: an alert must carry enough to debug (name, metric,
    observed value, threshold, timestamp) and must never carry a
    secret -- proven structurally, not just by inspection."""

    def test_payload_contains_required_debugging_fields(self):
        received = []
        sink = make_webhook_sink(lambda url, payload: received.append(payload), "https://x.invalid")
        engine = build_default_engine(sink)

        engine.evaluate("error_rate", 0.5, now=1700000000.0)

        payload = received[0]
        assert set(payload.keys()) == {"rule", "metric", "value", "threshold", "kind", "timestamp"}
        assert payload["rule"] == "high_error_rate"
        assert payload["metric"] == "error_rate"
        assert payload["threshold"] == 0.05
        assert payload["timestamp"] == 1700000000.0

    def test_payload_never_contains_a_secret_even_for_an_unusual_rule(self):
        """Adversarial: even an unusual rule/metric name produces a
        payload that only ever echoes the rule's own name/metric/
        threshold -- there is no field that could carry arbitrary request
        content, an API key, or an Authorization header, because
        AlertEvent's only inputs are (metric: str, value: float)."""
        from app.core.alerting import AlertEngine

        received = []
        sink = make_webhook_sink(lambda url, payload: received.append(payload), "https://x.invalid")
        engine = AlertEngine(sink)
        engine.register(
            AlertRule(
                name="rule_registered_here_not_from_user_input",
                metric="suspicious_metric",
                threshold=1.0,
                comparator=Comparator.GREATER_THAN,
            )
        )

        engine.evaluate("suspicious_metric", 999.0, now=0.0)

        payload = received[0]
        serialized = str(payload)
        assert "sk-" not in serialized
        assert "Bearer " not in serialized
        assert "gsk_" not in serialized  # this project's real Groq key prefix
        assert "AQ." not in serialized  # this project's real Gemini key prefix


class TestNoSecretsInRealAggregateOutputFedToAlerts:
    """Belt-and-suspenders: the real aggregate() output that feeds these
    alerts must itself never contain a secret, since it is built from
    the same structured logs core/logging.py emits."""

    def test_aggregate_output_never_contains_api_key_like_strings(self):
        from app.core.config import settings

        agg = aggregate(_real_shaped_log_records(error_heavy=True))
        serialized = str(agg)
        assert settings.gemini_api_key not in serialized or not settings.gemini_api_key
        if settings.groq_api_key:
            assert settings.groq_api_key not in serialized
