"""Tests for app/core/alerting.py -- the Module 10 threshold-based
alerting engine. Fully offline/deterministic.
"""

from app.core.alerting import (
    AlertEngine,
    AlertRule,
    Comparator,
    MockNotificationSink,
)


def test_metric_below_threshold_produces_no_alert():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN))

    events = engine.evaluate("error_rate", 0.01, now=0.0)

    assert events == []
    assert sink.delivered == []


def test_metric_crossing_threshold_triggers_one_alert():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN))

    events = engine.evaluate("error_rate", 0.10, now=0.0)

    assert len(events) == 1
    assert events[0].kind == "alert"
    assert events[0].value == 0.10
    assert len(sink.delivered) == 1


def test_repeated_threshold_crossing_within_debounce_does_not_storm():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN, debounce_seconds=60.0))

    engine.evaluate("error_rate", 0.10, now=0.0)
    engine.evaluate("error_rate", 0.12, now=5.0)  # still breached, within debounce window
    engine.evaluate("error_rate", 0.15, now=10.0)  # still within debounce window

    assert len(sink.delivered) == 1  # only the FIRST crossing alerted


def test_repeated_crossing_beyond_debounce_alerts_again():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN, debounce_seconds=60.0))

    engine.evaluate("error_rate", 0.10, now=0.0)
    engine.evaluate("error_rate", 0.10, now=61.0)  # past the debounce window, still breached

    assert len(sink.delivered) == 2


def test_metric_recovery_emits_recovery_event():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN))

    engine.evaluate("error_rate", 0.10, now=0.0)
    events = engine.evaluate("error_rate", 0.01, now=10.0)

    assert len(events) == 1
    assert events[0].kind == "recovery"
    assert len(sink.delivered) == 2  # 1 alert + 1 recovery


def test_recovery_then_re_breach_alerts_again_immediately():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN, debounce_seconds=60.0))

    engine.evaluate("error_rate", 0.10, now=0.0)  # alert
    engine.evaluate("error_rate", 0.01, now=1.0)  # recovery
    events = engine.evaluate("error_rate", 0.10, now=2.0)  # re-breach, well within old debounce window

    assert len(events) == 1
    assert events[0].kind == "alert"  # a fresh alert, not suppressed by the old debounce state


def test_notification_sink_failure_is_handled_safely():
    """A failing sink must not raise out of evaluate() -- alert delivery
    failure must never break the caller's request path."""
    sink = MockNotificationSink(fail=True)
    engine = AlertEngine(sink)
    engine.register(AlertRule("high_error_rate", "error_rate", 0.05, Comparator.GREATER_THAN))

    events = engine.evaluate("error_rate", 0.10, now=0.0)  # must not raise

    assert len(events) == 1  # the event is still returned/logged even though delivery failed
    assert len(sink.failed) == 1
    assert engine.audit_log[-1]["kind"] == "alert"  # audit trail exists even on delivery failure


def test_less_than_comparator_for_health_check_style_metrics():
    sink = MockNotificationSink()
    engine = AlertEngine(sink)
    engine.register(AlertRule("health_check_down", "health_check_success", 0.5, Comparator.LESS_THAN))

    below = engine.evaluate("health_check_success", 0.0, now=0.0)
    above = engine.evaluate("health_check_success", 1.0, now=1.0)

    assert below[0].kind == "alert"
    assert above[0].kind == "recovery"


def test_default_engine_registers_documented_rules():
    from app.core.alerting import build_default_engine

    sink = MockNotificationSink()
    engine = build_default_engine(sink)

    assert "error_rate" in engine._rules
    assert "p95_latency_seconds" in engine._rules
    assert "health_check_success" in engine._rules
    assert "estimated_cost_usd_per_request" in engine._rules
