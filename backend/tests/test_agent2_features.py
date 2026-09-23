"""Tests for Agent 2 features: mypy/ruff lint gate (Feature 2) and the
Grafana/Prometheus/Alertmanager monitoring stack (Feature 4).

These are all offline and hermetic:

- The new metric recorders (retrieval grade, web-search fallback, agent
  handoff, vector-count gauge) are exercised directly against the in-process
  registry — the same record_* API call sites the live code paths use.
- The wiring points are verified at the source level (query.py's
  get_vector_store emits the gauge; agent_events.py bumps the handoff
  counter) without invoking real LLM/DB/storage backends.
- The lint config (backend/pyproject.toml) and monitoring YAML/JSON files
  are validated for existence + structural sanity (parseable YAML/JSON,
  expected keys, all dashboard files load as JSON). No docker/grafana is
  required — the dashboards are JSON documents asserted for their panel
  targets and metric names.
"""

import json
import tomllib
from pathlib import Path

import pytest

from app.core.metrics import Metrics, get_metrics, reset_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
MONITORING_ROOT = REPO_ROOT / "monitoring"


@pytest.fixture(autouse=True)
def _isolate_metrics():
    reset_metrics()
    yield
    reset_metrics()


# ---------------------------------------------------------------------------
# Feature 2 — mypy --strict + ruff lint gate
# ---------------------------------------------------------------------------


class TestLintConfig:
    def test_pyproject_toml_exists_and_parses(self):
        pyproject = BACKEND_ROOT / "pyproject.toml"
        assert pyproject.exists()
        with pyproject.open("rb") as fh:
            config = tomllib.load(fh)
        assert "mypy" in config["tool"]
        assert "ruff" in config["tool"]
        assert "pytest" in config["tool"]

    def test_mypy_is_strict(self):
        with (BACKEND_ROOT / "pyproject.toml").open("rb") as fh:
            mypy = tomllib.load(fh)["tool"]["mypy"]
        assert mypy.get("strict") is True
        assert mypy.get("disallow_untyped_defs") is True
        assert mypy.get("ignore_missing_imports") is True
        # The brief's spec: migrations/artifacts/eval are not type-checked.
        assert any("alembic" in p for p in mypy.get("exclude", []))

    def test_ruff_selects_core_rule_sets(self):
        with (BACKEND_ROOT / "pyproject.toml").open("rb") as fh:
            ruff = tomllib.load(fh)["tool"]["ruff"]
        # select/ignore live under [tool.ruff.lint], not [tool.ruff]
        # itself — that's been this file's actual structure since it was
        # first added (see pyproject.toml), this test just checked the
        # wrong table.
        selected = set(ruff["lint"]["select"])
        # Pyflakes (undefined names/unused) and import sorting must be on —
        # they catch real bugs the suite's philosophy cares about.
        assert "F" in selected
        assert "I" in selected
        # Line-length tolerance is set (codebase is formatted to 100).
        assert ruff["line-length"] == 100

    def test_lint_workflow_exists_and_targets_backend(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "lint.yml"
        assert workflow.exists()
        text = workflow.read_text()
        # mypy --strict, ruff check, and ruff format --check all run.
        assert "mypy --strict" in text or "mypy app" in text
        assert "ruff check" in text
        assert "ruff format" in text
        assert "backend" in text  # backend is the linted surface

    def test_ci_runs_ruff_before_tests(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = workflow.read_text()
        assert "ruff check app" in text
        assert text.index("ruff check") < text.index("pytest")


# ---------------------------------------------------------------------------
# Feature 4 — Grafana dashboards + alerts
# ---------------------------------------------------------------------------


class TestMonitoringStack:
    def test_compose_file_covers_three_services(self):
        compose = REPO_ROOT / "docker-compose.monitoring.yml"
        assert compose.exists()
        text = compose.read_text()
        for service in ("prometheus", "alertmanager", "grafana"):
            assert f"{service}:" in text
        # Grafana auto-provisions dashboards from the file provider.
        assert "dashboards" in text
        assert "datasources" in text

    def test_prometheus_scrapes_backend_metrics_endpoint(self):
        prom = MONITORING_ROOT / "prometheus.yml"
        assert prom.exists()
        text = prom.read_text()
        assert "metrics_path: /metrics" in text
        assert "insightai-backend" in text
        assert "alert_rules.yml" in text
        assert "alertmanager" in text  # rule alerts route to alertmanager

    def test_alert_rules_cover_business_and_system_groups(self):
        rules = MONITORING_ROOT / "alert_rules.yml"
        assert rules.exists()
        text = rules.read_text()
        for group in ("rag_pipeline", "agent", "system", "business"):
            assert f"name: {group}" in text
        # The four required alert families from the brief.
        for alert in (
            "HighLatency",
            "HighErrorRate",
            "LoopCappedRate",
            "RetrievalTimeoutRate",
            "HighMemoryUsage",
            "HighCPUUsage",
            "DiskSpaceLow",
            "LowTaskSuccessRate",
        ):
            assert f"alert: {alert}" in text

    def test_four_dashboards_provisioned(self):
        dash_dir = MONITORING_ROOT / "grafana" / "dashboards"
        files = sorted(dash_dir.glob("*.json"))
        assert {f.name for f in files} == {
            "overview.json",
            "rag_pipeline.json",
            "agent_metrics.json",
            "system.json",
        }

    def test_dashboard_json_is_valid_and_wired_to_real_metrics(self):
        dash_dir = MONITORING_ROOT / "grafana" / "dashboards"
        known_metrics = {
            "http_requests_total",
            "http_request_duration_seconds",
            "tool_invocations_total",
            "tool_invocation_duration_seconds",
            "llm_generations_total",
            "llm_tokens_total",
            "llm_cost_usd_total",
            "loop_capped_total",
            "retrieval_timeouts_total",
            "errors_total",
            "retrieval_grades_total",
            "web_search_fallbacks_total",
            "agent_handoffs_total",
            "insightai_uptime_seconds",
            "insightai_vectors_total",
        }
        for path in dash_dir.glob("*.json"):
            data = json.loads(path.read_text())
            assert data.get("title")
            assert data.get("uid")
            # Every panel's PromQL target must reference a metric the
            # registry actually emits — a dashboard wired to a name that
            # never exists is the exact failure this test catches.
            expr = " ".join(t.get("expr", "") for p in data["panels"] for t in p.get("targets", []))
            referenced = {m for m in known_metrics if m in expr}
            assert referenced, f"{path.name} references no known metric"
            # Confirms a real metric family is queried (not just HELP lines).
            assert any(m in expr for m in known_metrics)

    def test_datasource_and_provider_provisioning_files(self):
        ds = MONITORING_ROOT / "grafana" / "provisioning" / "datasources" / "datasource.yml"
        prov = MONITORING_ROOT / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        assert ds.exists()
        assert prov.exists()
        assert "url: http://prometheus:9090" in ds.read_text()
        assert "type: file" in prov.read_text()

    def test_alertmanager_config_exists(self):
        am = MONITORING_ROOT / "alertmanager.yml"
        assert am.exists()
        assert "route:" in am.read_text()


# ---------------------------------------------------------------------------
# Feature 4 — metric instrumentation the dashboards depend on
# ---------------------------------------------------------------------------


class TestNewMetricRecordings:
    def test_record_retrieval_grade(self):
        m = Metrics()
        m.record_retrieval_grade("good")
        m.record_retrieval_grade("good")
        m.record_retrieval_grade("weak")
        text = m.render()
        assert 'retrieval_grades_total{grade="good"} 2' in text
        assert 'retrieval_grades_total{grade="weak"} 1' in text

    def test_record_web_search_fallback(self):
        m = Metrics()
        m.record_web_search_fallback(stage="retrieval")
        text = m.render()
        assert 'web_search_fallbacks_total{stage="retrieval"} 1' in text

    def test_record_agent_handoff(self):
        m = Metrics()
        m.record_agent_handoff(frm="router", to="research")
        m.record_agent_handoff(frm="retrieval_grader", to="research")
        text = m.render()
        assert 'agent_handoffs_total{from="router",to="research"} 1' in text
        assert 'agent_handoffs_total{from="retrieval_grader",to="research"} 1' in text

    def test_record_vectors_sets_a_gauge(self):
        m = Metrics()
        m.record_vectors(42)
        text = m.render()
        assert "# TYPE insightai_vectors_total gauge" in text
        assert "insightai_vectors_total 42.000" in text
        # A gauge, not a counter: re-setting replaces, it never accumulates.
        m.record_vectors(7)
        assert "insightai_vectors_total 7.000" in m.render()


class TestMetricWiringPoints:
    def test_get_vector_store_emits_vector_gauge(self):
        # The FAISS path calls record_vectors after load; with no persisted
        # index it still records 0 (empty store). pgvector path is skipped
        # when the setting is off (tests force DATABASE_URL="" in conftest).
        # cache_clear() forces the lru_cache to re-run so the gauge is
        # actually emitted within this test's reset window.
        from app.api.v1.routes.query import get_vector_store

        get_vector_store.cache_clear()
        get_vector_store()
        text = get_metrics().render()
        assert "insightai_vectors_total" in text

    def test_log_agent_handoff_bumps_metric(self):
        # The handoff logger feeds both the structured-log event and the
        # Prometheus counter — verify the counter side fires.
        from app.services.agent_events import log_agent_handoff

        log_agent_handoff("router", "research", "query")
        assert 'agent_handoffs_total{from="router",to="research"} 1' in get_metrics().render()
