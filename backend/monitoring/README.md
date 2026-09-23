# Monitoring

A lightweight, dependency-free observability story for the deployed
backend — the Render free tier ships no hosted uptime monitor, metrics, or
alerting, so these close the gap cheaply. Two complementary layers, both
pull-based:

1. **Live, in-process metrics** — `GET /metrics` on the backend itself
   (`app/core/metrics.py` + `app/api/v1/routes/metrics.py`) emits the
   same signal the offline rollups compute, but scraped live at any moment
   via a real Prometheus text exposition format — request latency
   histograms (p50/p95/p99 via `histogram_quantile`), `http_requests_total`
   by method/path/status (dynamic ids normalized to `{id}`), per-tool
   invocation counts + latency, LLM call/token/cost totals, corrective-loop
   cap count, retrieval timeouts, and error counts by `taxonomy_category`.
   Any Prometheus/Grafana Cloud / CloudWatch-agent scraper can collect it;
   optional `METRICS_BEARER_TOKEN` auth. See `backend/monitoring/README.md`
   "Metrics endpoint" below.
2. **Offline rollups + alerts** — the scripts below, run on a schedule or
   in CI against a captured log / a live probe.

These are NOT a replacement for Prometheus/Grafana/Loki/CloudWatch: the
scripts are on-demand, pull-based, with no long-term storage. `GET
/metrics` closes the "no live data" gap — the scraper/store is still yours
to choose. See `docs/CHECKLIST.md` §10.

## Files

- `uptime_check.py` — availability probe for the backend (`/health`) and
  frontend. Reports reachability, HTTP status, latency, and a trailing
  availability percentage persisted in `uptime_state.json`. Exit 0 = all
  healthy, exit 1 = any target failed (so a scheduler can alert on non-zero).
  Also pushes a webhook alert on failure — see "Push alerts" below.
- `log_aggregate.py` — windowed rollup over a captured backend JSON log
  (same line format `eval/metrics_report.py` consumes): error rate (with
  breakdown by `taxonomy_category`), p50/p95/p99 latency, corrective-loop
  cap rate, **per-tool success rate** (from the `tool_invocation` events
  emitted by `app/services/tool_registry.py`'s `@track_tool`), token
  totals, and estimated cost. Fails (exit 1) if any of
  `--max-error-rate`, `--max-p95-latency-ms`, `--max-loop-capped-rate`, or
  `--min-tool-success-rate` is breached — thresholds act as the alerts stub.
  Also pushes a webhook alert on breach — see "Push alerts" below.
- `alert_webhook.py` — shared, dependency-free helper the two scripts
  above call: POSTs a Slack-compatible `{"text": ...}` payload via stdlib
  `urllib`. Best-effort — a broken or unset webhook never fails the check
  it's reporting through; the check's own exit code remains the real signal.

## Metrics endpoint

`GET /metrics` (see `app/api/v1/routes/metrics.py`) is the live, in-process
half of this story — the same events the offline scripts read from logs are
also bumped into a thread-safe registry at their emit site:

| Metric | Family |
|---|---|
| Request count by method/path/status | `http_requests_total` |
| Request latency (percentiles via `histogram_quantile`) | `http_request_duration_seconds` histogram |
| Tool calls by tool + success/error, plus latency | `tool_invocations_total`, `tool_invocation_duration_seconds` |
| LLM calls, tokens (prompt/completion), estimated cost | `llm_generations_total`, `llm_tokens_total`, `llm_cost_usd_total` |
| Corrective-loop cap hits | `loop_capped_total` |
| Retrieval degradation to empty results | `retrieval_timeouts_total` |
| Errors by error-taxonomy category | `errors_total` |
| Process up-time | `insightai_uptime_seconds` |

Unauthenticated by default (metrics carry no payload; scrape cost is
trivial). To require a token, set `METRICS_BEARER_TOKEN` — scrapers must
then send `Authorization: Bearer <token>`, compared constant-time
(`hmac.compare_digest`), never logged.

Example scrape:

```bash
curl -s http://localhost:8000/metrics           # unauthenticated
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/metrics
promtool check metrics <(curl -s http://localhost:8000/metrics)   # validate
```

## Usage

```bash
# Availability, once (local defaults)
python monitoring/uptime_check.py

# Live deployed endpoints
BACKEND_URL=https://agrosense-rag-backend.onrender.com \
FRONTEND_URL=https://agrosense-rag.vercel.app \
python monitoring/uptime_check.py --json

# Log aggregation over a captured uvicorn log (see eval/metrics_report.py
# for how to capture it), with thresholds
python monitoring/log_aggregate.py app.log --max-error-rate 0.05 --max-p95-latency-ms 5000
```

## Push alerts

Both `uptime_check.py` and `log_aggregate.py` accept `--alert-webhook-url`
(or the `ALERT_WEBHOOK_URL` env var) and POST a short text summary to it
via `alert_webhook.py` whenever they'd otherwise exit non-zero — a Slack
incoming webhook URL works directly; Discord's `/slack`-compatible
webhook endpoint and a Teams connector both accept the same `{"text":
...}` shape. Unset (the default): behavior is unchanged from before —
the job exit code / failed workflow run remains the alert surface, no
network call is made.

```bash
python monitoring/uptime_check.py --alert-webhook-url https://hooks.slack.com/services/...
python monitoring/log_aggregate.py app.log --alert-webhook-url https://hooks.slack.com/services/...
```

## CI / scheduling

- No scheduled CI job currently runs `uptime_check.py` — the workflow that
  did (`monitoring.yml`, every 15 minutes against the deployed AWS
  endpoints) was removed along with that deployment path. Run it manually
  against whatever's currently deployed (`docs/OPERATIONS.md` "Deploying
  to EC2"), or re-add a scheduled workflow pointed at real endpoint URLs
  if that's needed again.
- For log aggregation in CI, first capture a log artifact (uvicorn stdout)
  and then run `log_aggregate.py` on it — see `.github/workflows/ci.yml`.

## Known limitations (deliberate)

- `uptime_state.json` lives on local disk / the ephemeral Actions workspace;
  trailing availability is therefore point-in-time, not a durable SLO.
- Push alerts are un-debounced: a long-lived `--loop` run that stays down
  alerts on every iteration, not just the first transition into failure.
  Not an issue for a single CI invocation, which only checks once.
- Latency figures from `uptime_check.py` are from the probing machine's
  network, not server-side. `GET /metrics` latency is server-side but
  in-process only — its registry is per-process, so on a multi-instance
  deployment each instance reports its own counters (scrape every instance;
  no multi-instance dedup is attempted).
- The `GET /metrics` registry resets on process restart (no long-term
  storage by design) — for durable trends, scrape into a real store.
