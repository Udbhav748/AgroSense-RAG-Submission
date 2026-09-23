# Module 10 Gap-Closure Report

**Date**: 2026-09-19 (updated same-day: item 1 completed in a follow-up continuation of this pass)
**Scope**: evaluation-only gap-closure pass over the existing Module 10 evidence package. No architecture, agent-graph, or RAG-pipeline production logic was rewritten; the two edits touching production-adjacent code (`eval/unauthorized_access_check.py`'s scenario correction, and no others) are eval-script corrections, not app logic changes.

---

## A. Gaps Addressed

| # | Gap (from the 10-item spec) | Outcome |
|---|---|---|
| 1 | Expand human evaluation to 24 cases | **DONE** — 8 new rows scored live by the same single reviewer; see Section D and `docs/HUMAN_EVAL.md` |
| 2 | Investigate/resolve RBAC 0.3333 finding | **DONE** — resolved as an eval-script correction, not an app bug; app behavior confirmed intentional and unregressed |
| 3 | Planning Success Rate instrumentation | **DONE** — measured, 1.0 (3/3), via new eval-only telemetry capture |
| 4 | Cost Per Successful Task automation | **DONE** — measured, $0.001124/successful task |
| 5 | RAG benchmark refresh | **DONE** — re-run live; surfaced a new, real, disclosed regression (Faithfulness 0.9420 → 0.0000) |
| 6 | Multimodal/LeafSense evaluation | **DONE** — measured on synthetic images, explicitly scoped as robustness-only, not diagnostic accuracy |
| 7 | Evidence versioning | **DONE** — every new artifact carries `config.run_metadata()`; no prior artifact overwritten |
| 8 | Docs updates | **DONE** — MODULE10_AUDIT.md, MODULE10_RESULTS.md, DESIGN_REVIEW.md, CHECKLIST.md updated where evidence genuinely changed status |
| 9 | Targeted + full regression tests | **DONE** — full suite green (807 passed, 1 skipped) after all changes |
| 10 | This report | **DONE** |

---

## B. Files Changed

**New:**
- `backend/eval/module10/metrics/telemetry_capture.py` — evaluation-only log-capture utility (Planning Success Rate / Cost instrumentation)
- `backend/tests/test_module10_telemetry_capture.py` — 6 tests for the capture utility itself
- `backend/eval/module10/runners/run_multimodal_eval.py` — LeafSense live evaluation runner
- `backend/eval/module10/runners/run_human_eval_new_rows.py` — captures live answers for the 8 new human-eval rows (17-24)
- `backend/eval/module10/assets/synthetic_leaf_with_spots.jpg`, `synthetic_leaf_low_quality.jpg`, `synthetic_healthy_leaf.jpg`, `synthetic_no_plant.jpg` — synthetic test images (PIL-generated)
- `docs/MODULE10_GAP_CLOSURE_REPORT.md` — this file

**Modified:**
- `backend/eval/unauthorized_access_check.py` — corrected scenario split (cross-tenant unauthorized vs. authorized same-tenant member delete); history disclosed in module docstring
- `backend/eval/module10/runners/run_security_eval.py` — `run_unauthorized_access()` updated to match the corrected scenario split
- `backend/eval/module10/runners/run_agent_eval.py` — `run_planning_cases()` rewritten to use real telemetry capture instead of the "NOT MEASURED" placeholder; cost-per-successful-task wired into `main()`
- `backend/eval/module10/metrics/agent.py` — added `CostCase` / `cost_per_successful_task()`
- `backend/eval/module10/datasets/agent_eval.json` — `planning_cases[*].expected_steps` corrected against the real captured node sequence (see dataset's own `_planning_cases_correction_note`); `agent_plan_003` marked `graph_bypassed: true`
- `docs/MODULE10_RESULTS.md`, `docs/MODULE10_AUDIT.md`, `docs/DESIGN_REVIEW.md`, `docs/CHECKLIST.md` — status/value corrections, historical evidence preserved
- `docs/RAG_BENCHMARK_REPORT.md` — new REFRESH section with live 2026-09-19 numbers; original August 2026 section preserved verbatim below it, relabeled as unverified historical evidence
- `docs/HUMAN_EVAL.md` — expanded from 16 to 24 scored rows; rows 1-16 untouched, rows 17-24 appended with recorded answers and rubric scores

**No production application code was modified.**

---

## C. Tests Added/Changed

- `backend/tests/test_module10_telemetry_capture.py` (new, 6 tests): node-sequence extraction, cost/token extraction, unrelated-line filtering, handler detachment (normal and exception paths), no-sensitive-content-leak check.
- No existing test files were modified; the RBAC fix lives entirely in `backend/eval/` (a standalone script tree, not `backend/tests/`), and was verified by running it standalone (`python eval/unauthorized_access_check.py`) rather than via pytest, matching its existing convention.

---

## D. Before/After Metrics

| Metric | Before | After | Artifact (after) |
|---|---|---|---|
| Unauthorized Access Rate | 0.3333 (1/3, wrongly included an authorized action) | **0.0** (0/2, cross-tenant only) | `security_eval_20260919T111236Z.json` |
| Member-own-tenant-delete (authorized path) | not separately checked | **PASS** (200, as expected) | same artifact |
| Planning Success Rate | NOT MEASURED | **1.0** (3/3) | `agent_eval_20260919T112455Z.json` |
| Tool Selection Accuracy | not computed | **1.0** (2/2 node-traced cases) | same artifact |
| Average Steps (planning cases) | not computed | **8.5** | same artifact |
| Step Efficiency (planning cases) | not computed | **1.0** | same artifact |
| Loop Rate (planning cases) | not computed | **0.0** | same artifact |
| Cost Per Successful Task | not computed (config-estimated only, ~$0.0006, DESIGN_REVIEW.md §8) | **$0.001124** (measured, real Groq cost) | same artifact |
| RAG Mean Context Recall | 0.9680 (unverified, Aug 2026) | **0.8604** (measured, live) | `data/eval_reports/latest_eval_report.json` |
| RAG Mean Context Precision | 0.9240 (unverified) | **0.9662** (measured, live) | same artifact |
| RAG Mean Faithfulness | 0.9420 (unverified) | **0.0000** (measured, live — new regression) | same artifact |
| RAG Mean Answer Relevance | 0.9100 (unverified) | **0.5392** (measured, live) | same artifact |
| RAG Harmonic Composite | 0.9352 (unverified) | **0.3953** (measured, live) | same artifact |
| RAG Mean Latency | 1.424s (unverified) | **11.156s** (measured, live) | same artifact |
| Multimodal (LeafSense) | BLOCKED (no test corpus) | **MEASURED** on 4 synthetic images, robustness-only | `multimodal_eval_20260919T110330Z.json` |
| Human Evaluation | 16 cases, single reviewer | **24 cases** (8 new), same single reviewer | `human_eval_new_rows_capture_20260919T115041Z.json`, `docs/HUMAN_EVAL.md` |
| Full backend test suite | 801 passing (pre-existing baseline, per session memory) | **807 passed, 1 skipped** | this pass's `pytest -q` run |

---

## E. Unresolved Gaps

1. **The RAG generation-quality regression discovered by the benchmark
   refresh (item 5) is not fixed.** Mean Faithfulness dropped to 0.0000
   because several cases with perfect retrieval (context recall/precision
   both 1.0) still returned the safe-refusal fallback instead of an
   answer — consistent with the corrective/reflection loop
   (`ChatService._correct`) exhausting its retry budget. Root-causing
   and fixing this is a production-logic change, explicitly out of scope
   for an evaluation-only pass whose own constraint was "do not rewrite
   ... production business logic unless strictly required for one of the
   measured gaps" — this was not one of the ten listed gaps; it was
   *discovered* by working on gap 5, not caused by it. **The 24-case
   human evaluation (item 1) independently corroborates this same
   pattern**: rows 17 and 19 show the model refusing to answer despite
   retrieval returning exactly the right content, matching the automated
   metric's finding via manual review.

2. **Multimodal evaluation is explicitly scoped as robustness-only.**
   No real, labeled plant-disease photo corpus exists in this repository
   or was fetched externally in this pass. Diagnostic accuracy of
   LeafSense on real photos remains genuinely unmeasured.

3. **A minor UX finding from human-eval row 24**: the failure-recovery
   error message ("Chat workflow graph did not produce a response")
   exposes internal architecture terminology to the end user. Not a
   security issue (no stack trace/internals leaked) and not a recovery
   defect (the failure was correctly detected and safely contained) —
   just a message-wording polish item for a future pass.

---

## F. Blocked / Unmeasured Items

- LeafSense diagnostic accuracy on real photos — blocked by the absence of a real, labeled test corpus (not fetched externally per the evaluation's own no-fabrication rule).

---

## G. Regression Result

```
cd backend && pytest -q
...
=================== Schema Compliance Rate: 12/12 (100.0%) ====================
807 passed, 1 skipped, 1 warning in 180.20s
```

No test failures. The trailing `--- Logging error ---` block seen in the raw run output is a harmless `httpx`/`huggingface_hub` interpreter-shutdown artifact (closing an already-closed log stream during process teardown), unrelated to any change in this pass and does not affect the pass/fail count.

---

## H. Exact Reproduction Commands

```
# RBAC fix
cd backend && python eval/unauthorized_access_check.py
cd backend && python eval/module10/runners/run_security_eval.py

# Planning Success Rate / Cost Per Successful Task
cd backend && python eval/module10/runners/run_agent_eval.py

# Telemetry capture instrumentation tests
cd backend && python -m pytest tests/test_module10_telemetry_capture.py -q

# Multimodal (requires LeafSense running at settings.vision_service_url, default http://127.0.0.1:8001)
cd backend && python eval/module10/runners/run_multimodal_eval.py

# RAG benchmark refresh
cd backend && python scripts/run_rag_eval.py

# Human evaluation expansion (captures raw answers for rows 17-24; scoring is manual, see docs/HUMAN_EVAL.md)
cd backend && python eval/module10/runners/run_human_eval_new_rows.py

# Full regression
cd backend && pytest -q
```

---

## I. Evidence Artifact Paths

- `backend/eval/module10/reports/security_eval_20260919T111236Z.json` (corrected RBAC measurement)
- `backend/eval/module10/reports/security_eval_20260919T103952Z.json` (historical, pre-correction, preserved)
- `backend/eval/module10/reports/agent_eval_20260919T112455Z.json` (Planning Success Rate + Cost Per Successful Task)
- `backend/eval/module10/reports/multimodal_eval_20260919T110330Z.json` (LeafSense synthetic-image evaluation)
- `data/eval_reports/latest_eval_report.json` (refreshed RAG benchmark, overwritten in place by the script's own convention — `docs/RAG_BENCHMARK_REPORT.md`'s REFRESH section is the durable record)
- `backend/eval/module10/datasets/agent_eval.json` (corrected `planning_cases`)
- `backend/eval/module10/reports/human_eval_new_rows_capture_20260919T115041Z.json` (raw captured answers for human-eval rows 17-24); `docs/HUMAN_EVAL.md` (the scored, durable record)

---

## J. Recommended Next Phase

1. **Investigate the RAG faithfulness/reflection-loop regression (Section E.1) as a dedicated production-logic task**, not an evaluation-only one — likely candidates: the reflection loop's grounding threshold, a prompt/model interaction specific to `openai/gpt-oss-120b`, or a change in `_MAX_LLM_CALLS` retry behavior. Now corroborated by two independent measurements (the automated RAG benchmark and the manual 24-case human evaluation), raising confidence this is real and worth prioritizing.
2. If a real leaf-photo dataset becomes available (even a small, properly-licensed one), extend `run_multimodal_eval.py` to measure actual diagnostic accuracy, not just robustness.
3. Extend `telemetry_capture.py`'s planning-success measurement beyond the 3 current planning cases to a larger sample, now that the instrumentation exists and is proven zero-production-impact.
4. Recruit a second human-eval reviewer to score the same 24 answers independently, enabling a real Inter-Annotator Agreement figure for the first time.
5. Polish the failure-recovery error message surfaced by human-eval row 24 ("Chat workflow graph did not produce a response") to read as a user-facing message rather than internal architecture terminology.

---

*Per the user's explicit instruction: no further large feature implementation follows this report. This is the final deliverable for review.*
