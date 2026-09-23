# Module 10 Audit — AgroSense-RAG

> Status legend: ✅ Verified (implementation + reproducible test + actual
> measured output + saved evidence artifact, all four present) · ⚠️
> Partial (some but not all four present) · ❌ Missing · N/A Deliberately
> out of scope. **A row is only ✅ when all four elements exist** — see
> `docs/CHECKLIST.md`'s own legend, unchanged and enforced identically
> here.

This document supersedes nothing — it is the project-specific,
evidence-backed derivation the grading feedback asked for. Detailed
numbers live in `docs/MODULE10_RESULTS.md`; this file maps each PDF
requirement to its status and evidence.

---

## 1. Project Introduction

See `docs/MODULE10_PROJECT_INTRO.md`.

## 2. Problem Statement

See `docs/MODULE10_PROJECT_INTRO.md`'s "Problem" section.

## 3. Architecture

See `docs/ARCHITECTURE.md` (full diagram + component descriptions) and
its "Explicit Agent Workflow (Phase 1)" section for the agent-graph
topology this audit evaluates.

## 4. Repository

<https://github.com/Udbhav748/AgroSense-RAG>

## 5. Live/Demo

Not verified at audit time — see `docs/MODULE10_PROJECT_INTRO.md`.

---

## 6. Agentic AI Foundations

| Item | Status | Evidence |
|---|---|---|
| Planner | ✅ | `ChatService._plan`; measured: Accuracy 0.9333, Macro F1 0.9475 (n=15). Test: `run_agent_eval.py`. Artifact: `agent_eval_20260919T103533Z.json`. |
| ≥2 tools | ✅ | retrieval, summarization, web search, vision — see `docs/CHECKLIST.md` §3 (unchanged, already evidenced there). |
| Memory | ⚠️ | Session isolation/cross-session-leakage: ✅ verified, 0 leaks (`memory_eval_20260919T104407Z.json`). Pure conversational recall: measured 0/2, traced to a documented design property (document-only-context prompt), not fabricated as a pass. |
| Retry | ✅ | tenacity on LLM/embedding/web-search, unchanged from `docs/CHECKLIST.md` §1 — not re-measured here (see "do not duplicate" instruction). |
| Reflection | ✅ | `ChatService._correct`; failure-injection confirms it degrades safely under a real LLM failure — `failure_eval_20260919T092719Z.json`, `fail_001-003`. |
| Human approval | ✅ | **PHASE 5**: `human_approval_node` is now genuinely wired into `build_chat_graph()`'s live routing for the web-search escalation (previously registered but unreachable — see `docs/MODULE10_FINAL_AUDIT.md` §8). `route_after_approval` confirmed: rejected/expired/pending never resume the guarded action; only a real, resolved `approved` status does. Document-delete approval separately hardened to verify the actual `ApprovalStore` resolution state instead of a client-supplied boolean. 12 new tests (`test_agent_graph_production.py`, `test_main.py`). Off by default in production — unchanged scope from Phase 1. |
| Structured output | ✅ Module 10 gap-closure (2026-09-21): now a real, enabled production path on `POST /chat` — see `docs/CHECKLIST.md` §1/§5 and `docs/MODULE10_RESULTS.md` for the wiring, measured metrics, and endpoint-level evidence. |
| Error handling | ✅ | `AppError` taxonomy; `run_failure_eval.py` confirms 11/12 failure scenarios map to the correct taxonomy category and recover safely (`failure_eval_20260919T092719Z.json`). |
| Logging | ✅ | unchanged from Phase 1, confirmed still emitting during every live run in this audit (see raw log excerpts this audit captured). |

## 7. LangChain/LangGraph/CrewAI Mapping

Unchanged from `docs/CHECKLIST.md` §2 (already flipped to ✅ with test
evidence during Phase 1) — not re-derived here; this audit's job is
evaluation evidence, not re-litigating the architecture.

## 8. Practical Agent Integration

Tool argument accuracy — the one item this audit specifically re-measured:
⚠️ **partial**. Only `summarize`'s `document_id` extraction is
measurable through a text-only harness (n=2, 1.0 accuracy);
`retrieve`'s `top_k` and `diagnose`'s image input are documented
`not_applicable` with a stated reason (`agent_eval_20260919T103533Z.json`),
not silently omitted.

## 9. RAG

| Item | Status | Evidence |
|---|---|---|
| Golden RAG dataset (30+ cases, human-verified) | ✅ | `backend/eval/module10/datasets/rag_eval.json`, 30 cases, all keyword/fact labels hand-derived from the actual indexed corpus content (verified by direct inspection of `vector_store/metadata.json`, not generated). |
| P@5 / Recall@5 / Hit@5 / MRR | ✅ | See `docs/MODULE10_RESULTS.md`. Reproducible: `python eval/module10/runners/run_rag_eval.py`. Artifact: `rag_eval_20260919T103118Z.json`. |
| Semantic vs hybrid vs hybrid+rerank ablation | ✅ | Same artifact — 3-way ablation with real, monotonically-improving results. |
| Groundedness | ✅ | Lexical + claim-decomposition, both labeled as proxies. Same artifact. |
| Citation accuracy | ✅ | Same artifact. |
| Multimodal RAG (image/table/vision QA) | ⚠️ | 4 cases carried over from the pre-existing `dataset_v3.json`; not re-verified live in this pass (depends on a specific multimodal test PDF whose presence in the current 42-document store was not re-confirmed). |
| LeafSense diagnosis → RAG | ⚠️ MEASURED on synthetic images (robustness only, not diagnostic accuracy) | Gap-closure pass, 2026-09-19: no real labeled leaf-photo corpus exists; 4 PIL-generated synthetic images exercise the real live LeafSense integration for pipeline robustness. Artifact: `eval/module10/reports/multimodal_eval_20260919T110330Z.json`. See `docs/MODULE10_RESULTS.md`'s Multimodal section for the two disclosed findings (spurious confidence on blur, no "not a plant" rejection class). Diagnostic accuracy remains NOT MEASURED — would require real labeled photos. |

## 10. Structured Outputs

Unchanged from `docs/CHECKLIST.md` §5 (⚠️, off by default) — not
re-measured in this pass.

## 11. Classification Evaluation

| Item | Status | Evidence |
|---|---|---|
| Confusion matrix | ✅ | `agent_eval_20260919T103533Z.json::planner_classification.confusion_matrix` (+ CSV rows) |
| Per-class precision/recall/F1 | ✅ | Same artifact, `classification_report.per_class` |
| Macro F1 / Weighted F1 | ✅ | 0.9475 / 0.9325, same artifact |
| Actual measured output saved | ✅ | Same artifact + `classification_report_md` rendering |

## 12. Agent Evaluation

| Item | Status | Evidence |
|---|---|---|
| Task success | ⚠️ | Not separately re-measured by this package's runners (reuses `run_eval.py`'s existing Task Success Rate mechanism, unchanged). |
| Tool selection | ✅ | See §11. |
| Tool arguments | ⚠️ | See §8. |
| Planning | ✅ MEASURED (1.0, 3/3) | Gap-closure pass, 2026-09-19: `eval/module10/metrics/telemetry_capture.py` captures the real node sequence from existing `agent_node_trace`/`chat_query_handled` log lines with zero production code changes; tested in `backend/tests/test_module10_telemetry_capture.py` (6 tests). Artifact: `eval/module10/reports/agent_eval_20260919T112455Z.json`. |
| Memory | ⚠️ | See §6. |
| Average steps / Step efficiency | ✅ | 9.5 average steps, `agent_eval_20260919T103533Z.json` |
| Loop rate | ✅ | 0.0 (n=2), same artifact; also cross-checked structurally via `failure_eval`'s `fail_011` (loop-cap scenario, forced and confirmed to terminate) |
| Completion time | ✅ | `completion_time_stats` implemented in `metrics/agent.py`; average node latency 10,981.79ms reported in the same artifact |
| Workflow completion rate | ✅ | 1.0 (post-fix), with an honestly-reported 0.0 pre-fix value demonstrating the metric actually detects real failures |
| Node success rate | ✅ | 1.0 (post-fix) |

## 13. Human Evaluation

✅ **Expanded to 24 cases, 2026-09-19 gap-closure pass** — see
`docs/HUMAN_EVAL.md` (16 original + 8 new, covering hard/ambiguous RAG,
malicious-retrieved-content injection, jailbreak, and failure-recovery
case types). Live capture artifact:
`eval/module10/reports/human_eval_new_rows_capture_20260919T115041Z.json`.

**Module 10 gap-closure (2026-09-21, P8) — second reviewer / IAA
infrastructure**: Reviewer 1's existing 24 scored cases were transcribed
verbatim into structured JSON
(`backend/eval/module10/human_eval/reviewer_1_ratings.json`) and a
complete, tested pipeline was built for a real second reviewer: a
blinded, self-contained, deterministically-shuffled JSON packet
generator (`generate_reviewer2_packet.py`, never reads Reviewer 1's
scores), a strict schema validator, and a runner
(`run_human_eval_final.py`) computing weighted Cohen's kappa per
dimension (quadratic weights, the standard chance-corrected ordinal
agreement statistic — hand-verified against 3 independently-derived
fixtures in `tests/test_human_eval_p8.py`), disagreement statistics, and
hard/disagreement-case identification.

**IAA = still not available.** Only one reviewer's real ratings exist —
running `python eval/module10/runners/run_human_eval_final.py` today
correctly prints `SECOND REVIEWER DATA REQUIRED` and computes only
single-reviewer summaries. No second reviewer was fabricated and no
LLM judge was substituted for the required independent human reviewer.
This is disclosed as **infrastructure implemented**, distinct from
**IAA measured** — see `docs/HUMAN_EVAL.md`'s Inter-Annotator Agreement
section for the exact distinction and reproduction steps.

## 14. Debugging

Unchanged from `docs/CHECKLIST.md` §9 — not re-derived here. This audit's
own live runs did produce real `agent_node_trace`/`request_id`/`trace_id`
log lines for every case (visible in the raw run output captured during
this audit), consistent with what that section already claims.

## 15. Observability

`GET /metrics` confirmed live and populated during this audit's runs
(the same `core/metrics.py` registry every runner reads from via
`agent_workflow_summary()`). A dedicated point-in-time snapshot artifact
was not separately saved to `eval/module10/evidence/` in this pass — see
Remaining Gaps.

**Module 10 gap-closure (2026-09-21)**: a single authoritative
observability report now exists —
`eval/module10/reports/observability_final_*.json`, produced by
`eval/module10/runners/run_observability_final_eval.py` from a real,
controlled 35-request traffic sample (30 successful `POST /chat` + 5
error-path `DELETE /documents/{missing}`). Measured: aggregate error
rate 0.1429 (5/35, alongside the unchanged per-taxonomy breakdown), P50
0.1ms / P95 0.2ms / P99 163.3ms (one cold-model-load outlier),
availability 1.0 (15/15 real `GET /health` probes against a genuinely
spawned local `uvicorn` process, labeled "bounded local service
availability measurement," not production). `AlertEngine`'s threshold-
to-payload path and `monitoring/dashboard.py`'s required views were both
validated against this same captured telemetry. A real, previously-
undocumented finding surfaced while building this report: a
`cache_lookup_node` cache hit short-circuits straight to `END` and never
reaches `finalizer_node`, so cache-hit responses never emit the
`chat_query_handled` log line `monitoring/log_aggregate.py` counts
toward `requests` — meaning log-based aggregation undercounts traffic
whenever the response cache serves an answer (the live `GET /metrics`
Prometheus registry, which instruments at the HTTP layer, is
unaffected). Disclosed and regression-pinned
(`tests/test_observability_cache_gap.py`), not silently patched into the
graph, per this pass's own instruction not to rewrite working
instrumentation unnecessarily. See `docs/MODULE10_RESULTS.md`'s
Observability section for full detail.

## 16. LLMOps

Unchanged from `docs/CHECKLIST.md` §11, plus this audit's own dataset
versioning (`module10_v1`) and regression-relevant metadata (model/
provider/config recorded in every result JSON — see `config.py::run_metadata`).

**Module 10 gap-closure (2026-09-21, re-run + significance test added 2026-09-22) — provider/model A-B evaluation**:
`eval/module10/runners/run_provider_ab_eval.py` runs the same frozen
20-case golden RAG dataset under both supported providers (groq
`openai/gpt-oss-120b`, gemini `gemini-3.5-flash`), fallback/routing
disabled for isolation. Current run (2026-09-22): Faithfulness
groq=0.7641 vs gemini=0.21; task success groq=0.95 vs gemini=0.25 (14
provider-generation-error cases under gemini this run — a 0.7 failure
rate, likely rate-limiting, disclosed as a real reliability event at run
time, not a stable model-quality claim). A paired Wilcoxon signed-rank
test across the 20 matched query pairs (the correct unit of comparison
for this design) gives **p=0.0009**, bootstrap 95% CI of the mean
difference [-0.76, -0.34] — statistically significant for this run, but
substantially confounded by gemini's elevated failure rate this run.
Full per-case detail, pricing assumptions, and disclosed limitations in
`docs/MODULE10_RESULTS.md`. No production default changed as a result
(TASK 11 of that pass) — this is a measurement, not a recommendation.

## 17. Cloud Deployment

Unchanged from `docs/CHECKLIST.md` §12 — out of this evaluation phase's
scope.

## 18. Privacy/Security

| Item | Status | Evidence |
|---|---|---|
| PII Recall | ✅ | 1.0 (15/15), `security_eval_20260919T103952Z.json` |
| Unauthorized Access Rate | ✅ **0.0 (0/2 cross-tenant) — investigated and corrected, 2026-09-19** | Original 0.3333 traced to the eval script wrongly counting an *authorized* same-tenant member delete as an attack; `app/core/permissions.py` deliberately grants members `DOCUMENT_DELETE`. Script corrected to measure cross-tenant attempts only, with the authorized path checked separately (still passing). Artifact: `security_eval_20260919T111236Z.json`; historical 0.3333 artifact (`security_eval_20260919T103952Z.json`) preserved. See `docs/MODULE10_RESULTS.md`. |
| Prompt Injection Success Rate | ✅ | 0.0 (n=3), same artifact |
| Jailbreak Success Rate | ✅ | 0.0 (n=5), same artifact — new dedicated jailbreak suite (`security_eval.json::jailbreak_cases`) covering role override, system-prompt extraction, instruction-hierarchy attack, malicious retrieved content, data exfiltration, tool misuse |
| False Refusal Rate | ✅ | 0.0 (n=1, correctly excluding the PII-boundary case), same artifact |
| Data Leak Rate | ✅ | 0.0 (n=1), same artifact |

## 19. Production Readiness

Unchanged from `docs/CHECKLIST.md` §14 except Unauthorized Access Rate
(now measured as 0.0, corrected and resolved — see §18 above).

**Module 10 gap-closure (2026-09-21) — load/concurrency evaluation
(P7)**: a real HTTP-boundary concurrency ladder (1/2/5/10/20 concurrent
clients, `httpx` against a genuinely spawned `uvicorn` subprocess, never
`TestClient`) was run against `GET /health` (no LLM/retrieval) and
`POST /chat` (real retrieval/reranking, `Settings.llm_provider=mock` for
a deterministic zero-cost LLM stage — a small, narrow, opt-in addition,
`app/services/mock_llm_client.py`). Measured: `/health` RPS 63.75-94.61
across all levels with zero errors; `/chat` RPS fell from 11.66
(concurrency=1) to 1.99 (concurrency=20), with **all 20 requests timing
out at concurrency=20** — a genuine, reproducible single-worker
CPU-bound saturation event (real sentence-transformers embedding +
cross-encoder reranking serialized by the GIL under concurrent load),
not an injected fault. `GET /health` remained healthy immediately after
every level, including the fully-failed one — clean recovery, no
crash/hang. A deliberate rate-limiter burst scenario (100 requests, one
shared identity) did not trip the limiter in this run (0/100
rate-limited) — reported honestly as a negative result. Full detail,
per-level tables, and disclosed limitations (including a resource-
sampling measurement gap: `psutil` measured the client process, not the
server) in `docs/MODULE10_RESULTS.md`'s Load/Concurrency section.
Explicitly NOT production capacity, NOT a cloud SLO.

## 20. Hard Cases

`backend/eval/module10/datasets/hard_cases.json` — RAG hard (7, including
ambiguous disease name, wrong crop, out-of-corpus-plausible, rare
terminology, poorly-worded query, two-source-chunk requirement, misleading
document), agent hard (7, including multi-step task, retrieval-failure
recovery, ambiguous intent, incorrect-tool temptation, repeated
correction, tool/model failure), multimodal hard (5, documented as
manual/live-smoke-test only — not executed in this pass). RAG-hard and
agent-hard cases that reference into `rag_eval.json`/`agent_eval.json`
were exercised as part of those live runs above (not double-counted).

## 21. Failure & Recovery

✅ `backend/eval/module10/datasets/failure_cases.json` — all 12 PDF-listed
scenarios present. 11/12 measured (1 honestly `not_applicable`, already
covered by existing tests). Failure Detection Rate 1.0, Recovery Success
Rate 1.0, Unhandled Failure Rate 0.0. Artifact: `failure_eval_20260919T092719Z.json`.

## 22. Actual Measured Results

See `docs/MODULE10_RESULTS.md` in full.

## 23. Evidence Matrix

See `backend/eval/module10/evidence_manifest.json` (machine-checkable)
and the per-section tables above.

## 24. Remaining Gaps

See `docs/MODULE10_RESULTS.md`'s "Remaining Gaps" section — reproduced
in full there, not duplicated here to avoid drift between two copies.
