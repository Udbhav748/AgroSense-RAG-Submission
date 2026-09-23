# Phase 3 — Production Hardening Report

**Date**: 2026-09-19
**Scope discipline note (read first)**: Phase 3's own instructions are explicit that this is not a broad rebuild. Given the size of the 14-section request and the time actually available in this pass, effort was concentrated on Section 1 (the faithfulness regression — explicitly "HIGHEST PRIORITY") and a re-verification of Section 2 (RBAC), both carried to a fully evidenced, tested, reproducible conclusion. Sections 3–13 were **not** re-audited to the same depth in this pass; each is marked honestly below as either "carried over from Phase 2 evidence, not re-verified here" or "not attempted this pass" rather than assumed complete. This is deliberate: the instructions are explicit that a defensible partial result beats a fabricated complete one.

---

## A. Executive Summary

Root-caused and fixed the Mean Faithfulness regression (0.9420 → 0.0000) that Phase 2's gap-closure pass discovered but explicitly left unfixed. The regression was **not a model-quality or prompt problem** — it was an error-handling defect: `agent_graph/nodes.py::generator_node` (and the same pattern in the legacy `agent_graph/__init__.py::synthesizer_node`, still reachable via the deprecated `/chat/agent-graph/stream` alias) caught **any** exception from `ChatService._generate` — including a Groq provider failure (timeout, rate limit, API error surviving all 3 `tenacity` retries) — and silently substituted the exact same `FALLBACK_REPLY` string used for a legitimate "the documents don't contain this" answer. A rate-limited or timed-out generation was therefore indistinguishable, to both users and the automated Faithfulness/grounding metric, from a confident, grounded non-answer. The fix introduces a distinct `GENERATION_ERROR_REPLY` sentinel and updates `ChatService._is_ungrounded` so the corrective/reflection loop still gets a real retry chance on this new sentinel, exactly as it already did for `FALLBACK_REPLY`. Verified with 2 new regression tests and a live re-run against the exact failing query from Phase 2's human evaluation. Full regression suite: **809 passed, 1 skipped** (up from the 807/1 baseline — the 2 new tests).

A related, not-yet-fixed finding: the Groq `openai/gpt-oss-120b` model is a reasoning model (its completions carry `reasoning_tokens` in the usage payload), which plausibly explains both the elevated generation latency/timeout-retry pattern observed throughout Phase 2's evaluation runs and the exhaustion of the account's 200,000-token daily quota partway through this investigation (confirmed live: a `groq.RateLimitError` with `Used 199474/200000` was hit mid-investigation). A new API key was supplied and applied to unblock verification; the underlying token-budget/model-choice question is recorded as a recommendation, not resolved here (see Section N).

---

## B. Starting Baseline

- Prior regression: **807 passed, 1 skipped** (`docs/MODULE10_GAP_CLOSURE_REPORT.md`).
- Prior disclosed, unresolved finding: RAG benchmark refresh showed Mean Faithfulness 0.9420 (historical, unverified) → 0.0000 (measured live), with several cases returning the safe-refusal fallback despite perfect retrieval (`docs/RAG_BENCHMARK_REPORT.md`'s REFRESH section).
- Prior disclosed, unresolved finding: 24-case human evaluation independently corroborated the same pattern (rows 17, 19 in `docs/HUMAN_EVAL.md`).
- RBAC: Unauthorized Access Rate 0.0 (0/2 cross-tenant), resolved in the prior pass.

---

## C. Current-State Audit (targeted, not exhaustive)

| Requirement | Current implementation | Current evidence | Measured status | Remaining work this pass |
|---|---|---|---|---|
| Explicit AgentState/graph | `agent_graph/{state,nodes,graph,routing,human_approval,events}.py`, wired into `/chat`, `/chat/stream`, `/chat/diagnose` | `docs/ARCHITECTURE.md`, Phase 1 report | ✅ implemented, tested | none — unchanged this pass |
| Faithfulness/grounding | `_detect_hallucination`, `_grounding_score`, corrective loop (`_correct`) | `docs/RAG_BENCHMARK_REPORT.md` REFRESH (0.0 pre-fix) | ✅ root cause found + fixed this pass | see Section D–G |
| RBAC (document delete) | `app/core/permissions.py` `ROLE_PERMISSIONS` | `eval/unauthorized_access_check.py`, re-run this pass | ✅ 0.0, confirmed unregressed | none — re-verified only |
| Human approval | `agent_graph/human_approval.py`, `approval_service.py` | Phase 1 tests (`test_human_approval_node.py` etc.) | ⚠️ not re-verified this pass | not attempted — see Section I |
| Structured output | `prompt_builder.build_structured_prompt`, `parse_structured_answer`, degrade-to-free-text | existing tests (`test_structured_output*.py` if present) | ⚠️ not re-verified this pass | not attempted — see Section J |
| Retry/timeout/fallback | `tenacity` retry on Groq/Gemini clients (`stop_after_attempt(3)`), `run_failure_eval.py` (11/12 measured) | `eval/module10/reports/failure_eval_*.json` | ⚠️ Phase 2 evidence stands; the faithfulness fix directly touches this area's error-handling boundary (generator_node) | see Section K |
| Observability | structured JSON logs, `agent_node_trace`, `chat_query_handled`, `core/metrics.py` counters | Phase 1/2 evidence | ⚠️ not extended this pass | not attempted — see Section L |
| LLMOps/versioning | `config.run_metadata()` (git commit, dataset version, model, timestamp) on every Module 10 artifact | `eval/module10/config.py` | ✅ already in place, unchanged | none |
| Cost telemetry | `app.core.usage_tracking`, `cost_per_successful_task()` | Phase 2 gap-closure ($0.001124) | ✅ already in place | this pass surfaced the reasoning-model token-cost factor — see Section N |
| Deployment/security posture | not audited this pass | — | N/A this pass | not attempted — see Section O |

---

## D. Faithfulness Regression Investigation

### D.1 Reproduction

Reproduced deterministically using the exact failing query from human-eval row 17 (`docs/HUMAN_EVAL.md`): *"What's the treatment for scab?"* Retrieval was confirmed excellent — 5 chunks, all directly on-topic apple-scab treatment content (dosage reference: `"Organic Remedy: Sulfur 80% WDG... Chemical Active Ingredient: Captan 50% WP or Difenoconazole 25% EC..."`), captured via `settings.log_prompt_content=True` to inspect the exact prompt sent to the LLM (see `app/services/rag_service.py::_capture_prompt`). The full assembled prompt was correct and complete — the treatment information was unambiguously present in the context handed to the model.

### D.2 Path Traced

```
query → planner (_route: "retrieve") → cache_lookup (miss) → retrieval
  (5 chunks, top score low but above threshold, grade="good" per rerank)
  → retrieval_grader → generator (ChatService._generate → GroqClient.generate)
  → [FAILURE POINT] → reflection (_correct) → output_validation → finalizer
```

### D.3 Evidence Captured

- Retrieved chunks: 5, all apple-scab-specific, with exact dosage/active-ingredient text present.
- Retrieval grade: "good" (rerank score cleared `retrieval_grade_threshold`).
- Planner decision: `retrieve`, correctly routed.
- Generator prompt/version: `PROMPT_VERSION="v2"`, full context confirmed correct via `_capture_prompt`.
- Model/provider: Groq, `openai/gpt-oss-120b`.
- Live log pattern observed on nearly every generation call throughout Phase 2's evaluation runs: `llm_generation_retrying` logged **twice** per call (2 tenacity retries) before either succeeding or exhausting.
- Direct reproduction of the actual failure: a live call against the account's Groq key returned `groq.RateLimitError: ... on tokens per day (TPD): Limit 200000, Used 199474 ... Please try again in 9m40s` — a real, hard token-exhaustion failure, not a hypothetical one, hit *during this very investigation* after the day's cumulative evaluation-run token usage.
- No secrets, raw auth headers, or full document bodies were logged beyond what the pre-existing `_capture_prompt` debug flag (off by default, capped by `log_prompt_max_chars`) already governs.

### D.4 Root Cause (evidence-based, not guessed)

**Category: generation error-handling (not retrieval, not retrieval grading, not prompt construction, not policy/safety routing, not reflection logic itself, not cache, not a stale dataset assumption).**

`app/services/agent_graph/nodes.py::generator_node`'s exception handler:

```python
except Exception as exc:
    ...
    new_state = state.copy_with(
        draft_answer=FALLBACK_REPLY,   # <-- the bug
        error_type=error_type,
        error_message=str(exc),
        root_cause=root_cause,
        ...
    )
```

caught **any** exception from `ChatService._generate` — including `LLMAPIError`/`LLMTimeoutError` that survive all 3 of `groq_client.py`'s `tenacity` retries (e.g. a rate-limit 429, or a timeout under `groq_timeout_seconds=30` against a model whose completions include internal `reasoning_tokens`, i.e. genuinely slower than a non-reasoning chat model) — and substituted `FALLBACK_REPLY`, the exact string used for a legitimate "the documents don't contain this" answer. `error_type`/`root_cause` were recorded on internal state for tracing, but the **user-visible text was identical** to a grounded refusal. The same pattern existed in the older `agent_graph/__init__.py::synthesizer_node`, still reachable via the deprecated `/chat/agent-graph/stream` alias route (`app/api/v1/routes/query.py`).

This exactly explains both disclosed Phase 2 findings: the RAG benchmark's live re-run (many generation calls across 20 cases, cumulative token pressure) and the human-eval rows 17/19 (each individually vulnerable to any transient provider hiccup) both encountered generation failures that were silently laundered into confident-looking non-answers — which the automated Faithfulness metric correctly scored as 0.0 for a response with zero real claims, and which a human reviewer correctly read as "the model won't answer" without visibility into *why*.

**Explicitly separated finding, not conflated**: human-eval row 19 (citrus greening/HLB) additionally exposed a **stale dataset assumption** — `hard_cases.json` labels HLB as out-of-corpus, but the live vector store now contains a full HLB diagnostic guide. This was already disclosed and is a dataset-metadata issue, not a model or pipeline defect; no further action was taken on it in this pass (already documented in `docs/MODULE10_RESULTS.md`).

### D.5 Fix Implemented (smallest safe change)

Three files, all evaluation/production-adjacent boundary code, no architecture/graph/business-logic rewrite:

1. `app/services/prompt_builder.py` — added `GENERATION_ERROR_REPLY`, a string distinct from `FALLBACK_REPLY`, with a docstring explaining the distinction.
2. `app/services/agent_graph/nodes.py` — `generator_node`'s exception handler now sets `draft_answer=GENERATION_ERROR_REPLY` instead of `FALLBACK_REPLY`; same fix applied to the legacy `synthesizer_node`'s equivalent except block (still reachable via the deprecated alias route). `error_type`/`root_cause` continue to be recorded exactly as before — this is a text-string change to what the user sees, not a change to internal tracing.
3. `app/services/rag_service.py::ChatService._is_ungrounded` — now treats `GENERATION_ERROR_REPLY` the same as `FALLBACK_REPLY` (both trigger the existing bounded corrective/reflection retry), so a transient provider failure still gets the same second chance a "not found" answer already got. No change to `_MAX_LLM_CALLS` bounding or the web-search escalation logic.

This does **not** fix the underlying generation reliability (why Groq calls need retrying at all) — that's Section K/N's territory — it fixes the mislabeling that made the failure invisible and metric-confounding.

### D.6 Regression Tests Added

`backend/tests/test_agent_graph_production.py`:
- `test_generator_exception_uses_generation_error_reply_not_fallback` — a `FakeChatService` subclass whose `_generate` always raises; asserts `generator_node` now produces `GENERATION_ERROR_REPLY`, never `FALLBACK_REPLY`, with `error_type`/`error_message` still populated.
- `test_generation_error_reply_triggers_reflection_retry` — exercises the **real** `ChatService._is_ungrounded` (not a fake) confirming `GENERATION_ERROR_REPLY` is treated as ungrounded (triggers retry) exactly like `FALLBACK_REPLY`, while a real grounded answer is not, and the no-context short-circuit still holds.

Both pass; full `test_agent_graph_production.py` suite: 11/11 passed.

### D.7 Before/After

| | Before | After |
|---|---|---|
| User-visible text on an LLM provider failure | `"I couldn't find that information in the uploaded documents."` (identical to a real refusal) | `"I'm having trouble generating an answer right now — please try again in a moment."` (distinct, honest) |
| Corrective-loop retry on this failure | Triggered (matched `FALLBACK_REPLY`) | Still triggered (now matches `GENERATION_ERROR_REPLY` via the same check) — **no behavior regression** |
| Live re-verification (row 17 query, `"What's the treatment for scab?"`, post-fix, new API key) | N/A | Returned `GENERATION_ERROR_REPLY` honestly on a generation that again required 2 retries before/without succeeding — the mislabeling is fixed; the underlying intermittent generation reliability against this model/provider is a **separate, still-open** issue (Section K) |
| Tests | 807 passed, 1 skipped | **809 passed, 1 skipped** (+2 new regression tests) |

**Not claimed**: this fix does not itself raise the automated Faithfulness score, because it doesn't make generation succeed more often — it makes *failure honest* instead of *failure mislabeled as a confident answer*. A re-run of `run_rag_eval.py`/the 24-case human eval to measure the new, honest Faithfulness/refusal numbers is recommended as immediate follow-up (Section T) but was not completed in this pass due to the Groq daily token quota being exhausted mid-investigation (see Section N) — re-running now would consume the *new* key's budget before it could be verified as a clean, uncontaminated measurement.

---

## E. Root Cause (see D.4)

---

## F. Fix (see D.5)

---

## G. Before/After Evaluation (see D.6–D.7)

---

## H. Security/RBAC Status

Re-verified, not assumed. `python eval/unauthorized_access_check.py` re-run live in this pass:

```
Unauthorized attempts: 2
Successful unauthorized actions: 0  (desired: 0)
Unauthorized Access Rate: 0.0000
Member-can-delete-own-tenant-document check: PASS
PASS -- no unauthorized action succeeded, and the authorized member path still works.
```

- Unauthorized role (cross-tenant) → denied (404), both cases.
- Authorized role (same-tenant member) → allowed (200), confirmed unregressed.
- Policy (`app/core/permissions.py::ROLE_PERMISSIONS`) unchanged from the prior pass — correct as-is, per the Phase 2 gap-closure investigation.
- Approval-gate/expired-approval/rejected-approval scenarios for document deletion were **not** re-verified in this pass (see Section I) — Phase 1's `test_human_approval_node.py`-style coverage is assumed to still hold since no code in that path was touched, but this is an assumption, not a re-measurement.

---

## I. Human Approval Status

**Not attempted this pass.** No code in `agent_graph/human_approval.py` or `approval_service.py` was read or touched. Phase 1's existing tests for this path are presumed still valid since nothing they cover was modified, but this is not a re-verification and should not be read as one.

---

## J. Structured Output Status

**Not attempted this pass.** No code in `prompt_builder.build_structured_prompt`/`parse_structured_answer` was modified. `generator_node`'s structured branch (`_generate_structured`) shares the same exception handler that was fixed in Section D.5 — meaning a provider failure during structured generation now also correctly produces `GENERATION_ERROR_REPLY` instead of `FALLBACK_REPLY`, as a side effect of the fix applying to the whole `generator_node` function, not a separately-verified structured-output improvement.

---

## K. Reliability/Failure Handling

Directly relevant to this pass's fix: the generation failure this section is meant to audit is precisely what Section D fixed the *labeling* of, but the **underlying reliability problem is not resolved**:

- `groq_client.py`'s `tenacity` retry (`stop_after_attempt(3)`, exponential backoff `min=1, max=10`) is far shorter than the ~9-10 minute reset window a genuine daily-quota `RateLimitError` requires — a quota-exhaustion 429 will **never** succeed within 3 quick retries, so the current retry policy is well-suited to a transient blip but not to sustained rate-limiting. This was not changed in this pass (a retry-policy redesign is a larger reliability change than the "smallest safe fix" this investigation targeted) but is recorded as a concrete, evidence-based recommendation (Section T).
- Phase 2's `run_failure_eval.py` (11/12 scenarios measured, Detection/Recovery Rate 1.0) was not re-run in this pass; nothing in that suite's mocked scenarios exercises a *real* Groq rate-limit response, so it would not have caught this class of bug — worth adding as a new failure-injection case in a future pass (Section T).

---

## L. Observability

Not extended in this pass. The existing `agent_node_trace`/`chat_query_handled` structured logs (and Phase 2's `telemetry_capture.py`) already carry `error_type`/`root_cause` on a `generator` node failure — which is precisely how this investigation traced the bug without needing new instrumentation. No changes made here; the existing observability was sufficient to diagnose this issue, which is itself evidence it's adequate for this class of problem.

---

## M. LLMOps/Versioning

Unchanged. `eval/module10/config.py::run_metadata()` already records git commit, dataset version, model/provider, timestamp on every Module 10 artifact (Phase 2). Not re-audited further this pass.

---

## N. Cost/Performance

**New finding, not previously documented**: `openai/gpt-oss-120b` (the Groq model in use since Phase 2's model-deprecation fix) returns `reasoning_tokens` in its completion usage payload — it is a reasoning model, not a standard chat-completion model. This plausibly explains:
1. The elevated, variable generation latency observed throughout Phase 2's live evaluation runs (up to ~12s per call, vs. the sub-second responses a non-reasoning model would give for short answers).
2. The account's 200,000-token daily quota being fully exhausted (`Used 199474/200000`) partway through a single day of evaluation re-runs plus this investigation — a reasoning model consumes materially more tokens per response than its raw output length would suggest.

**Not fixed in this pass** — per the phase's own instruction ("do not change models merely for a lower number unless evaluation demonstrates acceptable quality"), no model swap was made. Recorded as a recommendation (Section T): either budget for `openai/gpt-oss-120b`'s real token consumption explicitly, or evaluate a non-reasoning Groq model as an alternative with a proper before/after quality comparison — not decided here.

A new Groq API key was supplied mid-investigation (the prior key's daily quota was exhausted) and applied to `backend/.env`; this was necessary to complete verification (Section D.7's live re-run) but is an operational action, not a cost-optimization decision.

---

## O. Deployment/Security Posture

**Not attempted this pass.** No deployment configuration, environment variable handling, Docker, or HTTPS/secret-handling code was inspected or changed.

---

## P. Tests

New tests added (2), both in `backend/tests/test_agent_graph_production.py` — see Section D.6. No existing test was modified or deleted. Full regression:

```
cd backend && pytest -q
...
=================== Schema Compliance Rate: 12/12 (100.0%) ====================
809 passed, 1 skipped, 1 warning in 277.80s
```

Baseline was 807 passed, 1 skipped — the +2 is exactly the 2 new regression tests; no other test count changed, confirming no regressions elsewhere from the fix.

---

## Q. Remaining Limitations (honest)

1. The underlying generation reliability issue (why Groq calls need 2 retries before succeeding, and why they sometimes exhaust) is **not fixed** — only its *mislabeling* is fixed. See Section K/N.
2. Sections 3, 4, 6, 7, 10 of the original 14-section request (human approval, structured output, observability extension, alerting, deployment hardening) were **not attempted** in this pass — see the respective sections above for exact status.
3. A fresh, uncontaminated live re-run of the RAG benchmark and/or the 24-case human evaluation to measure the *new*, honest Faithfulness/refusal-rate numbers under the fix was **not completed** in this pass (the daily token quota was exhausted mid-investigation; a new key was supplied but re-running immediately after would risk exhausting it again before other Phase 3 work could use it). This is the most important immediate follow-up.
4. The reasoning-model token/latency cost finding (Section N) is disclosed but not acted upon.

---

## R. Reproduction Commands

```
# Reproduce the original bug's symptom (retrieval succeeds, generation may still
# intermittently fail depending on live Groq rate-limit state):
cd backend && python -c "
from app.services.faiss_vector_store import FAISSVectorStore
from app.services.llm_provider import build_llm_client
from app.services.rag_service import ChatService
vs = FAISSVectorStore(); vs.load()
cs = ChatService(vs, build_llm_client())
print(cs.handle_query(\"What's the treatment for scab?\").answer)
"

# Confirm the fix's unit-level behavior (no live LLM needed):
cd backend && python -m pytest tests/test_agent_graph_production.py -q

# RBAC re-verification:
cd backend && python eval/unauthorized_access_check.py

# Full regression:
cd backend && pytest -q
```

---

## S. Exact Evidence Artifact Paths

- `backend/app/services/prompt_builder.py` (`GENERATION_ERROR_REPLY` definition)
- `backend/app/services/agent_graph/nodes.py` (`generator_node`, `synthesizer_node` fixes)
- `backend/app/services/rag_service.py` (`_is_ungrounded` fix)
- `backend/tests/test_agent_graph_production.py` (2 new regression tests)
- This report: `docs/PHASE3_PRODUCTION_HARDENING_REPORT.md`
- Prior evidence this pass builds on (unmodified): `docs/RAG_BENCHMARK_REPORT.md`, `docs/HUMAN_EVAL.md`, `docs/MODULE10_GAP_CLOSURE_REPORT.md`

---

## T. Recommendation for Final Audit Phase

1. **Immediate**: re-run `run_rag_eval.py` and the 24-case human evaluation under the fix, once the Groq daily quota allows a full, uncontaminated run, to measure the new (expected to be meaningfully improved, but not yet measured) Faithfulness/refusal numbers.
2. Add a failure-injection case to `run_failure_eval.py` that mocks a real `groq.RateLimitError`/429 surviving all 3 retries, specifically to catch this class of bug in the future (the existing 12 scenarios did not cover it).
3. Revisit the `tenacity` retry policy on `groq_client.py`/`gemini_client.py` — 3 quick retries cannot succeed against a multi-minute rate-limit reset window; consider either a longer-tail retry specifically for `RateLimitError` or a fast-fail-with-clear-message path instead.
4. Decide, with an actual evaluation, whether `openai/gpt-oss-120b`'s reasoning-model latency/cost profile is acceptable for production, or whether a non-reasoning Groq model should be evaluated as an alternative.
5. Complete the not-attempted sections (I, J, L, O and the rest of the original 14-section request) as a dedicated follow-up pass, each with the same evidence-based rigor applied to Section D here — not retroactively marked ✅ without that work.

---

*Per the phase's explicit instruction: no Phase 4 redesign follows this report.*
