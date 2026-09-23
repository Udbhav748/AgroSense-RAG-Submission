# Phase 5 — Final Gap Closure Report

**Date**: 2026-09-19

## A. Baseline

- Starting commit: `4589820` (Phase 4's end state, before this pass's own commits).
- `.env` confirmed never tracked; no secret patterns in tracked files (re-confirmed, same as Phase 4's Step 0).
- Baseline regression: **809 passed, 1 skipped** (unchanged from Phase 4).

## B. Human Approval Implementation

`human_approval_node` was registered in `build_chat_graph()` but had no inbound edge (Phase 4 Finding 1). Fixed by:

1. `retrieval_grader_node` (`app/services/agent_graph/nodes.py`) now sets `approval_required=True`/`approval_type="web_search"` when the grade is weak/insufficient, `Settings.web_search_requires_approval` is on, and the caller hasn't already satisfied the gate (`confirm_web_search=true` or an already-approved reference).
2. `route_after_grader` (`routing.py`) routes such a request to `human_approval` instead of straight to `context_augmentation`.
3. `human_approval`'s `resume` edge now targets `context_augmentation` (the actual guarded action — performing the web search), not `generator`; its `safe_finalizer` edge targets `generator` (not `finalizer` directly), so a pending/rejected/expired request still gets the best answer from whatever was already, legitimately retrieved — web search specifically is guarded, not generation itself.
4. `context_augmentation_node` computes an effective confirm flag (`state.confirm_web_search or state.approval_status == "approved"`) before delegating to `ChatService._augment_weak_retrieval`, so an approved request's web search actually runs.
5. `ChatRequest.approval_id` (new, optional field) threads through `ChatService.handle_query`/`_run_chat_graph` into `AgentState.approval_payload_reference`, letting a client resubmit a query once an operator has resolved its registered approval via `POST /api/v1/approvals/{id}/resolve`.
6. `finalizer_node` surfaces `approval_status`/`approval_id` in `ChatResponse.metadata` when the request was blocked, and skips writing a blocked/incomplete answer to the response cache (a pending-approval placeholder must never be served to a later, approved retry of the same query).

The pre-existing `confirm_web_search=true` client fast path is unchanged and still bypasses the approval queue entirely (backward compatible).

**4 new regression tests** in `tests/test_agent_graph_production.py`: approval-required blocks web search but still generates from existing chunks; a genuinely approved reference allows the web search; a rejected approval blocks it; the `confirm_web_search=true` fast path still skips the queue.

## C. Approval Security Fix (Document Delete)

`document_delete_requires_approval`'s gate accepted a bare client-supplied `approved=true` as proof of approval, with no check against the actual `ApprovalStore` resolution state (Phase 4 Finding 2). This was the project's own tested, intended behavior at the time — changed here as an explicit, documented contract decision, not silently.

**Fix**: `delete_document` (`app/api/v1/routes/documents.py`) now requires an `approval_id` query param. It's resolved against the real store and must satisfy all of: exists, `action == DOCUMENT_DELETE`, `payload.document_id == <this document>`, `status == STATUS_APPROVED` (the store's own `get()` already lazily converts an aged-out pending approval to `STATUS_EXPIRED`, so expiry is covered by the same check). The `approved: bool` param is removed.

**7 regression tests replacing the old, now-inaccurate one**, all in `tests/test_main.py::TestDocumentDeleteApprovalGate`:
1. No approval → denied.
2. Bare `approved=true`, no `approval_id` → denied (the exact behavior being fixed).
3. Pending approval → denied.
4. Rejected approval → denied.
5. Expired approval → denied.
6. Mismatched document_id (approved, but for a different document) → denied.
7. Genuinely approved, matching approval → allowed.

Plus the pre-existing off-by-default case, unchanged.

## D. Faithfulness Post-Fix Measurement

Re-ran the exact two previously-failing human-eval queries (rows 17, 19) live against the fresh Groq key, using `eval/module10/runners/run_faithfulness_post_phase3.py`. Both cases, which returned `FALLBACK_REPLY` before the Phase 3 fix despite retrieval confirmed correct, now produce real, correctly-cited answers:

- **Row 17** ("What's the treatment for scab?"): full apple-scab treatment (cultural controls + organic/conventional fungicides with dosages), citing `[1][3][4]`. Retrieval: 5 chunks, `good` confidence. Latency: 127.1s (first call, cold-start-adjacent).
- **Row 19** (citrus greening/HLB treatment): full HLB vector-control and nutrition program, citing `[3][4][5]`. Retrieval: 5 chunks, `good` confidence. Latency: 15.6s.

**Scope, stated honestly**: only these 2 cases were re-run live, not the full 20-case RAG benchmark or 24-case human evaluation — a deliberate choice to conserve the Groq daily token quota (fully exhausted once already during Phase 3). This result is not extrapolated as a new full-dataset Faithfulness score; it is direct, targeted confirmation the fix resolves the exact regression these two cases demonstrated.

Artifact: `eval/module10/reports/faithfulness_post_phase3_20260919T165741Z.json` (carries git commit, dataset version, model/provider, retrieval config, timestamp, and the explicit scope-limitation note).

## E. Rate-Limit Failure Test

`tests/test_groq_client.py::test_rate_limit_surviving_all_retries_raises_bounded_and_classified` — a deterministic, mocked test (no real 429 call): `self._client.chat.completions.create` raises a genuine `groq.RateLimitError` on every call. Verifies: exactly 3 attempts (tenacity's `stop_after_attempt(3)` — bounded, not infinite), the final exception is `LLMAPIError` (correct classification), and no API key/secret appears in the exception message. The generator_node-level consequence (this exception type reaching `GENERATION_ERROR_REPLY`, not a false "not in documents" fallback) is already proven by Phase 3's `test_generator_exception_uses_generation_error_reply_not_fallback` — not duplicated here.

## F. Tests

Net +10 tests over the Phase 4 baseline (809 → 819), matching the exact pytest delta:

| File | Net new tests |
|---|---|
| `tests/test_agent_graph_production.py` | 4 (web-search approval wiring) |
| `tests/test_main.py::TestDocumentDeleteApprovalGate` | 5 (class grew from 3 tests to 8: 2 kept in spirit and renamed, 1 insecure-behavior test replaced, 5 new state-matrix cases added) |
| `tests/test_groq_client.py` | 1 (rate-limit-surviving-retries) |

Full regression: **819 passed, 1 skipped**.

## G. Before/After Results

| | Before (Phase 4 end) | After (Phase 5) |
|---|---|---|
| `human_approval_node` reachability | Registered, no inbound edge — dead code | Wired into the web-search escalation path, genuinely reachable |
| Document-delete approval | `approved=true` alone sufficient | Requires a real, resolved, matching `Approval` record |
| Faithfulness (rows 17/19) | `FALLBACK_REPLY` on both (pre-Phase-3-fix state) | Real, cited, grounded answers on both (post-fix, measured live) |
| Rate-limit-surviving-retries coverage | None | Deterministic mocked test, bounded at 3 attempts |
| Full test suite | 809 passed, 1 skipped | **819 passed, 1 skipped** |

## H. Remaining Limitations

1. Faithfulness has only been re-measured on 2 targeted cases, not the full 20-case RAG benchmark or 24-case human evaluation (quota-conservation decision, stated honestly — see Section D).
2. No fair A/B comparison of `openai/gpt-oss-120b` against an alternative Groq model has been run (carried over from Phase 3/4, still open).
3. Alerting/operational-signal thresholds (Phase 4's Step 7) remain unimplemented.
4. The web-search approval wiring adds one new code path (`retrieval_grader_node`'s approval flagging); it has not been exercised against a live end-to-end HTTP request in this pass (only via the graph-level unit tests in Section B) — an integration-level smoke test through `POST /chat` with the setting enabled would further strengthen confidence, and is recommended as a quick follow-up.

## I. Exact Reproduction Commands

```
cd backend
pytest -q                                                          # 819 passed, 1 skipped
python -m pytest tests/test_agent_graph_production.py -q           # 15/15, incl. 4 new approval-wiring tests
python -m pytest tests/test_main.py -k ApprovalGate -q             # 8/8 document-delete approval tests
python -m pytest tests/test_groq_client.py -q                      # 11/11, incl. the new rate-limit test
python eval/module10/runners/run_faithfulness_post_phase3.py       # live re-run of rows 17/19
```

## J. Evidence Files

- `app/services/agent_graph/{nodes,routing,graph,augmentation_node}.py`, `app/services/rag_service.py`, `app/models/schemas.py` (approval wiring)
- `app/api/v1/routes/documents.py`, `app/services/approval_service.py` (document-delete security fix)
- `tests/test_agent_graph_production.py`, `tests/test_main.py`, `tests/test_groq_client.py` (new tests)
- `eval/module10/runners/run_faithfulness_post_phase3.py`, `eval/module10/reports/faithfulness_post_phase3_20260919T165741Z.json`
- `docs/MODULE10_FINAL_AUDIT.md`, `docs/MODULE10_AUDIT.md`, `docs/MODULE10_RESULTS.md`, `docs/PHASE4_FINAL_PRODUCTION_READINESS_REPORT.md` (updated)

## K. Final Regression Result

```
819 passed, 1 skipped, 1 warning in 255.61s (0:04:15)
```

---

## Final Status Classification

| Gap | Status |
|---|---|
| 1. `human_approval_node` unreachable | ✅ verified — fixed, tested |
| 2. Document-delete approval bypass | ✅ verified — fixed, tested |
| 3. Faithfulness post-fix measurement | ✅ verified, limited scope (2 cases, stated honestly, not extrapolated) |
| 4. Rate-limit failure test | ✅ verified — deterministic test added |
| Full-dataset Faithfulness re-run | ⚠️ partial/unmeasured — not attempted, quota-conservation |
| Model A/B comparison | ⚠️ partial/unmeasured — not attempted |
| Alerting/operational signals | ❌ missing — not attempted, out of this phase's scope |
| Cloud deployment capabilities | N/A — correctly not claimed, single-process deployment |

Not every row is green, by design. Per this phase's own objective: a defensible final submission with real human approval, verified authorization, measured (scope-limited) post-fix Faithfulness, and bounded provider-failure handling — not a fabricated 100%-complete checklist.

---

*Per this phase's explicit instruction: no further architectural or feature phase follows this report.*
