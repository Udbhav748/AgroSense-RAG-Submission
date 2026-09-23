# Production Checklist — Status

A living map of the strict production checklist (Agentic AI, LLMOps, Cloud
Deployment, Privacy) against what's actually implemented in this repo.
Statuses are updated as gaps are closed. Evidence is `file:line` where
practical.

**Module 10 audit**: `docs/MODULE10_AUDIT.md` derives a project-specific,
evidence-backed audit from this checklist — every ✅ row there requires
implementation evidence + a reproducible test + an actual measured
result + a saved artifact, all four, per that document's own rule.
Several rows below are marked ✅ on implementation/methodology grounds
alone; `MODULE10_AUDIT.md` is stricter and should be treated as
authoritative wherever the two disagree. Unauthorized Access Rate was
corrected from an unmeasured ✅ to a measured ⚠️ 0.3333 during the initial
Module 10 audit, then investigated and resolved to a measured ✅ 0.0 during
the 2026-09-19 gap-closure pass (the 0.3333 was traced to the eval
script wrongly counting an app-authorized same-tenant member delete as
an attack — see `docs/MODULE10_GAP_CLOSURE_REPORT.md`).

Status legend:

- ✅ Implemented and verified (tests or measured)
- ⚠️ Partial — exists but incomplete against the checklist's ask
- ❌ Missing
- N/A Deliberately out of scope (see `docs/NOT_APPLICABLE.md`)

---

## 1. Agentic AI Foundations

| Item | Status | Evidence / notes |
|---|---|---|
| Has a planner | ✅ | `ChatService._plan`, `backend/app/services/rag_service.py:311-326` (keyword/regex, no LLM). Actions: `conversational`, `summarize`, `retrieve`, `diagnose`. |
| Has at least two tools | ✅ | Retrieval, summarization, web search, vision — see §3. |
| Memory | ✅ | Session store, in-memory + optional Postgres, `backend/app/services/session_store.py:158-183`, `postgres_session_store.py`; last-6-turns into prompt, `rag_service.py:145`; frontend `session_id` in `localStorage`, `frontend/src/hooks/useChat.js:33-40`. |
| Retry | ✅ | tenacity on LLM + embedding, `gemini_client.py:60-66`, `groq_client.py:60-66`, `embedding_service.py:91-97`. Streaming generation is deliberately not retried (`gemini_client.py:118-126`). Web search now retries transient failures too (`web_search_service.py:44-53`). |
| Reflection | ✅ | Corrective loop `_correct`, `rag_service.py:447-523`; `REFLECTION_INSTRUCTION`, `prompt_builder.py:55-59`; capped at `_MAX_LLM_CALLS=3` (`rag_service.py:58`). |
| Human approval | ✅ | The two existing gates (web search, document deletion) are standardized behind one reusable, traced node — `agent_graph/human_approval.py::human_approval_node`, backed by `approval_service.ApprovalStore` (states: `not_required\|pending\|approved\|rejected\|expired`; never auto-approves — `route_after_approval` only resumes on `"approved"`). **Phase 5**: genuinely wired into `build_chat_graph()`'s live routing for the web-search escalation (`retrieval_grader_node`/`route_after_grader`) — previously registered but unreachable. Document-delete approval is separately hardened at the route level to verify a real, resolved `Approval` record rather than a client-supplied boolean. The pre-existing `confirm_web_search=true` fast path still bypasses the queue, unchanged. Tests: `tests/test_human_approval_node.py` (6), `tests/test_agent_graph_metrics.py::test_approval_metrics_recorded_for_required_approved_rejected`, `tests/test_agent_graph_production.py` (4 new wiring tests), `tests/test_main.py::TestDocumentDeleteApprovalGate` (8 tests). |
| Structured output | ✅ | **Module 10 gap-closure (2026-09-21): now a real, enabled production path**, not merely off-by-default infrastructure. `Settings.structured_output_enabled` is `True` by default; `POST /chat` (only — `/chat/stream`/`/chat/diagnose(/stream)` intentionally never request it, since token-streaming and vision-diagnosis text are free-form contracts) honors `ChatRequest.structured_response=true` end to end: `generator_node` → `ChatService._generate_structured` → provider JSON mode (`response_mime_type`/`response_format`, gemini/groq clients) → `parse_structured_answer` → `StructuredAnswer` Pydantic validation. The validated `{answer, sources}` payload is now surfaced in `ChatResponse.metadata["structured_output"]` with `metadata["structured_output_used"]` — previously silently discarded even when the path succeeded. A malformed/schema-invalid provider response degrades to the plain free-text `_generate()` path and is reported as `structured_output_used: false`, never presented as a fabricated success. Measured: `backend/eval/module10/reports/structured_output_production_20260921*.json` (17-case parser dataset, unchanged for before/after comparison) + `tests/test_structured_output_production.py` (real `POST /chat`/`POST /chat/stream` endpoint tests via `TestClient`, not just the parser in isolation). Limitation: only `POST /chat`'s non-streaming path is structured-capable; disclosed, not claimed universal. |
| Error handling | ✅ | `AppError` taxonomy + one global handler, `app/core/exceptions.py`, `error_handlers.py:26-61`. |
| Logging | ✅ | Structured JSON + `request_id` on every line, `app/core/logging.py:17-40`, `main.py:51-68`. |

### Metrics

| Metric | Status | Evidence |
|---|---|---|
| Tool Selection Accuracy | ✅ | Planner confusion matrix + per-class P/R/F1 + Plan-Execution Consistency (decided action vs tool that ran), `run_eval.py`. |
| Task Success Rate | ✅ | Keyword-based, `run_eval.py:448-449,559-561`. |
| Step Efficiency | ✅ | `avg_step_efficiency` = min(expected/actual steps) + avg steps, `run_eval.py` (`EXPECTED_MIN_STEPS`). |
| Tool Success Rate | ✅ | Per-tool successes/attempts, `run_eval.py` `tool_success_rate` (offline) + runtime `tool_invocation` events (`tool_registry.py` `@track_tool`) aggregated by `monitoring/log_aggregate.py`; `--min-tool-success-rate` alert threshold. |
| Loop Rate | ✅ | `agent_loop_limit_hits_total` / total workflows, `core/metrics.py::Metrics.agent_workflow_summary()` — recorded whenever a graph run hits `build_chat_graph(max_steps=16)`'s cap without reaching END (`engine.py`). Also cross-checked offline from `graph_cycle_capped_max_steps` log lines, `metrics_report.py::report_agent_workflow_metrics`. `loop_capped` (the pre-existing, narrower "correction loop hit `_MAX_LLM_CALLS`" signal) still exists separately. |

---

## 2. LangChain, LangGraph & CrewAI

Approach: **a dependency-free graph runtime this repo owns**
(`agent_graph/engine.py`), not a third-party framework — justified in
`docs/ARCHITECTURE.md` "Framework choice". As of Phase 1, the chat
workflow is genuinely expressed as explicit nodes/edges/state (below),
not just conceptually mapped.

| Concept | Status | Mapping |
|---|---|---|
| LangChain components | ✅ | `RecursiveCharacterTextSplitter` used for chunking, `chunking_service.py:24-32`. |
| LangChain chains | ✅ | Pipeline = deterministic chain (validate→extract→chunk→embed→index; plan→retrieve→grade→generate→correct). |
| LangChain agents / tools | ✅ | Tool callables + formal Pydantic tool I/O schemas + `@track_tool` invocation tracking, `app/services/tool_registry.py`. |
| LangChain memory | ✅ | Session store maps to LangChain `ConversationBufferWindowMemory` (bounded last-6-turns). |
| LangGraph nodes | ✅ | 12 named node functions (`agent_graph/nodes.py` + `cache_node.py`/`augmentation_node.py`/`human_approval.py`), each a thin wrapper delegating to an existing `ChatService`/service method — see `docs/ARCHITECTURE.md`'s "Explicit Agent Workflow" section. Tests: `tests/test_agent_graph_production.py`, `tests/test_vision_node.py`, `tests/test_cache_lookup_node.py`, `tests/test_augmentation_node.py`. |
| LangGraph edges / state | ✅ | `agent_graph/state.py::AgentState` — one explicit, typed Pydantic state object (~50 fields) threaded through every node via `copy_with(...)`; no secrets, no raw large payloads (`tests/test_agent_graph_state.py`). Edges wired in `agent_graph/graph.py::build_chat_graph()`. |
| LangGraph workflow | ✅ | `build_chat_graph()` backs `POST /chat` (`ChatService._run_chat_graph`) and shares its cache/retrieval/grading nodes with `POST /chat/stream`. Bounded (`max_steps=16`), traced, explicit start (`validate_request`) and end (`finalizer`/cache-hit `END`). |
| LangGraph conditional routing | ✅ | `agent_graph/routing.py` (`route_after_planner`, `route_after_grader`, `route_after_validation`, `route_after_approval`) plus `route_after_cache_lookup`/`route_after_augmentation` — pure `(AgentState) -> str` functions, reviewable independent of node side effects. |
| CrewAI role | ✅ | `AGENT_ROLE` constant, `prompt_builder.py` (CrewAI-style role/goal/backstory spec) |
| CrewAI goal | ✅ | `AGENT_GOAL` constant, `prompt_builder.py` |
| CrewAI backstory | ✅ | `AGENT_BACKSTORY` constant, `prompt_builder.py` |
| CrewAI task | ✅ | Prompt template + task framing per tool (`_INSTRUCTIONS`, `_WEB_RESULTS_INSTRUCTION`). |
| Agent collaboration | ❌ | Single agent; no collaboration. N/A per `NOT_APPLICABLE.md`. |

### Implementation checklist

| Item | Status |
|---|---|
| Tool abstraction | ⚠️ — tools are plain functions behind `VectorStore`/`LLMClient` interfaces; no shared tool envelope |
| Prompt templates | ✅ — `prompt_builder.py` |
| State management | ✅ — `AgentState` (`agent_graph/state.py`), immutable-update via `copy_with`, request/conversation/retrieved-context/persistent-data lifetimes kept explicit (see `ARCHITECTURE.md`) |
| Retry | ✅ — LLM/embedding only (§1) |
| Conditional routing | ✅ — planner + retrieval grading, now explicit `routing.py` functions (see §2 table above) |
| Human node | ✅ — `human_approval_node`, standardizing the same two gates behind `approval_service.ApprovalStore`; see §1's Human approval row |
| Parallel execution | ✅ — real concurrent `asyncio` branches (`run_concurrent_branches`, `agent_graph/engine.py`) wired into the non-streaming diagnose workflow (vision + weather run concurrently); chat corrective loop remains sequential by design (dependent steps), streaming diagnose not converted; ingestion also embeds in batch (`embedding_service.py:126`) — see `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md` §2 for full disclosure and measured evidence |
| Multi-agent design | ❌ — single agent, N/A |

### Metrics

| Metric | Status |
|---|---|
| Workflow Completion Rate | ✅ — `agent_workflow_completed_total` / (`_completed_total` + `_failed_total`), `Metrics.agent_workflow_summary()`; surfaced in `run_eval.py`'s report and `GET /metrics`. Test: `tests/test_agent_graph_metrics.py::test_workflow_completion_metrics_recorded`. |
| Agent Handoff Accuracy | N/A — single agent (`NOT_APPLICABLE.md`) |
| Node Success Rate | ✅ — `agent_node_executions_total{status="success"}` / total, same summary method — one counter per node execution, emitted by every node via `emit_node_trace`. Test: `tests/test_agent_graph_metrics.py::test_node_execution_metrics_recorded_per_run`. |
| Average Node Latency | ✅ — `agent_node_latency_seconds` histogram (per node + aggregate), same summary method; offline cross-check in `metrics_report.py::report_agent_workflow_metrics`. |
| Agent Idle Time | N/A — single request-scoped agent |

---

## 3. Practical Agent Integration

### Tools / capabilities

| Tool | Status | Evidence |
|---|---|---|
| RAG | ✅ | §4 |
| PDF | ✅ | `document_service.py` (PyMuPDF) |
| OCR | ✅ | pytesseract fallback, `document_service.py` |
| Vision | ✅ | `vision_client.py` → LeafSense HTTP |
| API | ✅ | LeafSense HTTP client + FastAPI routes |
| Calculator / Weather / SQL / Email / GitHub / Speech | N/A | Out of scope, `NOT_APPLICABLE.md` |

### Tool checklist

| Item | Status | Evidence / notes |
|---|---|---|
| Every tool documented | ✅ | README + `docs/ARCHITECTURE.md` components |
| Input schema | ✅ | Formal per-tool Pydantic input schemas + runtime validation, `app/services/tool_registry.py` (`RetrievalInput`, `SummarizationInput`, `WebSearchInput`, `DiagnoseInput`); invalid args raise `ToolInputError` (422). |
| Output schema | ✅ | Per-tool output schemas declared in `tool_registry.py` (`RetrievalOutput`, `SummarizationOutput`, `WebSearchOutput`, `DiagnoseOutput`) alongside the typed domain models (`RetrievedChunk`/`WebSearchResult`/`VisionPrediction`). |
| Retry | ✅ | LLM/embedding tenacity + web-search now retries transient failures (`web_search_service.py:44-53`); vision/retrieval fail fast by design (documented). |
| Timeout | ⚠️ | Web 10s (`config.py:151`), vision 15s (`config.py:209`), LLM 30s; retrieval has none |
| Authentication | ⚠️ | Inbound API keys ✅; outbound vision call to LeafSense is **unauthenticated** (`vision_client.py:91-95`) |
| Cost tracking | ✅ | Per-LLM-call cost (`gemini_client.py`/`groq_client.py`) + **per-request rollup** in `chat_query_handled` via `app/core/usage_tracking.py` (`llm_calls`, `total_tokens`, `estimated_cost_usd`). |
| Latency measurement | ✅ | `processing_duration` logged per tool + `tool_invocation` events with `latency_ms` |
| Security / input validation | ✅ | Request schemas + validation_service for uploads |
| Input validation | ✅ | `validation_service.py`, Pydantic request models + tool-registry schemas |

### Metrics

| Metric | Status |
|---|---|
| API Success Rate | ⚠️ | Implicit in error taxonomy; per-tool success rate now logged via `tool_invocation` events |
| Retry Success Rate | ✅ | `report_retry_success_rate`, `metrics_report.py:104-156` — correlates `llm_generation_retrying` events to `llm_generation_completed` by `request_id` (with a time-window fallback), reports successes/retried-requests/rate. |
| Timeout Rate | ⚠️ | Timeouts logged; no rate aggregated |
| Argument Accuracy | ⚠️ | Summarize `document_id` only (`run_eval.py:410-420`) |

---

## 4. Retrieval-Augmented Generation — RAG

### Architecture

Question → Embedding (`embedding_service.py`) → Vector search (`faiss_vector_store.py` +
`hybrid_search.py`) → Top-K (`retrieval_service.py`) → Prompt with evidence
(`prompt_builder.py`) → LLM answer (`gemini_client.py`/`groq_client.py`).

### Checklist

| Item | Status | Evidence |
|---|---|---|
| Chunking | ✅ | `RecursiveCharacterTextSplitter`, 1000/200, `chunking_service.py:24-32` |
| Metadata | ✅ | `{document_id, chunk_index, total_chunks, source}` per chunk, `chunking_service.py:34-47` |
| Embedding | ✅ | `all-MiniLM-L6-v2`, L2-normalized, `embedding_service.py` |
| Vector database | ✅ | FAISS `IndexFlatIP` + positional `metadata.json`, `faiss_vector_store.py` |
| Citation | ✅ | Untrusted-excerpt markers + inline sources instruction, `prompt_builder.py` |
| Source display | ✅ | `ChatResponse.sources`, `schemas.py:45-52,67-68`; `SourceReferences.jsx` |
| Hybrid search | ✅ | BM25 + FAISS fusion 0.6/0.4, `hybrid_search.py` |
| Re-ranking | ✅ | Opt-in cross-encoder, `reranking_service.py` |
| Image extraction | ✅ | Embedded figures + low-text page rasters via PyMuPDF, `document_service.py` `extract_images_from_pdf` (gated: `image_extraction_enabled`) |
| Image captioning | ✅ | Gemini vision captions indexed as `source="image_caption"` chunks, `image_captioning_service.py` (gated: `image_captioning_enabled`) |
| Table extraction | ✅ | PyMuPDF `find_tables` → structured markdown chunks, `table_extraction_service.py` (gated: `table_extraction_enabled`) |
| Vision-grounded QA | ✅ | Weak-retrieval vision fallback over low-text page rasters, `vision_qa_service.py` + `rag_service.py` (gated: `vision_qa_enabled`) |

### Metrics

| Metric | Status | Evidence |
|---|---|---|
| Precision@K | ✅ | `precision_at_k`, `run_eval.py:151-177` (k=5) |
| Recall@K | ✅ | `recall_at_k`, same |
| Hit Rate@K | ✅ | `hit_at_k` (binary "any relevant in top-k"), `run_eval.py` |
| Mean Reciprocal Rank | ✅ | `reciprocal_rank`, same |
| Groundedness | ✅ | Lexical `is_grounded` (`run_eval.py:118-131`) + LLM-as-judge entailment (`235-265`) |
| Citation Accuracy | ✅ | `citation_supported`, `run_eval.py:180-191` |
| Faithfulness (20-case golden benchmark, `scripts/run_rag_eval.py`) | ⚠️ | **Measured 0.7809** (raw, 2026-09-22), below the 0.80 target but close. Root-caused and fixed across two passes: (2026-09-20) 3 of 4 previously-zero-scored cases fixed (2 via wiring the already-existing `FallbackLLMClient` provider-failover, 1 via raising `retrieval_top_k` 5→8); (2026-09-22) a real bug in the eval script itself (`retrieve()` called with a nonexistent `rerank_candidates` kwarg, silently degrading every retrieval to a raw fallback) fixed `eval-potato-02` (0.0 → 0.6) and raised the full-dataset mean 0.7093 → 0.7809. `eval-potato-01` (0.4) and a newly-visible `eval-apple-01` (0.0) remain weak under the now-correctly-exercised retrieval path — same disclosed cross-encoder/dosage-table-chunk ranking limitation, not fixed. Full before/after: `backend/eval/module10/reports/faithfulness_final_*.json`, `backend/eval/module10/reports/rag_eval_retrieve_signature_fix_*.json`, `docs/MODULE10_RESULTS.md`. |

---

## 5. Structured Outputs

| Item | Status | Evidence |
|---|---|---|
| JSON output | ✅ | `generate_structured()` uses `response_mime_type="application/json"` (gemini_client.py) / `response_format={"type":"json_object"}` (groq_client.py); `StructuredAnswer` validated by `structured_output.py`. **Now enabled by default** (`structured_output_enabled=True`) on `POST /chat` when the caller sends `structured_response=true`; still off for `/chat/stream` and `/chat/diagnose(/stream)` by design (free-form contracts). |
| Validation | ✅ | Pydantic request/response models, `schemas.py` |
| Pydantic model | ✅ | `schemas.py`, `models/document.py`, `models/db_models.py` |
| Required fields | ✅ | Pydantic required fields + FastAPI 422s |
| Error messages | ✅ | `AppError.detail` + handler mapping |
| Schema Compliance Rate | ✅ | Measured on the ChatResponse wire contract (`run_eval.py` `check_response_schema`) and on the LLM's own JSON-mode output: `eval/module10/runners/run_structured_output_eval.py`'s 17-case dataset — see "Structured output" row in §4 above for the current numbers and their honest methodology (compliance is measured over a mixed valid/intentionally-malformed fixture set, not misread as a defect rate). |
| Field Accuracy | ✅ | Fraction of ChatResponse field checks passing across entries, `run_eval.py`; LLM-output field accuracy via `StructuredAnswer` validation, `run_structured_output_eval.py` (1.0 on all cases that parse). |

---

## 6. Classification Evaluation

| Item | Status | Evidence |
|---|---|---|
| Confusion matrix | ✅ | Planner, `run_eval.py:294-298` |
| Accuracy | ✅ | `run_eval.py:301-330` |
| Precision | ✅ | Same |
| Recall | ✅ | Same |
| Specificity | ✅ | Per-class specificity in `classification_report`, `run_eval.py` |
| F1 Score | ✅ | Macro + weighted, same |
| Macro average | ✅ | Same |
| Weighted average | ✅ | Same |
| TP/FP/TN/FN surfaced | ⚠️ | Computed internally for the matrix, not reported per-class |

---

## 7. Agent Evaluation

| Item | Status | Evidence |
|---|---|---|
| Tool selection verified | ✅ | `expected_action` vs `plan.action`, `run_eval.py:406-420` |
| Tool arguments verified | ⚠️ | Only summarize `document_id`; no generic arg check |
| Planning (action sequence) verified | ✅ | `plan_execution_consistent` (decided action vs executed tool), `run_eval.py` |
| Memory verified | ✅ | `memory_recall_rate` + dataset `history`/`expected_memory_keywords` entries, `run_eval.py` |
| Hallucination detected | ⚠️ | Groundedness proxies; no explicit hallucination detector |
| Grounding verified | ✅ | Lexical + LLM-judge groundedness |
| Task success verified | ✅ | Keyword match, `run_eval.py:448-449` |
| Human approval enforced | ⚠️ | `confirm_web_search` gate on the web-search tool (`web_search_requires_approval`) and `approved` gate on document deletion (`document_delete_requires_approval`); both off by default |
| Task Success Rate | ✅ | |
| Tool Selection Accuracy | ✅ | |
| Average Steps | ✅ | `metrics_report.py:116-157` + `avg_steps_taken`, `run_eval.py` |
| Loop Count | ✅ | `loop_capped` reporting |
| Completion Time | ⚠️ | `processing_duration` logged; no dedicated completion-time metric |

---

## 8. Human Evaluation

| Item | Status | Evidence |
|---|---|---|
| Correctness / Helpfulness / Completeness / Safety / Tone / Groundedness / Citation Quality | ✅ | 7-dimension 1–5 rubric, `docs/HUMAN_EVAL.md:15-39` |
| 1–5 rating scale | ✅ | Anchored per score |
| Likert scale | ✅ | 1–5 Likert-style |
| Inter-Annotator Agreement | ⚠️ | **Module 10 gap-closure (2026-09-21, P8)**: full two-reviewer/IAA infrastructure implemented and tested — `backend/eval/module10/human_eval/` (reviewer-1 ratings transcribed to structured JSON, blinded reviewer-2 packet generator, schema validation) + `backend/eval/module10/runners/run_human_eval_final.py` (weighted Cohen's kappa per dimension, disagreement stats, hard-case identification). Still ⚠️ not ✅: **only one reviewer's real ratings exist** — no second reviewer has been fabricated or substituted with an LLM judge, so IAA is not yet a measured figure. Running the command today prints `SECOND REVIEWER DATA REQUIRED`. See `docs/HUMAN_EVAL.md`'s Inter-Annotator Agreement section and `docs/MODULE10_RESULTS.md`'s Human Evaluation section. |

---

## 9. Debugging

| Item | Status | Evidence |
|---|---|---|
| Trace (every step of one request) | ✅ | SSE `trace` events (`rag_service.py:720-887`) + `trace_event_emitted` logs (`235-243`) |
| Prompt recorded (exact + version) | ⚠️ | `prompt_version` logged (`rag_service.py:339-349`); prompt **content not captured** |
| Tool logs (names, args, outputs, failures) | ✅ | `tool_invocation` events carry tool name, input summary, success/failure + error type, and output summary (`tool_registry.py:133-179`); inputs/outputs are shape-bounded summaries, not full content. |
| Token logs | ✅ | `llm_generation_completed` fields, `gemini_client.py:93-114` |
| Error logs | ✅ | `request_failed`/`unhandled_exception`, `error_handlers.py` |
| Stack trace | ✅ | `exc_info` on unhandled, `error_handlers.py:50-53` |
| Root cause identified | ⚠️ | `taxonomy_category` narrows it; no explicit root-cause attribution |

### Error taxonomy

| Category | Status |
|---|---|
| input / tool / retriever / prompt / reasoning / output / deployment | ✅ Exceptions exist |
| intent / planner / memory | ⚠️ Declared in vocabulary (`exceptions.py:3-11`) but no exceptions use them |

---

## 10. Observability

| Item | Status | Evidence |
|---|---|---|
| Tracing | ✅ | SSE trace + request_id |
| Logging | ✅ | Structured JSON stdout |
| Metrics | ✅ | Live `GET /metrics` (Prometheus text exposition, in-process registry at `app/core/metrics.py`, emitted by `app/api/v1/routes/metrics.py`) — request latency histogram, `http_requests_total` by method/path/status, per-tool invocations/latency, LLM call/token/cost, loop-capped, retrieval timeouts, errors by taxonomy; optional `METRICS_BEARER_TOKEN`. Offline `metrics_report.py` + `monitoring/log_aggregate.py` remain for post-hoc/long-window rollups. |
| Alerts | ⚠️ | Threshold breaches via `log_aggregate.py`/`uptime_check.py` exit codes, with a best-effort Slack-compatible webhook push available (`monitoring/alert_webhook.py`, `--alert-webhook-url`/`ALERT_WEBHOOK_URL`, off by default). **Module 10 gap-closure (2026-09-21)**: a separate, real, tested, debounced threshold engine also exists — `app/core/alerting.py::AlertEngine` (rule registration, GREATER_THAN/LESS_THAN comparators, debounce, recovery events, pluggable notification sink) — validated end to end (metric input → threshold evaluation → alert triggered → payload produced, against both synthetic breach values and this pass's own real measured `error_rate`) via `tests/test_alerting.py` + `tests/test_alert_engine_integration.py`, with a fake webhook sink (no real Slack account needed) and tests proving no secret ever appears in an alert payload. Still **not continuously invoked** against a live target — no scheduled job calls `AlertEngine.evaluate()` periodically, and it is a separate code path from `log_aggregate.py`'s own simpler `_breaches()`/`send_alert()` mechanism (both exist; not unified). Run manually or wire a schedule against a real deployment target for continuous automated alerting. |
| Dashboards | ✅ | Text dashboard `monitoring/dashboard.py` (availability, latency, loop/tool rates, retry activity, requests/min, per-endpoint breakdown, tokens/cost; `--json` for machine consumers). A dependency-free stand-in for a hosted Grafana-style stack, sharing `log_aggregate.aggregate` so rollups can't drift. **Module 10 gap-closure (2026-09-21)**: validated against real captured telemetry from a controlled traffic run (not a hypothetical log file) — all 8 required views present (availability, latency, error rate, tool success, retry activity, requests, token usage, cost); `tests/test_dashboard.py` (7 tests, previously untested). |
| Prompt logs | ⚠️ | Version always recorded (`generation_requested`); exact content only via `Settings.log_prompt_content` (off by default) — a deliberate, documented debug-only capture, not enabled globally. **Module 10 gap-closure (2026-09-21)**: boundary explicitly tested — `tests/test_prompt_capture_boundary.py` confirms the off-by-default behavior, the truncated debug capture, and that `prompt_version` is unconditionally recorded regardless of the flag. Still intentionally off by default; not turned into normal production prompt logging. |
| Tool logs | ✅ | Names, latency, and input/output summaries — success `tool_invocation` events now carry a bounded `output` shape (`_summarize_output`, `tool_registry.py:201-225`), so what each tool produced is visible at a glance. |
| Token usage | ✅ | Per generation |
| Latency | ✅ | P50/P95/P99 in `metrics_report.py:67-94`; live scrape latency via `http_request_duration_seconds` histogram on `GET /metrics` (quantile via `histogram_quantile`); `monitoring/uptime_check.py` can also probe latency manually against a running deployment |
| Errors | ✅ | By taxonomy category |
| Cost | ✅ | `estimated_cost_usd` per generation; summed in `metrics_report.py:160-180` |
| User feedback | ✅ | Thumbs + comment endpoint |
| P50/P95/P99 latency | ✅ | Measured from a real, controlled 35-request sample (2026-09-21): P50=0.1ms P95=0.2ms P99=163.3ms (one cold-model-load outlier; see `docs/MODULE10_RESULTS.md`'s Observability section). `eval/module10/reports/observability_final_*.json`. |
| Error Rate | ✅ | `report_error_rate_by_category` (per-taxonomy breakdown) **plus** the explicit single aggregate `error_rate = failed_requests / total_requests` (`monitoring/log_aggregate.py::aggregate`, already computed there; formalized and pinned by `tests/test_log_aggregate.py`'s new deterministic tests) — both reported together, neither replaces the other. Measured on a real controlled traffic sample (2026-09-21): 0.1429 (5/35). |
| Availability | ⚠️ | Probe script `monitoring/uptime_check.py` exists but its scheduled workflow (`monitoring.yml`, 15-min cron) was removed along with the AWS deployment path it monitored — run manually against a real deployment target; point-in-time only, no durable SLO. **Module 10 gap-closure (2026-09-21)**: a reproducible **bounded local service availability measurement** now exists — `eval/module10/runners/run_availability_eval.py` starts a real `uvicorn` subprocess and probes `GET /health` repeatedly over a bounded window, computing `successful probes / total probes` (measured: 15/15 = 1.0 over a 15s local window). Explicitly labeled "bounded local service availability measurement," not production availability — still no continuous/durable SLO. |

---

## 11. LLMOps

| Item | Status | Evidence |
|---|---|---|
| Prompt version | ✅ | `PROMPT_VERSION = "v1"`, `prompt_builder.py:15` |
| Dataset version | ✅ | `dataset_vN.json` files + `dataset_version` recorded per run, `run_eval.py` |
| Model version | ✅ | `llm_model_name` + `reranking_model_name` + `embedding_model_name` recorded per run, `run_eval.py` |
| Evaluation pipeline | ✅ | Harness + manual `eval.yml`; now includes regression gate |
| A/B testing | ✅ | Manual offline before/after runs, `OPERATIONS.md:8-54`, **plus** a controlled provider/model A-B evaluation (Module 10 gap-closure, 2026-09-21): `eval/module10/runners/run_provider_ab_eval.py` runs the same frozen 20-case golden dataset under groq (`openai/gpt-oss-120b`) and gemini (`gemini-3.5-flash`) with fallback/routing disabled to isolate each provider's own reliability, measuring Faithfulness/task success/tool selection/latency/cost neutrally without declaring a winner. See `docs/MODULE10_RESULTS.md`'s "Provider/Model A-B Evaluation" section. |
| Rollback | ✅ | Documented procedure, exercised on `v0.1.0`, `OPERATIONS.md` |
| Monitoring | ⚠️ | Stand-in `metrics_report.py` + `monitoring/log_aggregate.py` (windowed thresholds, now with optional webhook push — `alert_webhook.py`); no hosted dashboard/live-metrics stack |
| Regression Rate | ✅ | `regression_check.py` gate wired into `eval.yml` (compares vs `baselines/v2_groq.json`) |
| Acceptance Rate | ✅ | Thumbs-up ratio, `metrics_report.py:183-208` |
| Failure Rate | ⚠️ | Error rate by category, not a single failure-rate metric |
| Deployment Frequency | ⚠️ | Auto-deploy on push; no metric |

---

## 12. Cloud Deployment

| Item | Status | Evidence |
|---|---|---|
| FastAPI | ✅ | `app/main.py` |
| Docker | ✅ | `backend/Dockerfile` |
| AWS / EC2 / Lambda / Bedrock / SageMaker / Vertex AI / Azure AI / GPU | ⚠️ | Self-hosted via Docker Compose on a plain EC2 instance — no managed AWS compute service (Lambda/ECS/etc.); see `docs/OPERATIONS.md` "Deploying to EC2" |
| HTTPS | ⚠️ | TLS via the Caddy overlay (`docker-compose.caddy.yml`) is now the **documented default** deploy path in `docs/OPERATIONS.md` (domain/DNS as an explicit step-0 prerequisite, security-group table and first-time-deploy steps built around it) — still not enforceable by the code itself (needs a real domain, which the app can't require), and the no-domain fallback still exists and works, so still not "on by default" in the strict sense. Status held at ⚠️ rather than raised to ✅ for that reason. |
| Secrets | ⚠️ | AWS SSM Parameter Store (`_load_secrets_from_ssm()`, `SECRETS_SSM_PREFIX`) is now the **documented recommended path** in `docs/OPERATIONS.md`'s "Secrets" section, with concrete `aws ssm put-parameter` steps and IAM guidance — plain `.env` is now explicitly the fallback, not the only path shown. Still ⚠️ rather than ✅: nothing in the code enforces this, so a deploy can still skip it and land on plaintext `.env`. |
| Load balancer | N/A | Single EC2 instance, no load balancer in front |
| Autoscaling | N/A | Single EC2 instance, `docker-compose.yml` never scales the backend service beyond one replica |
| Monitoring | ✅ | Live `GET /metrics` endpoint in the app itself (Prometheus text format — `app/core/metrics.py`, `app/api/v1/routes/metrics.py`), for any Prometheus/Grafana Cloud agent to scrape; plus the offline stand-ins `metrics_report.py` + `monitoring/*.py` scripts (uptime probe + log rollup, both with optional webhook push on breach). No hosted dashboard stack of our own — but a live, scrapeable endpoint is now on the wire, which is what "no live metrics" was flagging. |
| Centralized logging | ⚠️ | Stdout JSON only; `log_aggregate.py` remains the offline rollup tool — no managed log service wired up |
| Requests per second | ⚠️ | **Module 10 gap-closure (2026-09-21, P7)**: measured for real, over a real HTTP/uvicorn boundary — `eval/module10/runners/run_load_concurrency_final_eval.py`, concurrency ladder 1/2/5/10/20 against `GET /health` (63.75-94.61 RPS, stable) and `POST /chat` (11.66 RPS at concurrency=1, falling to 1.99 RPS at concurrency=20, with a full timeout saturation at concurrency=20 — see `docs/MODULE10_RESULTS.md`'s Load/Concurrency section). Still ⚠️, not ✅: this is a single local machine's measured throughput under bounded local concurrency, explicitly not cloud-scale/production RPS capacity. |
| Latency | ✅ | P50/P95/P99 offline |
| Availability | ⚠️ | `monitoring/uptime_check.py` probes exist but the scheduled workflow that ran them was removed with the AWS deployment path — point-in-time, run manually |
| Cost per hour | ⚠️ | Not measured live; static EC2 estimate in `docs/OPERATIONS.md`'s "Deploying to EC2" §Cost (~$15-20/month) |
| CPU/GPU/Memory utilisation | ⚠️ | Not dashboarded. **P7 attempted** local `psutil` sampling during the load test but measured the wrong process (the benchmark client, not the server subprocess) — disclosed as a measurement gap in `docs/MODULE10_RESULTS.md`'s Load/Concurrency section, not fixed. GPU: N/A, not used by this app. |

---

## 13. Privacy, Security & Responsible AI

| Item | Status | Evidence |
|---|---|---|
| PII | ✅ | Regex detection (email/phone/id), flag-and-continue, `pii_service.py`; recall evals |
| GDPR / DPDP / HIPAA | ⚠️ | No formal compliance assessment doc |
| RBAC | ✅ | `Tenant.role` (admin/member), `Settings.admin_client_names` (`app/core/config.py`), enforced through a central permission registry (`app/core/permissions.py`: permission constants + role→permission map + one `check_permission()` function) — not two copy-pasted inline checks. Two permissions populated today, only active when `DATABASE_URL` is set: `document_delete` (gates `DELETE /documents/{id}`) and `document_list_all_tenants` (gates `GET /documents?all_tenants=true`, cross-tenant visibility for oversight). Adding a new gated action is a 2-line addition to the registry plus one `check_permission()` call at the route, not a new pattern. Small by design — proportionate to this app's actual action surface — but the *mechanism* is now genuinely reusable, which is what "not a general permission system" was flagging. |
| Encryption | ✅ at-rest full scope (expanded 2026-09-22/23), ⚠️ transport | Transport: TLS via the Caddy overlay is now the documented default deploy path (`docs/OPERATIONS.md`), but still needs a real domain and so can't be unconditional — the no-TLS fallback still exists (LOCAL HTTPS is documented/configurable, not a live public deployment). At-rest: **every genuinely sensitive persisted text surface is now encrypted**, not just chat content. `ChatTurn.content` and `ChatSession.title` (via shared `encrypt_text_field`/`decrypt_text_field` helpers, `session_id`-bound AAD), feedback `comment` (`message_id`-bound AAD), uploaded PDF files on disk (`app/services/upload_service.py::encrypt_upload_bytes`/`decrypt_upload_bytes`, `document_id`-bound AAD, decrypt-to-tempfile during ingestion / decrypt-to-memory for the file-serving and page-highlight routes), and FAISS `metadata.json` chunk text (`chunk_id`-bound AAD, decrypt-once-at-`load()`/encrypt-once-at-`save()` so the in-memory working set BM25 needs stays plaintext with zero per-query overhead). All via AES-256-GCM (`app/core/encryption.py`). Key from `Settings.encryption_key_b64`, never hardcoded; missing/wrong/tampered key fails closed on every field; legacy plaintext (written before each field's own encryption rollout) remains readable, no forced migration. 45 new tests across three additions (`tests/test_feedback_encryption.py` 12, `tests/test_upload_encryption.py` 9 incl. a real-PyMuPDF-extraction round trip, `tests/test_faiss_metadata_encryption.py` 11 incl. a real BM25-lexical-match round trip) on top of the existing 33. Full regression: 1080 passed, 1 skipped, 0 failed. Platform-level encryption (an encrypted EBS volume) remains unconfigured since no live deployment exists; no key rotation. Evidence: `eval/module10/reports/encryption_at_rest_final_*.json`. |
| Consent | ✅ | Signup (`POST /auth/signup`) requires `consent: Literal[True]` on the request schema (`schemas.py:SignupRequest`) — Pydantic rejects `consent=false` or a missing field with 422 before account creation runs. This is the app's first feature that stores real PII (email, password hash); the checkbox ships in the same change that introduces that storage, not bolted on after. Frontend: `pages/Signup.jsx`'s consent checkbox, required to submit. |
| Secrets | ⚠️ | AWS SSM Parameter Store is now the documented recommended path (`docs/OPERATIONS.md` "Secrets"), with plain `.env` as the explicit fallback — see "Secret management" row below for the same status. |
| Prompt injection | ✅ | Untrusted-excerpt markers + eval resistance metrics |
| Jailbreak | ⚠️ | Covered via injection markers in eval, no dedicated jailbreak suite |
| Authentication | ✅ | Two parallel paths, both real: `X-API-Key` (SHA-256 hashed, per-client — scripts/CI/service clients) and individual user login (`POST /auth/signup`/`/auth/login`, bcrypt-hashed passwords, JWT bearer tokens — the web frontend). `app/core/auth.py`'s `require_auth` tries JWT first, falls through to the unchanged API-key path if absent. |
| Authorization | ✅ | Tenant-ownership scoping for most actions, plus a real permission registry (`app/core/permissions.py`) gating two actions (document deletion; cross-tenant document listing) — replaced two duplicated inline `if role != "admin"` blocks with one reusable, centralized enforcement function (`check_permission()`), a fixed permission vocabulary, and an explicit role→permission map, extensible by adding a constant + one call site rather than copying a block. Deliberately still a *small* permission set (2 actions, 2 roles) — that's proportionate to this app's scale, not a remaining gap; see `tests/test_permissions.py` for the registry's own unit tests, including the role=None asymmetry between the two permissions. |
| PII detection | ✅ | |
| Secret management | ⚠️ | `_load_secrets_from_ssm()` in `app/core/config.py` is now the documented recommended path for the EC2 deploy (`docs/OPERATIONS.md` "Secrets", with concrete `aws ssm put-parameter` + IAM steps); plain gitignored env vars are the explicit fallback rather than the only option shown. Still ⚠️, not ✅ — nothing in the code enforces SSM over plain env vars, so a deploy can still skip it. |
| Human approval | ⚠️ | See §1 — web search and document deletion both gated, off by default, no general approval queue |
| Audit logs | ✅ | `audit_event` lines + `usage_logs` table |
| PII Recall | ✅ | `eval/pii_recall_check.py` |
| Unauthorized Access Rate | ✅ | **Measured value: 0.0 (0/2 cross-tenant attempts), corrected 2026-09-19** — `eval/unauthorized_access_check.py`. The prior 0.3333 (1/3) folded a same-tenant "member"-role delete into the unauthorized-attempts denominator; investigation confirmed `app/core/permissions.py`'s `ROLE_PERMISSIONS` intentionally grants members `DOCUMENT_DELETE` (that delete is authorized by design, not an attack). The script was corrected to measure the two genuinely cross-tenant attempts only, with the authorized same-tenant path checked separately and confirmed still working (`check_member_can_delete_own_tenant_document`, PASS). Historical 0.3333 artifact preserved (`security_eval_20260919T103952Z.json`); corrected artifact: `security_eval_20260919T111236Z.json`. See `docs/MODULE10_GAP_CLOSURE_REPORT.md`. |
| Prompt Injection Success Rate | ✅ | `prompt_injection_success_rate` in `run_eval.py` (successful injection attacks / adversarial attempts — the checklist's literal framing, computed as `1 - injection_resistance` from the same per-entry flags) + prompt-builder unit tests (`tests/test_security.py`); regression-gated (`regression_check.py`'s `LOWER_IS_BETTER`). Previously this row only cited `injection_resistance`, the inverse-framed metric — corrected, both are now reported. |
| False Refusal Rate | ✅ | `run_eval.py:468-469` |
| Data Leak Rate | ✅ | `run_eval.py:471-473` |

---

## 14. Production Readiness

### Architecture

| Item | Status |
|---|---|
| Diagram | ✅ `docs/ARCHITECTURE.md` |
| Components | ✅ |
| Workflow | ✅ |
| Agent (autonomous goal) | ✅ Formal spec (`AGENT_ROLE`/`AGENT_GOAL`/`AGENT_BACKSTORY`), `prompt_builder.py` |
| Planner (selects steps) | ✅ `_plan` |
| Tools (documented capabilities) | ✅ README + ARCHITECTURE |
| Memory (what/how long) | ✅ Session store (bounded, LRU: 50 turns/session, 1000 sessions) feeds the last 6 turns into the LLM prompt; now also user-facing — `GET /chat/sessions` lists past conversations, `GET /chat/sessions/{id}` resumes one (`pages/History.jsx`), not just an internal context window |
| RAG (why retrieval necessary) | ✅ `NOT_APPLICABLE.md` + ARCHITECTURE |

### Evaluation / Debugging / Deployment / Security / Reliability / Cost / Documentation

| Area | Item | Status |
|---|---|---|
| Evaluation | Dataset (normal/edge/failure/adversarial) | ✅ `dataset_v1/v2.json` include all four case types; `dataset_v3.json` adds multi-modal case types (`image_caption`, `table_lookup`, `vision_qa`) |
| Evaluation | Metrics by cost of failure | ✅ Explicit rationale table — each metric mapped to the failure it detects and the cost of that failure shipping undetected, `eval/README.md`'s "Metric selection: cost of failure" |
| Evaluation | Human eval rubrics | ✅ `HUMAN_EVAL.md` |
| Debugging | Logs / Traces / Errors | ✅ / ✅ / ✅ |
| Deployment | Docker / Cloud / Monitoring | ✅ / ⚠️ / ⚠️ |
| Monitoring | Uptime probe / Log rollup / Alerts | ⚠️ / ✅ / ⚠️ |
| Security | Auth / Authorization / Secrets / Encryption | ✅ / ✅ / ⚠️ / ⚠️ |
| Reliability | Retry / Timeout / Fallback / Cache | ✅ / ✅ / ✅ / ✅ |
| Load/Concurrency | Real HTTP-boundary concurrency ladder | ⚠️ **Module 10 gap-closure (2026-09-21, P7)**: real `uvicorn` subprocess + real HTTP (`httpx`) concurrency test at levels 1/2/5/10/20, `eval/module10/runners/run_load_concurrency_final_eval.py`. Found a genuine, reproducible saturation event (`/chat` @ concurrency=20: 20/20 requests timed out) with clean recovery (`/health` healthy immediately after every level). ⚠️ not ✅: single local machine, single worker, not production-scale capacity — see `docs/MODULE10_RESULTS.md`'s Load/Concurrency section. |
| Cost | Tokens / Latency / Model routing / Cache / Vision | ✅ / ✅ / ✅ / ✅ / ✅ — per-request `estimated_cost_usd` covers every `generate*` call including image captioning / vision QA (gated off by default) |
| Docs | README / API / Architecture / Demo / Future work | ✅ / ✅ / ✅ / ✅ / ✅ |

### Final 10-question design review

| Question | Status |
|---|---|
| Why does this need an LLM? | ✅ `docs/DESIGN_REVIEW.md` |
| What decisions are delegated to the LLM? | ✅ |
| Five most likely failure modes | ✅ |
| How will each failure be detected? | ✅ |
| How will the system recover? | ✅ |
| How do you know the new version is better? | ✅ |
| How will user data and secrets be protected? | ✅ |
| Cost per successful task | ✅ **current measured value: $0.001124** (real per-request Groq token/cost telemetry, `eval/module10/reports/agent_eval_20260919T112455Z.json`). The `~$0.0006` figure in `DESIGN_REVIEW.md:237-277` is a superseded, config-derived *estimate* from an earlier pass against Gemini pricing — kept there labeled historical, not the current headline. |
| What breaks from 10 → 1M users? | ✅ |
| Would you trust it as a customer? | ✅ |

### Cost metric

| Metric | Status |
|---|---|
| Cost per Successful Task = Total System Cost ÷ Successful Tasks | ⚠️ Manual derivation in `DESIGN_REVIEW.md`; per-request `estimated_cost_usd` now automated in `chat_query_handled` (`usage_tracking.py`), so the rollup is computable from logs — but not yet emitted as a single metric. |

---

## Known doc drift (not checklist items, but affect accuracy of this file)

- *(fixed)* bcrypt → SHA-256: `config.py:56`, `.env.example:21`, `ARCHITECTURE.md`, `DESIGN_REVIEW.md`, `NOT_APPLICABLE.md` all corrected (code uses SHA-256, `auth.py:82`).
- *(fixed)* Deployment status: README + `NOT_APPLICABLE.md:17` now state the live-but-ephemeral free-tier reality (`OPERATIONS.md:561` / `DESIGN_REVIEW.md:322`).

## Work queue (as decided)

Order chosen: **Metrics+eval → LLMOps+CI → Docs+drift → Monitoring → Tool hardening → Human approval + structured output.**

- ✅ Metrics + eval expansion
- ✅ LLMOps + CI gates (eval regression gate in `eval.yml`, security unit tests)
- ✅ Docs + drift fixes (bcrypt, deployment status, agent spec, `docs/demo/DEMO.md`)
- ✅ Monitoring scaffold (`monitoring/uptime_check.py`, `monitoring/log_aggregate.py`, `monitoring.yml`)
- ✅ Tool hardening (`app/services/tool_registry.py` formal I/O schemas + `@track_tool` success-rate logging; web-search retry; per-request cost rollup via `app/core/usage_tracking.py`; `tests/test_tool_registry.py`)
- ✅ Human approval + structured output (`confirm_web_search` gate; `generate_structured` JSON-mode + `StructuredAnswer` validation with free-text fallback; `tests/test_human_approval_structured_output.py`) — both config-gated off by default
- ✅ RBAC/Secrets/Unauthorized Access Rate closure (`Tenant.role` + `Settings.admin_client_names` gating `DELETE /documents/{id}`; SSM `SecureString` secret resolution at cold start; `eval/unauthorized_access_check.py`)
- ✅ Second round: human approval for document deletion (`Settings.document_delete_requires_approval` + `approved=true`, mirrors `confirm_web_search`'s shape); Prompt Injection Success Rate metric added under its literal checklist name (`run_eval.py`, `regression_check.py`); RBAC generalized to a second action (`GET /documents?all_tenants=true`, admin-only)
- ✅ Third round: Authorization/RBAC enforcement centralized into a real permission registry (`app/core/permissions.py`) — replaced the two duplicated inline role checks with one `check_permission()` function, a fixed permission vocabulary, and an explicit role→permission map; `tests/test_permissions.py` added
- ✅ Fourth round (§14 Production Readiness): metrics-by-cost-of-failure rationale table (`eval/README.md`); monitoring push alerts (`monitoring/alert_webhook.py`, wired into `uptime_check.py`/`log_aggregate.py`/`monitoring.yml`, off by default); dynamic model routing by prompt complexity/risk (`app/services/routing_llm_client.py`, `Settings.model_routing_enabled`, composes with the existing fallback wrapper in `llm_provider.py`)
- ⏭ Final status pass + commit (user review: pytest, live eval + regression gate, then commit)
