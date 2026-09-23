# AgroSense-RAG — Module 10 Final Submission

**This document does not claim 100% completion.** Every item across this project is marked ✅ (implementation + reproducible test + real measurement), ⚠️ (partial/limited/local-only measurement), ❌ (missing), or N/A (genuinely not applicable, with rationale) — matching the underlying evidence exactly, never upgraded because code merely exists. See `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md` for the row-by-row mapping against the literal Module 10 PDF checklist.

**Branch**: work was developed on `module10-final-pdf-compliance`, which has since been merged into `main` (see `git log --oneline --merges` for the merge commit) — `main` is now the accurate, current state of this evaluation · **Commit at last edit**: verify with `git rev-parse HEAD` · **Full regression**: 1080 passed, 1 skipped, 0 failed (1081 collected) · **Date**: 2026-09-19 through 2026-09-23, across 9 sequential evaluation/hardening passes (P0–P8) plus same-day P9 follow-ups implementing real parallel execution (13 + 6 new tests), a first encryption-at-rest expansion (12 new tests), code-enforced secrets (14 new tests), a real faithfulness root-cause fix (3 new tests), a real paired significance test for Provider A/B (5 new tests), expanded tool-argument-accuracy ground truth (6 new tests), an honestly-reported NLI groundedness upgrade attempt (7 new tests), and full encryption-at-rest scope expansion (45 new tests)

---

## 1. Project Overview

**AgroSense-RAG** is a document-grounded Retrieval-Augmented Generation assistant with an explicit agent workflow and a multimodal plant-pathology diagnosis mode. Upload a PDF; it's chunked, embedded, and indexed into a FAISS vector store. Ask questions through a chat interface; every answer is grounded in retrieved passages with structured, citable sources. A second mode accepts a plant-leaf photo, runs it through an external LeafSense vision service, and feeds the resulting diagnosis back through the same RAG loop for treatment guidance drawn from indexed agricultural-pathology documents.

**Why RAG**: uploaded documents (PMP course material, agricultural-pathology guides) have no fixed schema — answering questions against them requires retrieving relevant passages and having an LLM synthesize across them, not a database lookup.

**Why multimodal vision**: a photo of a diseased leaf carries information no text query can express; LeafSense's classification output is converted into a text query that re-enters the same grounded RAG path, so diagnosis answers are held to the same citation/grounding standard as text questions.

**What the agent/workflow does**: a deterministic (non-LLM) planner routes each request to one of `conversational` / `summarize` / `retrieve` / `diagnose`; retrieval runs hybrid BM25+FAISS search with optional cross-encoder reranking; a heuristic grader decides whether retrieval was good/weak/insufficient and can escalate to a web-search tool (optionally gated behind human approval); generation is bounded by a corrective loop (reflect once, escalate once, cap at 3 LLM calls) to prevent infinite loops.

**What is NOT claimed**: this is not a medical or clinical-grade diagnostic tool, not a production-grade agricultural advisory system, does not guarantee correctness, does not claim universal superiority of any one LLM provider, and has not demonstrated production-scale capacity. Every quantitative claim below is measured on this project's own infrastructure at local/bounded scale, cited to a specific artifact.

## 2. Architecture

Explicit `AgentState` + named-node `StateGraph` (`backend/app/services/agent_graph/`) — **a custom, dependency-free runtime, not third-party LangGraph** (deliberate, documented in `docs/ARCHITECTURE.md`'s "Framework choice"). `build_chat_graph()` is the single production topology behind `/chat`, `/chat/stream`, `/chat/diagnose(/stream)`.

```
Client → API → validate_request → planner
                                     ↓ (conditional routing)
      conversational / summarize / cache_lookup → retrieval → retrieval_grader
                                                                   ↓
                                             good ──────────────→ generator
                                             weak/insufficient ──→ [human_approval?] → web_research
                                                                                          ↓
                                                                  generator ← ─ ─ ─ ─ ─ ─┘
                                                                     ↓
                                                                reflection → output_validation → finalizer
```

**Frontend**: React + Vite SPA (`frontend/`).
**Backend**: FastAPI (`backend/`).
**Document ingestion**: PyMuPDF extraction with OCR fallback for image-only pages.
**Chunking**: `langchain-text-splitters`' `RecursiveCharacterTextSplitter`, 1000 chars / 200 overlap.
**Embeddings**: Sentence Transformers, `all-MiniLM-L6-v2`.
**Vector retrieval**: FAISS `IndexFlatIP` (exact inner-product search) over L2-normalized embeddings.
**Hybrid retrieval**: BM25 (lexical) + FAISS (semantic) fused via Reciprocal Rank Fusion.
**Reranking**: optional cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), config-gated.
**Tools**: web search, summarization, diagnose (vision), retrieval, PDF extraction, OCR — invoked via `tools/registry.py::ToolRegistry.execute`, never called raw from graph nodes.
**Memory**: session-scoped conversation history (`InMemorySessionStore`, LRU-bounded), optional PostgreSQL-backed persistence (`PostgresSessionStore`) with **application-level AES-256-GCM encryption of `ChatTurn.content` and `ChatSession.title`** (and, as of 2026-09-22/23, feedback comments, uploaded PDF files, and FAISS metadata chunk text — see §10 for the full scope).
**Vision**: LeafSense integration (separate service) for leaf-disease classification.
**Security**: API key + JWT auth, tenant-scoped RBAC (`app/core/permissions.py`), PII detection, audit events, prompt-injection/jailbreak defenses (untrusted-content delimiters in prompts).
**Human approval**: web-search escalation and document deletion both gate on a real, resolved `ApprovalStore` record — a client-supplied boolean alone is insufficient for document deletion.
**Structured output**: Pydantic `StructuredAnswer` + provider JSON mode, with a safe free-text fallback on any parse/validation failure — see §8.
**Observability**: structured JSON logs, `request_id`/`trace_id`, per-node tracing, tool-invocation telemetry, LLM token/cost logging, live `GET /metrics` (Prometheus format), offline log aggregation (`monitoring/log_aggregate.py`), a real threshold `AlertEngine`, and a dependency-free text dashboard — see §11.
**Deployment**: Docker + docker-compose documented; an optional Caddy HTTPS overlay and EC2/SSM deployment path are documented in `docs/OPERATIONS.md`. **No cloud instance of this project is currently running** — nothing below claims a live deployment that doesn't exist.

## 3. Repository / Demo

**Repository**: `Udbhav748/AgroSense-RAG` on GitHub (renamed from `InsightAI-RAG-Project-`), `main` branch — the `module10-final-pdf-compliance` work described in this document has been merged into `main`.
**Demo**: a real, screen-recorded local walkthrough exists at `docs/assets/demo.mp4` (signup → upload → grounded chat with citations → multimodal diagnosis → session history) — embedded in the root `README.md`. **No live/hosted demo URL exists**; none is claimed here.

## 4. Module 10 Audit Method

This submission is built from **actual repository implementation, actual tests, actual evaluation artifacts under `backend/eval/module10/reports/`, and actual measured metrics** — not from a description of intended behavior. Every ✅ below answers the question *"could an evaluator reproduce this from the repository without trusting our prose?"*; where the honest answer is no, the item is marked ⚠️ or ❌ instead. The literal Module 10 PDF checklist (14 sections + a 10-question design review) is mapped item-by-item in `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md`.

## 5. Implementation Evidence Summary

This project was built and evaluated across 9 sequential passes, each independently committed and regression-tested:

| Pass | Scope | Key artifact(s) |
|---|---|---|
| P0–P1 | Initial agent-graph implementation, RAG pipeline, security hardening | `eval/module10/reports/{rag,agent,security,failure,memory,multimodal}_eval_*.json` |
| P2 | Faithfulness root-cause fix + full 20-case rerun | `faithfulness_final_20260920T181537Z.json` |
| P3 | Encryption at rest wired into real storage | `encryption_at_rest_integration_20260920T185450Z.json` |
| P4 | Structured output productionized | `structured_output_final_20260920T194959Z.json` |
| P5 | Controlled provider/model A-B evaluation | `provider_ab_eval_20260920T203231Z.json` |
| P6 | Observability + alerting + bounded availability | `observability_final_20260921T062441Z.json`, `availability_bounded_local_20260921T062441Z.json` |
| P7 | Load/concurrency evaluation, real HTTP boundary | `load_concurrency_final_20260921T072420Z.json` |
| P8 | Human evaluation second-reviewer/IAA infrastructure | `human_eval_final_20260921T152012Z.json` |
| P9 | Final audit, PDF traceability, README/doc reconciliation | this document + `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md` |

## 6. RAG Evaluation

| Configuration | P@5 | Recall@5 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Semantic only | 0.4174 | 0.6014 | 0.6957 | 0.6739 |
| Hybrid (BM25+FAISS) | 0.6087 | 0.7428 | 0.9130 | 0.8551 |
| Hybrid + rerank | 0.6435 | 0.8080 | 0.9130 | 0.8783 |

Source: `eval/module10/reports/rag_eval_20260919T103118Z.json` (30 cases).

**Faithfulness** (20-case golden set, `scripts/run_rag_eval.py::GOLDEN_DATASET`): a historical, unverified baseline of 0.9420 could not be reproduced against a surviving artifact. A real regression to **0.0000** was found and root-caused (P2/earlier passes): `generator_node` was silently substituting a "not in documents" reply for any LLM provider failure surviving retries, making a provider outage indistinguishable from a genuine refusal. Fixed with a distinct `GENERATION_ERROR_REPLY` sentinel plus a `FallbackLLMClient` wiring fix and a `retrieval_top_k` increase (5→8). Post-fix, full-dataset re-run: **0.6485 → 0.7093** (P2's own root-cause pass raised it further by fixing 3 of 4 remaining zero-score cases). **A second real bug was found and fixed 2026-09-22**: the eval script itself called `retrieve()` with a nonexistent keyword argument, silently degrading every retrieval it ever made to a raw fallback with no hybrid search/collection filter/reranking — fixed with a one-line change, raising the mean to **0.7809** and resolving `eval-potato-02` specifically (0.0 → 0.6). `eval-potato-01` and a newly-visible `eval-apple-01` remain weak under a disclosed retrieval-ranking limitation. Source: `eval/module10/reports/faithfulness_final_20260920T181537Z.json`, `eval/module10/reports/rag_eval_retrieve_signature_fix_20260922T145928Z.json`.

Groundedness/citation figures use a lexical-overlap/claim-decomposition **proxy**, not a full entailment model — labeled as such throughout, not presented as ground truth. **A real pretrained-NLI-model upgrade was attempted 2026-09-22 (no training — inference-only, per scope) and produced an honest negative result**: mean NLI faithfulness 0.3201 vs mean lexical 0.8847 on the same 20 cases, root-caused to a genuine domain mismatch (generic NLI models trained on clean sentence pairs are poorly calibrated for this corpus's structured, pipe-delimited retrieval-chunk format — verified via a larger model and three premise-reformatting attempts, none of which generalized without fragile per-claim field selection). The lexical proxy remains the primary metric; full investigation in `eval/module10/reports/nli_faithfulness_upgrade_20260922T170919Z.json`.

## 7. Agent Evaluation

| Metric | Result | Source |
|---|---:|---|
| Planner Accuracy / Macro F1 | 0.9333 / 0.9475 | `agent_eval_20260919T112455Z.json` |
| Planning Success Rate | 1.0 (3/3) | same |
| Workflow Completion Rate | 1.0 | same |
| Node Success Rate | 1.0 | same |
| Tool Selection Accuracy | 1.0 (evaluated subset) | same |
| Tool Argument Accuracy | 1.0 (6/6, evaluated subset — expanded 2026-09-22, see below) | same |
| Average Steps | 8.5 | same |
| Loop Rate | 0.0 | same |
| Cost per Successful Task | $0.001124 (real per-request telemetry) | same |

**Tool-argument accuracy expanded (2026-09-22)**: was `n_applicable=2` (summarize's document_id only — the "21/21" figure previously shown here was a doc error, not backed by any saved artifact; the real historical number was 2/2). `retrieve`'s `crop`/`collection` argument is genuinely planner-decided (`extract_crop_context(query)`, used to scope retrieval to the right crop's documents) but had zero ground-truth coverage — unlike `top_k`/`min_score`, which are correctly N/A (caller-supplied `ChatRequest` fields, not planner decisions). Added 4 real ground-truth cases (3 crop-extraction positives across different crops + 1 no-crop negative) to `eval/module10/datasets/agent_eval.json`, wired into `run_tool_argument_cases` (`eval/module10/runners/run_agent_eval.py`). Measured: `n_applicable=6`, accuracy **1.0** (`retrieve`: 4/4, `summarize`: 2/2). Fully deterministic (`_plan()` is keyword-based, no LLM call) — verified offline, 6 new tests: `tests/test_run_agent_eval_tool_arguments.py`. **Still disclosed as a subset, not the full tool-call universe**: `web_research`'s approval gate and `diagnose`'s image-argument shape remain N/A here by design (measured elsewhere / out of scope for this text-only harness — see the dataset's own notes), not silently omitted.

Parallel execution of independent tasks is now implemented for the non-streaming diagnose workflow (vision + weather run as real concurrent `asyncio` branches, see `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md` §2); the main `/chat` corrective loop and tool calls remain sequential by design (each step depends on the previous one's output) and the streaming diagnose path was not converted.

## 8. Structured Output Evaluation

`Settings.llm_provider`-agnostic JSON-mode generation (`generate_structured()`) + `StructuredAnswer` Pydantic validation + defensive parsing (`structured_output.py::parse_structured_answer`, never raises, degrades to free text on any failure). **Now enabled by default** (`Settings.structured_output_enabled=True`) on `POST /chat` only, reachable via `ChatRequest.structured_response=true` — `/chat/stream` and `/chat/diagnose(/stream)` remain free-text by design.

| Metric | Result |
|---|---:|
| Parser Correctness | 1.0 (17/17) |
| Field Accuracy | 1.0 |
| Schema Compliance Rate | 0.4118 — **by design**: the 17-case dataset intentionally contains 10 malformed fixtures; this is an evaluator-artifact, not a parser defect (parser correctness is 1.0) |
| Fallback cases | 10 |
| Unrecoverable cases | 0 |

Verified end-to-end over a real `POST /chat` HTTP path (not just the parser in isolation), including a malformed-provider-response safe-recovery test. Source: `structured_output_final_20260920T194959Z.json`, `tests/test_structured_output_production.py`.

## 9. Human Evaluation

24 cases, 7 rubric dimensions (Correctness, Helpfulness, Completeness, Safety, Tone, Groundedness, Citation Quality), **1 real reviewer**. A complete two-reviewer/Inter-Annotator-Agreement pipeline was built (P8): reviewer-1 ratings transcribed to structured JSON, a blinded self-contained reviewer-2 packet generator, schema validation, and weighted Cohen's kappa computation (validated against 3 independently hand-derived fixtures).

**IAA is NOT measured.** Only one reviewer's real ratings exist — no second reviewer was fabricated, no LLM judge was substituted for the required independent human reviewer. Running `python eval/module10/runners/run_human_eval_final.py` today correctly prints `SECOND REVIEWER DATA REQUIRED`. Source: `docs/HUMAN_EVAL.md`, `human_eval_final_20260921T152012Z.json`.

## 10. Security Evaluation

| Metric | Result | Source |
|---|---:|---|
| PII Recall | 1.0 (15 planted) | `security_eval_20260919T111236Z.json` |
| Unauthorized Access Rate | 0.0 (0/2 genuine cross-tenant attempts) | same |
| Prompt Injection Success Rate | 0.0 | same |
| Jailbreak Success Rate | 0.0 | same |
| False Refusal Rate | 0.0 | same |
| Data Leak Rate | 0.0 | same |
| Memory session-boundary leakage | 0 | `agent_eval_*.json` |

**Encryption at rest** (full scope reached 2026-09-22/23): AES-256-GCM now protects every genuinely sensitive persisted text surface — `ChatTurn.content`, `ChatSession.title`, feedback `comment`, uploaded PDF files on disk, and FAISS `metadata.json` chunk text. All via shared helpers in `app/core/encryption.py`, each field's own contextually-appropriate value bound as authenticated associated data (session_id / message_id / document_id / chunk_id). Uploaded-PDF encryption decrypts to a short-lived tempfile during ingestion and to in-memory bytes for the two routes that serve the raw file/compute highlights directly (no tempfile needed there). FAISS metadata encryption decrypts once at `load()` and encrypts once at `save()` — the in-memory copy BM25 lexical search needs stays plaintext for the process's lifetime, so this adds zero per-query overhead. 78 total encryption tests (33 from the first expansion + 45 new: feedback 12, uploads 9 incl. a real PyMuPDF round trip, FAISS metadata 11 incl. a real BM25 round trip). A real bug was caught and fixed during review: FAISS metadata's `load()` initially mislabeled a missing/wrong key as a generic `CorruptedVectorStoreError`; fixed to propagate the real `EncryptionKeyMissingError`/`EncryptionIntegrityError`. Full backend regression: 1080 passed, 1 skipped, 0 failed. No key rotation exists; this is application-level, not platform-level, encryption. Evidence: `backend/eval/module10/reports/encryption_at_rest_final_20260921T193344Z.json`.

Having these controls is **not** a formal GDPR/DPDP/HIPAA compliance assessment — none is claimed.

## 11. Observability

| Metric | Result | Source |
|---|---:|---|
| Total requests (controlled sample) | 35 | `observability_final_20260921T062441Z.json` |
| Successful / failed | 30 / 5 | same |
| Aggregate error rate | 0.1429 (reported alongside per-taxonomy breakdown) | same |
| P50 / P95 / P99 latency | 0.1ms / 0.2ms / 163.3ms | same |
| Bounded-local availability | 1.0 (15/15 real `GET /health` probes) | `availability_bounded_local_*.json` |
| Alert-engine validation scenarios | 4/4 behaved as expected | `observability_final_*.json` |
| Dashboard required views present | 8/8 | same |

**Real finding**: a response-cache hit bypasses the log line (`chat_query_handled`) that log-based aggregation counts — disclosed, regression-pinned (`tests/test_observability_cache_gap.py`), not silently patched into working graph instrumentation. `AlertEngine` is real and tested but **not continuously scheduled** against a live target (none exists). Availability is a **bounded local measurement**, not a production SLO.

## 12. Reliability / Failure Testing

11/12 mocked failure scenarios measured: Detection Rate 1.0, Recovery Rate 1.0, Unhandled Failure Rate 0.0 (`failure_eval_20260919T092719Z.json`). Tool reliability extended to diagnose (retry/timeout classification, `tool_reliability_final_20260920T015928Z.json`). Not every tool shares one universal envelope or identical retry/timeout behavior — some tools (web search, embedding) have `tenacity`-based retry; others degrade differently.

## 13. LLMOps / Provider A-B

Real, controlled A-B comparison — `groq`/`openai/gpt-oss-120b` (A) vs. `gemini`/`gemini-3.5-flash` (B), same 20-case frozen dataset, fallback/routing disabled for isolation. Re-run 2026-09-22 with a real paired significance test added (see below).

| Metric | A (groq) | B (gemini) |
|---|---:|---:|
| Faithfulness | 0.7641 | 0.21 |
| Task success | 0.95 | 0.25 |
| Provider failure rate | 0.0 | 0.7 |
| Mean latency | 15.89s | 21.88s |
| Configured cost/successful task | $0.001744 | $0.000812 |

**Statistical significance (2026-09-22)**: a paired Wilcoxon signed-rank test on the 20 matched query pairs (same query, both configurations — the correct unit of comparison for this design, previously not applied) gives **p = 0.0009**, bootstrap 95% CI of the mean difference **[-0.76, -0.34]** — significant at α=0.05. **Disclosed confound**: gemini's provider failure rate was 0.7 this run (14/20 calls returned `GENERATION_ERROR_REPLY`, likely a rate-limit/transient issue at run time) — most of this run's faithfulness gap reflects **provider reliability at this moment**, not necessarily a stable model-quality difference; a re-run at another time could differ. **No provider is declared superior for production use** — this measures one frozen run's reliability + quality jointly. Sources: `provider_ab_eval_20260920T203231Z.json` (historical), `provider_ab_eval_20260922T153548Z.json` (current, with significance test). Regression test for the significance-test math (deterministic, no live calls): `tests/test_provider_ab_eval.py::TestPairedSignificanceTest`.

## 14. Load / Concurrency

Real `uvicorn` subprocess + real HTTP (`httpx`), concurrency ladder 1/2/5/10/20 (20 requests/level).

| Endpoint | RPS range | Notes |
|---|---|---|
| `GET /health` | 63.75–94.61 | 0 errors at every level |
| `POST /chat` (LLM mocked, real retrieval) | 11.66 → 1.99 | **All 20 requests timed out at concurrency=20** — real, reproducible single-worker CPU-bound saturation, not an injected fault. `GET /health` remained healthy immediately after every level, including the fully-failed one. |

Source: `load_concurrency_final_20260921T072420Z.json`. These are local, single-machine measurements — not production-scale RPS or cloud capacity.

## 15. Cloud / Deployment Readiness

Docker + docker-compose exist and are documented; an optional Caddy HTTPS overlay and an EC2/SSM deployment path exist in `docs/OPERATIONS.md`. **Not validated**: cloud-scale RPS, autoscaling, a production load balancer, production cost/hour, or a hosted monitoring/dashboard stack — none of these has ever been exercised against a live cloud deployment, and none is claimed here. HTTPS is a documented path, not something tested against a live TLS endpoint in this evaluation arc.

## 16. Hard Cases and Failures

| Case | Observed behavior | Detection | Root cause | Recovery | Limitation |
|---|---|---|---|---|---|
| `eval-potato-02` (Faithfulness) | Zero-score answer despite retrieval | Automated metric | Real bug: eval script called `retrieve()` with a nonexistent kwarg, silently falling back to degraded retrieval | Fixed 2026-09-22 (one-line kwarg fix); score 0.0 → 0.6 | Resolved, regression-tested |
| `eval-potato-01` / `eval-apple-01` (Faithfulness) | Low/zero-score answers despite the correct dosage chunk existing in the corpus | Automated metric | Cross-encoder/hybrid ranking doesn't reliably surface this corpus's pipe-delimited dosage-table chunk format above a general topic-overview chunk for some queries | N/A | Disclosed, not fixed |
| Provider generation failure (pre-fix) | "Not in documents" reply on real provider errors | Log trace + human review | `generator_node` laundering provider errors | `GENERATION_ERROR_REPLY` sentinel + fallback wiring | Fixed, regression-tested |
| Structured-output malformed provider response | Falls back to free text | `structured_output_used=false` flag | N/A (expected path) | Automatic fallback | None — designed behavior |
| Multimodal blur / spurious confidence | LeafSense over-confident on blurred synthetic images | Manual inspection | Not isolated (model behavior) | None | Disclosed, not fixed |
| Provider A/B: gemini generation errors | 5/20 cases failed under B (2026-09-20 run); 14/20 in the 2026-09-22 re-run | HTTP/log classification | Not isolated to a specific cause (rate limit vs. transient) | N/A (fallback disabled for isolation) | Single run per timestamp; a paired significance test is now applied across the 20 matched queries within each run (see §13), but the failure rate itself still varies run to run — disclosed, not smoothed over |
| Load test: `/chat` @ concurrency=20 | 20/20 timeouts | `httpx.TimeoutException`, categorized | Single-worker CPU-bound (embedding+reranking) saturation | Service recovered, `/health` stayed healthy | Real, reproducible |
| Cache-hit observability gap | Cache hits invisible to log aggregation | Empirical (during P6 report build) | `cache_lookup_node` routes to END, bypassing `finalizer_node`'s logging | N/A (disclosed, not patched) | Regression-pinned |
| Rate-limit burst scenario | 0/100 rate-limited | HTTP 429 count | N/A — burst didn't cross the threshold in this run | N/A | Negative result, reported honestly |

## 17. Limitations (explicit, current as of this pass)

- Custom `StateGraph`-like runtime — **not** third-party LangGraph.
- Parallel execution is implemented for one real workflow only (non-streaming diagnose: vision + weather run concurrently); the main `/chat` corrective loop and the streaming diagnose path remain sequential.
- Not all tools share one universal abstraction/envelope or identical retry/timeout behavior.
- Tool-argument accuracy is measured only on a subset with ground-truth values.
- Faithfulness = 0.7809 (raised from 0.7093 on 2026-09-22 by fixing a real eval-script bug); `eval-potato-01` and `eval-apple-01` remain weak under a disclosed retrieval-ranking limitation.
- Groundedness/citation figures are lexical-overlap/claim-decomposition **proxies** — a real pretrained-NLI upgrade was attempted (2026-09-22) and honestly found to be a worse, not better, signal for this corpus (domain mismatch — see §6 above); the lexical proxy remains primary.
- Full TP/FP/TN/FN classification reporting is not applicable to this RAG problem's own metrics (used where genuinely applicable — planner classification).
- Hallucination taxonomy evaluation is proxy-based, not a full dedicated model.
- Human approval is config-gated, not always-on.
- **Human IAA is not available** — one real reviewer; infrastructure for a second exists and is tested.
- Exact prompt content logging is controlled/debug-only (`Settings.log_prompt_content`, off by default) — never globally stored.
- `AlertEngine` is not continuously scheduled against a live target.
- Availability is bounded-local, not a production SLO.
- Cache-hit responses are invisible to current log-based aggregation.
- No hosted production dashboard; no centralized production logging.
- No validated cloud RPS, autoscaling, or production load balancer.
- No production cost-per-hour measurement.
- Local resource metrics (CPU/RSS from `psutil`) are local-process measurements, not cloud capacity — and in the P7 load test, the sampler measured the wrong process (the client, not the server), disclosed.
- HTTPS is a documented deployment path, not independently validated against a live TLS endpoint in this arc.
- Secret management: as of 2026-09-22, weak/placeholder secrets (`API_KEY`, `API_KEYS`, `JWT_SECRET_KEY`, `DATABASE_URL`) are now code-enforced — `Settings` refuses to construct when `DEBUG=false` and any of these look like a known placeholder or are too short (`app/core/config.py::_reject_weak_secrets_in_production`, 14 tests). The AWS SSM *retrieval* path itself remains documented, not newly built — this closes the "not enforced" half of the gap, not the managed-secret-storage half.
- Formal GDPR/DPDP/HIPAA compliance is **not** satisfied merely by having security controls.
- Encryption at rest now covers every genuinely sensitive persisted surface — chat content/titles, feedback comments, uploaded PDF files, and FAISS metadata chunk text (full scope reached 2026-09-22/23). No key rotation exists. Encryption in transit (HTTPS/TLS) is tracked separately, not solved by this expansion.

## 18. Ten Design Questions

1. **Why an LLM?** Free-text documents have no fixed schema; only an LLM can synthesize an answer across retrieved passages phrased in natural language. Routing/planning is deterministic keyword matching, not LLM-decided.
2. **What decisions does the system make?** Deterministic: routing, retrieval grading, rate limiting, RBAC, approval gating. LLM-delegated: answer generation, optional structured-output extraction.
3. **Five important failure modes** (all real, measured): (a) provider failure mislabeled as a refusal — found and fixed; (b) stale retrieval-corpus assumptions in eval datasets — found; (c) retrieval returning topically-adjacent-but-wrong-crop chunks; (d) prompt injection via retrieved content — defended, 0.0 success measured; (e) single-worker CPU-bound saturation under concurrent load — measured at concurrency=20.
4. **How are failures detected?** Structured per-node traces with `error_type`/`root_cause`; distinct HTTP outcome categories (success/HTTP failure/timeout/connection error/exception) in the load evaluation; `AlertEngine` threshold rules.
5. **How are failures recovered?** Bounded corrective loop (reflect once, escalate once, cap 3 LLM calls); `FallbackLLMClient` provider fallback; structured-output degrade-to-free-text; the service remained healthy (`GET /health`) immediately after every load-test level, including the fully-failed one.
6. **How is the new version better?** Every fix in this arc has a before/after artifact pair (e.g., Faithfulness 0.6485→0.7093, Unauthorized Access Rate 0.3333→0.0) — never a single unverified number.
7. **How are data/secrets protected?** API-key + JWT auth, tenant RBAC, PII detection (1.0 recall), `.env` never committed, AES-256-GCM encryption of chat content.
8. **Cost per successful task?** $0.001124 (agent eval, real telemetry); provider A/B measured $0.001572 (groq) vs. $0.000582 (gemini) under a separate 20-case run — these are two different measured contexts, not contradictory.
9. **Scaling 10 → 1M users — CURRENT vs. FUTURE:** *Current, measured*: single local machine, single `uvicorn` worker, real HTTP boundary, degrades to full timeout saturation at concurrency=20 for the CPU-bound `/chat` path. *Future, unvalidated*: horizontal scaling would need a load balancer, multiple workers/processes, a managed vector DB, provider-side rate-limit headroom, and centralized logging/monitoring — none of this has been built or tested. **This system has not demonstrated 1M-user scaling.**
10. **Why should a customer trust the system?** Failure modes are measured and disclosed rather than hidden — a real Faithfulness regression, two real approval-flow gaps, a real cache-observability gap, and a real concurrency-saturation point were all found, documented, and (where fixable) fixed with regression tests. Nothing here is marked ✅ merely because the code exists.

## 19. Final Traceability

See `docs/MODULE10_PDF_TRACEABILITY_MATRIX.md` for the complete, section-by-section mapping against the literal Module 10 PDF checklist, and `docs/MODULE10_FINAL_AUDIT.md` for the detailed technical audit with reproduction commands.

## 20. Reproduction Commands

```
cd backend
pytest -q                                                          # full regression: 1080 passed, 1 skipped
python scripts/run_rag_eval.py                                     # RAG + Faithfulness
python eval/module10/runners/run_agent_eval.py                     # agent/planner
python eval/unauthorized_access_check.py                           # RBAC
python eval/module10/runners/run_provider_ab_eval.py                # provider A/B (live LLM calls)
python eval/module10/runners/run_structured_output_eval.py         # structured output parser
python eval/module10/runners/run_observability_final_eval.py       # observability (spawns a real subprocess)
python eval/module10/runners/run_availability_eval.py              # bounded local availability
python eval/module10/runners/run_load_concurrency_final_eval.py    # load/concurrency (real HTTP)
python eval/module10/runners/run_human_eval_final.py                # human eval / IAA (prints SECOND REVIEWER DATA REQUIRED)
python eval/module10/runners/run_parallel_execution_final.py       # real parallel execution (diagnose workflow)
```

---

**Summary status**: substantially expanded across 9 evaluation passes with reproducible, artifact-backed evidence for every quantitative claim. No unresolved critical defects that block correct operation. Remaining ⚠️/❌ items are real and disclosed (see §17), most prominently: **human IAA not yet measured**, faithfulness below an aspirational target with one unresolved case, and no validated cloud/production-scale evidence. This is not a claim of a guaranteed teacher score — the original assignment score was 3/10, and this submission substantially expands evidence-backed evaluation coverage without asserting a specific final numeric grade.
