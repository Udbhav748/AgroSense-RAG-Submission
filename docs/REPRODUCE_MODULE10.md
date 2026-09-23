# Reproducing the Module 10 Evidence

All commands below are run from `backend/`. Each is labeled with what it needs beyond the checked-out repo — most need nothing beyond `pip install -r requirements.txt`; a few need live API credentials or an external service, called out explicitly so nothing here is claimed reproducible when it actually isn't.

## Prerequisites

```
cd backend
pip install -r requirements.txt
```

Everything in this document assumes `backend/.env` exists with at least `GEMINI_API_KEY` set (see `.env.example`). It is never committed to git — copy `.env.example` and fill in your own key.

## Full test suite

**Requires**: nothing external — fully offline (LLM/embedding/vector-store calls are mocked or stubbed throughout).

```
pytest -q
```

Expected: `1080 passed, 1 skipped (1081 collected)`.

## Targeted test subsets

**Requires**: nothing external.

```
pytest tests/test_agent_graph_production.py -q       # graph routing + faithfulness-fix + approval-wiring tests (15)
pytest tests/test_main.py -k ApprovalGate -q          # document-delete approval state matrix (8)
pytest tests/test_groq_client.py -q                   # LLM client error mapping + rate-limit test (11)
pytest tests/test_module10_telemetry_capture.py -q    # telemetry capture utility (6)
pytest tests/test_human_approval_structured_output.py -q  # structured output + web-search approval (11)
pytest tests/test_human_approval_node.py -q           # human_approval_node in isolation (6)
pytest tests/test_agent_graph_parallel_execution.py -q  # generic concurrency primitive (13)
pytest tests/test_handle_diagnose_parallel.py -q        # real diagnose-workflow concurrency (6)
pytest tests/test_encryption.py tests/test_postgres_session_store_encryption.py tests/test_session_repository_encryption.py -q  # encryption at rest, both fields (33)
pytest tests/test_settings_secret_validation.py -q       # production-mode weak-secret rejection (14)
pytest tests/test_run_rag_eval_retrieval_signature.py -q # eval-script retrieve() signature regression (3)
pytest tests/test_provider_ab_eval.py -q                 # provider A/B harness + paired significance test (12)
pytest tests/test_run_agent_eval_tool_arguments.py -q     # expanded tool-argument-accuracy ground truth (6)
pytest tests/test_nli_faithfulness.py -q                  # NLI groundedness upgrade attempt (7, incl. 1 real-model test)
pytest tests/test_feedback_encryption.py -q                # feedback comment encryption (12)
pytest tests/test_upload_encryption.py -q                  # uploaded PDF encryption, incl. a real PyMuPDF round trip (9)
pytest tests/test_faiss_metadata_encryption.py -q          # FAISS metadata encryption, incl. a real BM25 round trip (11)
```

## NLI groundedness upgrade evidence (honest negative result)

**Requires**: nothing external for the primitive itself (pretrained model, inference only); the full comparison run needs a real `GEMINI_API_KEY`/`GROQ_API_KEY` (20 live LLM calls) plus local NLI inference.

```
python eval/module10/runners/run_nli_faithfulness_eval.py
```

Runs the full 20-case dataset through both the existing lexical-overlap faithfulness proxy and a real pretrained NLI cross-encoder, side by side. Reproduces the honest negative finding: NLI scores are real but substantially and systematically lower than the lexical proxy on this corpus's structured, pipe-delimited retrieval-chunk format — a domain-mismatch limitation of generic pretrained NLI models, not a bug. See the saved report's `honest_finding_domain_mismatch` field for the full investigation (which models/premise formats were tried and why none generalized).

## Provider A/B evidence (with paired significance test)

**Requires**: real `GROQ_API_KEY` and `GEMINI_API_KEY` (live LLM calls to both providers, 20 cases each — real API cost).

```
python eval/module10/runners/run_provider_ab_eval.py
```

Runs the frozen 20-case dataset under both providers and computes a paired Wilcoxon signed-rank test + bootstrap 95% CI on the matched per-case faithfulness/composite-score differences. Exact p-value/CI/failure-rate numbers vary run-to-run (live provider behavior isn't deterministic) — the math itself is pinned offline by `tests/test_provider_ab_eval.py::TestPairedSignificanceTest` on synthetic data, no live calls needed to verify correctness.

## Faithfulness eval-script fix evidence

**Requires**: a real `GEMINI_API_KEY`/`GROQ_API_KEY` and a populated FAISS vector store (live LLM + retrieval calls).

```
python scripts/run_rag_eval.py
```

Re-runs the full 20-case golden dataset through the now-fixed `execute_retrieval()`. Expect `eval-potato-02` to score non-zero faithfulness (previously 0.0) and the overall mean faithfulness to land near 0.78, not the pre-fix 0.71 — exact numbers vary slightly run-to-run since LLM generation isn't fully deterministic.

## Encryption-at-rest evidence

**Requires**: nothing external — runs against a real in-memory SQLite-backed SQLAlchemy session.

```
python eval/module10/runners/run_encryption_at_rest_eval.py
```

Runs real encrypt/decrypt/tamper/wrong-key/missing-key/plaintext-leakage checks against the actual `PostgresSessionStore`/`session_repository` code (not a description of intended behavior) and reruns the encryption test files, reporting the real pass count. Saves a timestamped JSON artifact under `backend/eval/module10/reports/encryption_at_rest_final_<timestamp>.json` with an 11-category coverage matrix distinguishing protected vs. disclosed-unprotected data.

## Parallel execution performance evidence

**Requires**: nothing external — all I/O (vision, weather) mocked with real sleeps standing in for real latency.

```
python eval/module10/runners/run_parallel_execution_final.py
```

Reproduces a real serial-vs-parallel timing comparison over the identical `ChatService.handle_diagnose` call and identical mocked I/O, isolating concurrency as the only variable. Expect a parallel mean close to the max single-branch duration and well below the serial mean; exact numbers may vary slightly run-to-run but the direction and rough magnitude should reproduce. Saves a timestamped JSON artifact under `backend/eval/module10/reports/parallel_execution_final_<timestamp>.json`.

## RAG evaluation (Module 10 package, 30-case ablation)

**Requires**: a live LLM provider (Groq or Gemini, per `.env`'s `LLM_PROVIDER`) and an indexed document in the vector store (`backend/vector_store/`). Consumes real API quota.

```
python eval/module10/runners/run_rag_eval.py
```

Produces a new timestamped `eval/module10/reports/rag_eval_<timestamp>.json`; never overwrites a prior one.

## RAG benchmark (20-case golden agricultural dataset)

**Requires**: live LLM provider, indexed plant-pathology documents. Consumes real API quota; the live 2026-09-19 run took several minutes.

```
python scripts/run_rag_eval.py
python scripts/run_rag_eval.py --no-llm   # retrieval-only, no LLM calls, no quota consumed
```

Overwrites `data/eval_reports/latest_eval_report.json` in place (the script's own convention) — `docs/RAG_BENCHMARK_REPORT.md` is the durable, versioned record of each run's numbers.

## Agent evaluation (planner, planning success, cost)

**Requires**: live LLM provider, indexed document. Consumes real API quota (each of the 3 planning cases makes at least one real generation call).

```
python eval/module10/runners/run_agent_eval.py
```

## Security evaluation (PII, RBAC, injection, jailbreak, false refusal)

**Requires**: live LLM provider for the injection/jailbreak/false-refusal cases; the PII and unauthorized-access checks are fully mocked/offline and need no live calls.

```
python eval/module10/runners/run_security_eval.py
```

To re-verify just the RBAC finding, offline, no quota:

```
python eval/unauthorized_access_check.py
```

## Memory evaluation

**Requires**: nothing external — deterministic, in-process session-store checks.

```
python eval/module10/runners/run_memory_eval.py
```

## Failure/recovery evaluation

**Requires**: nothing external — all 11 measured scenarios are mocked failure injections.

```
python eval/module10/runners/run_failure_eval.py
```

## Rate-limit-surviving-retries (Phase 5)

**Requires**: nothing external — a deterministic mocked `groq.RateLimitError`, no real 429 call.

```
pytest tests/test_groq_client.py -k rate_limit -q
```

## Human evaluation (24 cases)

**Requires**: live LLM provider to capture the raw answers; scoring itself is manual (a human reviewer against the rubric in `docs/HUMAN_EVAL.md`) and cannot be automated/reproduced by a script alone.

```
python eval/module10/runners/run_human_eval_new_rows.py   # captures rows 17-24's raw answers
```

Rows 1-16's answers were captured directly via `ChatService.handle_query()` in an earlier session (see `docs/HUMAN_EVAL.md`'s "Recorded answers" section) — not re-capturable by a single script since it predates the Module 10 package's own runners.

## Faithfulness post-fix verification — targeted (Phase 5, rows 17/19 only)

**Requires**: live LLM provider. Consumes real API quota (2 real generation calls). **Not a full-dataset re-run.**

```
python eval/module10/runners/run_faithfulness_post_phase3.py
```

## Faithfulness post-fix verification — full 20-case dataset (Phase 7)

**Requires**: live LLM provider. Consumes significant real API quota (20 real generation calls, several minutes). This is the authoritative full-dataset Faithfulness measurement, superseding the 2-case targeted verification above for full-dataset claims.

```
python scripts/run_rag_eval.py
```

Produces `data/eval_reports/latest_eval_report.json` (overwritten in place by the script's own convention); the durable, versioned snapshot is `eval/module10/reports/faithfulness_full_postfix_<timestamp>.json` and `docs/RAG_BENCHMARK_REPORT.md`'s "POST-FIX FULL RE-RUN" section.

## Multimodal (LeafSense) evaluation

**Requires**: the LeafSense vision service running as a separate process at `settings.vision_service_url` (default `http://127.0.0.1:8001`). If it's not reachable, the runner saves a report with `"status": "BLOCKED"` rather than fabricating results — this is by design, not a bug.

```
python eval/module10/runners/run_multimodal_eval.py
```

## Full Module 10 runner (everything above except the human/faithfulness/multimodal manual pieces)

**Requires**: live LLM provider, indexed documents. Consumes significant API quota — this runs the RAG, agent, security, and memory evaluators in sequence.

```
python eval/module10/runners/run_full_module10.py
```

## Secrets hygiene check

**Requires**: nothing external — pure git inspection.

```
git log --all --full-history -- backend/.env      # expect: empty (never tracked)
git grep -nE "gsk_[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}"   # expect: no matches
```

## What is NOT independently reproducible from this repo alone

- **LeafSense diagnostic accuracy on real photos** — no real, labeled plant-disease photo corpus exists in this repository or is fetched by any script here; the multimodal evaluation above only exercises pipeline robustness on synthetic images.
- **A second human-evaluation reviewer's scores** — Inter-Annotator Agreement is N/A by design (one reviewer); reproducing IAA would require a second person to independently score the same 24 cases, which this repository cannot automate.
- **The exact historical (August 2026) RAG benchmark numbers** in `docs/RAG_BENCHMARK_REPORT.md`'s original section — no surviving raw log/artifact for that run exists to re-verify against; the REFRESH section's 2026-09-19 numbers are the reproducible ones.
