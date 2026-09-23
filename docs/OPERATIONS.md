# Operations

Procedures for comparing changes, rolling back, and deploying this app —
what actually exists to support each, and what's still manual.

## A/B testing prompt and model changes

There's no built-in A/B traffic-splitting — comparisons are done
offline, sequentially, using the eval harness:

1. **Establish a baseline.** With the current code/config, run:
   ```bash
   cd backend
   python eval/run_eval.py
   ```
   This writes `eval/results/<UTC-timestamp>.json`, capturing planner
   accuracy (confusion matrix, precision/recall/F1), Task Success Rate,
   the groundedness proxy, and Injection Resistance for the full
   `dataset_v1.json` run (see `eval/README.md` for what each metric
   means). Keep this file — rename it or note its timestamp as "baseline."

2. **Change one variant at a time:**
   - **Model variant**: change `GEMINI_MODEL_NAME` in `backend/.env`
      (e.g. try a different Gemini model) and restart the backend.
   - **Prompt variant**: edit `app/services/prompt_builder.py` and bump
      `PROMPT_VERSION` (e.g. `"v1"` → `"v2"`) alongside whatever prompt
      text changed, then restart the backend. Versioning it is what makes
      the change traceable in logs (`generation_requested`/
      `llm_generation_completed` both log `prompt_version`) — bump it for
      *any* prompt change, not just wording tweaks.

3. **Re-run the eval**: `python eval/run_eval.py` again against the
   changed backend. This writes a second, separate
   `eval/results/<timestamp>.json`.

4. **Compare the two result files directly** — `task_success_rate`,
   `groundedness_proxy`, `injection_resistance`, and the planner's
   `accuracy`/`macro_f1`/`weighted_f1` are the headline numbers; the
   `entries` array in each file has the raw per-query answers if you need
   to see *what* changed, not just the aggregate score.

Each results JSON now records `llm_provider`, `fallback_llm_provider`, and
`prompt_version` at the top level, so which variant produced a given run is
no longer something you have to reconstruct from logs — it's in the file
itself. (It still doesn't record `GEMINI_MODEL_NAME`/`GROQ_MODEL_NAME`
specifically if you change the model within a provider rather than the
provider itself; for that, cross-reference the run's timestamp against
`llm_generation_completed` log lines, which do carry `model_name`.)

Because this eval calls the real Gemini API and needs a real indexed
document (`eval/README.md`'s preconditions), it isn't run in CI
(`.github/workflows/ci.yml` has a TODO noting exactly this) — A/B
comparisons are a manual, deliberate step, not something that runs on
every push.

### Running the eval via GitHub Actions (CI/CD)

A separate manual workflow, `.github/workflows/eval.yml`, automates the
same local procedure on GitHub-hosted runners. It is **only triggered by
`workflow_dispatch` (manual run)** — never on push — because it calls the
real LLM API and consumes real quota.

**Prerequisites (one-time setup):**
1. In the GitHub repo settings → **Secrets and variables → Actions → New repository secret**, add:
   - `EVAL_GEMINI_API_KEY` — your real Google Gemini API key
   - `EVAL_GROQ_API_KEY` — your real Groq API key
   (These are *different* secret names from the dummy `ci-test-key` used
   by the CI workflow (`ci.yml`), so the two workflows' secret requirements
   never collide. CI uses a dummy key because its tests stub the LLM
   client; this workflow needs real keys.)

**To run:**
1. Go to the **Actions** tab → select **"Eval (Manual)"** on the left →
   click **Run workflow** (top right).
2. Choose inputs:
   - `dataset`: `dataset_v1.json` (default) or `dataset_v2.json`
   - `llm_provider`: `groq` (default, cheaper/higher free tier) or `gemini`
   - `delay`: seconds between entries (e.g. `15` for free-tier rate limits)
3. Click **Run workflow**.

The workflow will:
1. Check out the repo
2. Install Python + dependencies + tesseract
3. Seed the vector store with demo corpus (`backend/demo_corpus/pmp_key_concepts.pdf`) or agricultural knowledge base (`data/plant_disease_docs/`) via `backend/scripts/bulk_ingest.py`
4. Run `python eval/run_eval.py` with your chosen inputs
5. Upload the resulting `eval/results/<timestamp>.json` as a **workflow artifact** (retention: 30 days)

**To download the artifact:** After the run completes, open the workflow
run page → scroll to **Artifacts** → click the `eval-results-...` zip.

### Regression Rate check against the last committed baseline

Each run produces a timestamped JSON in `eval/results/`. The repo already
commits a baseline result file from a known-good run (e.g.
`eval/results/20260807T215538Z.json` for the v0.1.0 tag — see the
"Rollback plan" section). To use the workflow artifact as a regression
check:

1. **Download the artifact** from the workflow run (as above).
2. **Compare the new run's key metrics against the baseline file** —
   focus on the headline numbers that should *not* regress:
   - `planner.accuracy` (should stay at 1.0)
   - `planner.macro_f1` (should stay at 1.0)
   - `task_success_rate` (should stay ≥ baseline)
   - `injection_resistance` (should stay at 1.0)
   - `false_refusal_rate` (should stay at 0.0)
   - `data_leak_rate` (should stay at 0.0)
   - `precision_at_5` / `recall_at_5` / `mrr` (retrieval quality)
   - `citation_accuracy` (citation surface quality)
3. **Optional: commit the new run as the new baseline** if it represents
   an intentional improvement (e.g. after a prompt/model change that
   you're keeping). Rename or copy the downloaded JSON to
   `eval/results/<new-baseline-timestamp>.json`, commit it, and update
   the reference timestamp in this document's "Rollback plan" section.
   If the run shows a regression (any headline metric materially worse
   than baseline), do *not* commit it — investigate and revert the change
   instead.

The workflow artifact is the *evidence* for that decision; it exists
independently of the repo so you can inspect it without committing
anything automatically.

## Retrieval ablation

Two retrieval enhancements, `HYBRID_SEARCH_ENABLED` (FAISS semantic +
BM25 lexical, fused; **on by default**, see below) and
`RERANKING_ENABLED` (cross-encoder re-ranking of the candidate pool; off
by default) are config-gated specifically so they can be A/B'd against
each other and against a semantic-only baseline — same procedure as
above, but read `precision_at_5`/`recall_at_5`/`mrr` from the results
JSON instead (`eval/README.md`'s "Precision@5 / Recall@5 / MRR" section
has the exact definitions). These measure retrieval itself, independent
of what the LLM did with the chunks afterward, which is what makes them
the right metric for this specific comparison — Task Success Rate and
the groundedness proxy can move for reasons that have nothing to do with
retrieval quality (a different LLM sample, in particular).

**A real 3-way run** (`dataset_v2.json`, `LLM_PROVIDER=groq`,
7 content-bearing entries with `expected_chunk_keywords`; "baseline"
below means both flags off, i.e. not this project's actual default):

| Config | Precision@5 | Recall@5 | MRR | Results file |
|---|---|---|---|---|
| Baseline (both off) | 0.3143 | 0.8214 | 0.7500 | `eval/results/20260806T093248Z.json` |
| `+hybrid` | 0.3714 | 0.9643 | 0.9048 | `eval/results/20260806T093526Z.json` |
| `+hybrid+rerank` | 0.4000 | 0.9643 | 0.8095 | `eval/results/20260806T093845Z.json` |

What this run actually shows:

- **Hybrid search was a clean win on all three metrics** — BM25 catching
  exact-term matches (acronyms like "WBS"/"CPI", specific technique
  names) that a pure semantic search sometimes ranks lower than a
  looser paraphrase match. Recall@5 in particular jumped from 0.82 to
  0.96: one more of the "right" chunks made it into the top 5 per query
  on average.
- **Reranking improves Precision@5 further (0.37 → 0.40) at flat
  Recall@5 (0.96 → 0.96); the MRR delta (0.90 → 0.81) is noise, not a
  finding.** Checked directly against the raw per-entry data in
  `20260806T093526Z.json` vs. `20260806T093845Z.json`: exactly **one**
  of the 7 scored queries changed at all — the risk-monitoring edge
  case, whose first relevant chunk moved from rank 1 to rank 3 (still
  inside the top 5, just not first). `(1/1 - 1/3) / 7 ≈ 0.0952`, which
  is the entire observed MRR delta (0.0953) — one query, not a pattern,
  and at n=7 a single query is already ~14% of the sample.

  More importantly, **MRR is the wrong metric to weight for this
  system's architecture.** MRR measures how early the first relevant
  result appears — the metric that matters when a human scans a ranked
  list and clicks one link. `ChatService` doesn't do that: every
  retrieved chunk goes into the prompt as one flat context block
  (`prompt_builder.build_prompt`), so whether the relevant chunk landed
  at rank 1 or rank 3 makes no difference to what the LLM actually sees.
  What matters for this architecture is Precision@5 (how much of what's
  in context is relevant — less noise for the LLM to sift) and Recall@5
  (whether the relevant chunk made it into context *at all*). Both
  either improved or held under reranking. Read this ablation as
  **"reranking improves precision at flat recall; the rank-order
  metric it also moved on is both noise-level and the least
  decision-relevant one for how this system actually consumes
  retrieval results."**
- **Task Success Rate (0.8182) and Injection Resistance (1.0) were
  identical across all three configs**; the groundedness proxy actually
  *dipped* slightly with hybrid/rerank on (1.0 → 0.57 → 0.50). That's
  consistent with the metrics measuring different things: groundedness
  is lexical overlap between the *answer* and *whatever chunks came
  back*, and a wider/different candidate pool changes which chunks
  came back — it's not evidence retrieval got worse (Recall@5 says the
  opposite).

**Is 0.40 Precision@5 actually good? Only relative to how many relevant
chunks exist to find.** Computed directly from the indexed document's 99
chunks (`vector_store/metadata.json`), counting how many chunks actually
contain each query's `expected_chunk_keywords`:

| Query | Relevant chunks in corpus | Max achievable P@5 |
|---|---|---|
| "What is a project…" | 5 | 1.00 |
| "What is a Work Breakdown Structure?" | 1 | 0.20 |
| "What is a Business Case…" | 3 | 0.60 |
| "…CPI greater than 1…" | 3 | 0.60 |
| "…conflict resolution technique…" | 3 | 0.60 |
| "What is Earned Value Management used for?" | 36 | 1.00 |
| risk-monitoring edge case | 1 | 0.20 |
| **Mean** | **7.4** | **0.60** |

(Max achievable P@5 per query is `min(relevant_chunks, 5) / 5` — you
can't score higher than "all 5 slots relevant," and you can't clear
`relevant_chunks / 5` if fewer than 5 relevant chunks exist at all.)

The mean ceiling (0.60) is well above the measured `+hybrid+rerank`
score (0.40) — this system is **not** near a ceiling effect, there's
real precision headroom left. The variance is worth noting too: the
"Earned Value Management" query's ceiling is inflated by "measuring"/
"forecasting"/"performance" being generic project-management vocabulary
that appears in 36 of 99 chunks, not all of them really about EVM
specifically — the same keyword-substring heuristic that defines
Precision@5/Recall@5 (see `eval/README.md`) is doing the counting here
too, so it inherits the same noise for generic-word queries. WBS and the
risk-monitoring query sit at the other extreme: exactly one chunk in the
whole document is relevant, so no retrieval strategy can score above
0.20 on those specific queries no matter how good it is. Read 0.40
against 0.60, not against 1.0.

**Why hybrid search defaults on and reranking doesn't** (`app/core/config.py`):
hybrid search's improvement was unambiguous on every metric with no
added model/network dependency (BM25 is pure CPU, always available), so
it ships as the default rather than sit as an unproven opt-in. Reranking's
gain is real (flat recall, better precision) but thinner evidence (one
ablation run, n=7) against a real cost every other feature here doesn't
have: a second model loaded into memory and cross-encoder inference on
every `retrieve`-action request. That combination — genuine signal, small
n, nonzero cost — is exactly what config-gating an opt-in feature is for;
flipping it to default-on is a call worth revisiting once it's been run
against more queries or a different document.

**Why Groq, not Gemini**: the first attempt used the default
`LLM_PROVIDER=gemini` and hit the free tier's **daily** quota (20
requests/day for `gemini-3.5-flash`) partway through the very first
run — most entries after query 3 failed with `429
RESOURCE_EXHAUSTED`, and the aggregate metrics accordingly cratered
because the failure path scores those entries `0.0` rather than skipping
them (see `run_eval.py`'s except-branch). That contaminated run is not
included in the table above; it's a genuine illustration of why a clean
baseline matters before trusting an ablation number — see
`eval/results/20260806T092932Z.json` if you want to see what a
quota-exhausted run looks like. Switching to `LLM_PROVIDER=groq`
(already a supported provider, see "Comparing providers" below) gave a
clean run for all three configs; the two `summarize`-action entries still
failed on Groq's tokens-per-minute limit for the full-document prompt,
but that doesn't affect Precision@5/Recall@5/MRR, which only score
`retrieve`-action entries.

**To reproduce**: set `LLM_PROVIDER=groq` (or accept Gemini's daily cap
and spread the three runs across multiple days), then for each config,
set `HYBRID_SEARCH_ENABLED`/`RERANKING_ENABLED` in `backend/.env`,
restart the backend, and run `python eval/run_eval.py --dataset
dataset_v2.json --delay 5`.

## Bulk corpus ingestion

`POST /upload` goes through the frontend's 60s axios timeout (and
whatever patience sits in front of the server beyond that). OCR alone
runs roughly a second or more per page at `Settings.ocr_dpi`
(`document_service.py`) — a single 60-page scanned document can exceed
that timeout on its own, well before a real multi-document corpus would.
Don't bulk-ingest a corpus through the HTTP endpoint.

`backend/scripts/ingest_corpus.py` drives `DocumentProcessingService`
directly instead — one file at a time, from a plain Python process, no
HTTP layer or request timeout involved:

```bash
cd backend
python scripts/ingest_corpus.py path/to/corpus_dir
```

It walks `corpus_dir` recursively for `*.pdf` by default (`--pattern` and
`--no-recursive` adjust that), runs each file through the exact same
`validate_pdf_upload` + `DocumentProcessingService.process()` the HTTP
route uses — no pipeline logic duplicated — and prints per-file progress
(pages, chunks, pages OCR'd, duration) plus a final summary. One bad file
(corrupt, oversized, wrong type) is logged and skipped; it doesn't stop
the rest of the corpus.

By default it writes to the same index the running app uses
(`backend/vector_store/`). Pass `--index-path`/`--metadata-path` (both
required together) to build a separate index instead — useful for a dry
run, a scratch/experimental corpus, or comparing corpora side by side
without touching the live one. This is also the repeatable way to
rebuild the index from a known set of source PDFs whenever it's needed
— e.g. after a chunking/embedding config change, or once documents carry
shared/private metadata tags.

Each ingested file is still copied into `backend/uploads/` under a UUID
name, same as a real `/upload` (see `upload_service.py`) — ingesting a
large corpus roughly doubles its disk footprint, which is expected, not
a bug.

## Comparing providers (Gemini vs. Groq)

`app/services/llm_provider.py` picks the LLM implementation from
`Settings.llm_provider` ("gemini" or "groq"); both `GeminiClient` and
`GroqClient` implement the same `LLMClient` interface, log the same event
shape (`llm_generation_completed`, now including a `"provider"` field on
both), and share the same tenacity retry policy — so they're a valid,
apples-to-apples A/B pair using the exact procedure above:

1. Run the baseline with `LLM_PROVIDER=gemini` in `backend/.env`,
   `python eval/run_eval.py`.
2. Set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `backend/.env`, restart
   the backend, re-run `python eval/run_eval.py`.
3. Compare the two results files: `task_success_rate` and
   `groundedness_proxy` for quality, `processing_duration` (from the
   backend logs, keyed by `provider`) for latency, and
   `estimated_cost_usd` (also per-`provider` in the logs) for cost —
   Groq's LPU inference and per-token pricing are expected to win on
   latency and cost; Gemini is the current default because it was the
   original, validated integration.

As above, the results JSON's `llm_provider` field records which one
produced a given run, so the two files from steps 1-2 are
self-identifying — no need to track it separately.

## Automatic provider fallback

Setting `FALLBACK_LLM_PROVIDER` to the *other* provider (e.g.
`LLM_PROVIDER=gemini`, `FALLBACK_LLM_PROVIDER=groq`) wraps the primary
client in `FallbackLLMClient` (`app/services/fallback_llm_client.py`):
if the primary provider's own retries are exhausted
(`LLMTimeoutError`/`LLMAPIError`), one attempt is made against the
fallback provider before the request fails — logged as
`llm_fallback_triggered` with both provider names. This is single-hop
only (see the README's Known Limitations): if both providers are down,
the request still fails, and there's no automatic recovery back to the
primary once it's healthy again — a request-by-request decision, not a
sticky failover state.

## Query embedding cache

`embedding_service.embed_query` — the single function both
`retrieval_service.py` and `hybrid_search.py` call to embed an incoming
chat query — caches its result in-process via `functools.lru_cache`
(`maxsize=256`), keyed on the query string normalized by `.strip()`
alone (deliberately not case-folded — the normalized string is also
exactly what gets encoded on a miss, so a hit can never return an
embedding for text other than what that exact call would have produced
itself).

**Scope: query embeddings only, deliberately not retrieval results or
generated answers.** A query embedding is a pure function of (normalized
query text, model weights) — the model loads once per process
(`get_embedding_model`'s own `lru_cache`) and never changes mid-process,
so this can never go stale. Retrieval results and generated answers are
different: both depend on the vector store's *mutable* state (a document
can be uploaded or deleted between two otherwise-identical requests), so
caching either risks serving a stale answer after the underlying index
changed — a correctness bug, not just a missed optimization. That's why
only the embedding step is cached here.

One implementation detail worth being explicit about: the cached helper
(`_embed_query_cached`) returns a `tuple`, not the `list[float]`
`embed_query` itself returns to callers. `lru_cache` shares the exact
same return object across every cache hit — if the cache held a mutable
`list` directly, one caller mutating its result in place would silently
corrupt the value every other caller sharing that entry sees next.
`embed_query` always returns a fresh `list` built from the cached tuple,
so its return-type contract to callers doesn't change at all; only the
private cached layer deals in the immutable form.

**Measured latency** (`eval/embedding_cache_benchmark.py`, real
`all-MiniLM-L6-v2` model, cold model already loaded before timing
starts): averaged over 4 distinct queries, cache-miss latency
0.0538s vs. cache-hit latency 0.0001s — a **739x speedup, 99.9% lower
latency** on a cache hit. Run it yourself with:

```bash
cd backend
python eval/embedding_cache_benchmark.py
```

## Rollback plan

Nothing here is more sophisticated than git:

- **Tag known-good states.** Before a deploy, tag the commit:
  `git tag -a v0.1.0 -m "known-good: llama-3.3-70b-versatile via Groq, PROMPT_VERSION v1"`.
  Since `PROMPT_VERSION` is a source constant (`prompt_builder.py`), not
  an env var, the tagged commit *is* the pin for prompt version — there's
  no separate config file to keep in sync with it.
- **Pin the model via config, not code.** `GEMINI_MODEL_NAME`/`GROQ_MODEL_NAME`
  (and every other tunable in `app/core/config.py`) is read from
  `backend/.env` at startup. Keep whatever `.env` values were live for a
  given tagged release recorded somewhere outside git (`.env` is
  gitignored, deliberately, since `backend/.env.example` is the tracked
  template) — a deploy runbook or your platform's secret manager, not
  the repo.
- **To roll back**: redeploy the tagged commit's Docker image
  (`backend/Dockerfile`, already built and verified locally — see
  `docker-compose.yml`) with the `.env`/secrets that were live for that
  tag. Because `PROMPT_VERSION` and application code move together in
  git, checking out an old tag automatically reverts prompt behavior
  alongside everything else; only the model name (env var) needs to be
  restored separately.
- **Verify the rollback** by re-running `eval/run_eval.py` against it and
  confirming the metrics land back near that tag's original
  `eval/results/*.json`.

**This procedure has actually been exercised, not just written down.**
`v0.1.0` was tagged at `0e289c3`, checked out into an isolated
`git worktree` (so the build genuinely came from the tagged ref, not
just whatever happened to already be checked out), built into a Docker
image, and run against `eval/run_eval.py` — reproducing the same planner
accuracy (1.0), Task Success Rate (1.0), Injection Resistance (1.0),
tool-argument accuracy (1.0), and the same known Groq TPM 413 on the two
`summarize` entries as the baseline run
(`eval/results/20260807T215538Z.json`) it was compared against. The one
metric that moved (groundedness proxy: 0.8333 → 0.9167) is expected
run-to-run noise from LLM sampling on a 12-sample lexical-overlap
heuristic, not a regression — worth knowing as the expected variance
band before ever needing this procedure under real pressure.

There's no automated rollback trigger (no health-based auto-revert, no
canary) — this is a manual procedure, appropriate for the current
single-instance, not-yet-deployed scale (see next section and Q9 in
`docs/DESIGN_REVIEW.md`).

## Deploying to Render (historical)

Superseded by [Deploying to EC2](#deploying-to-ec2-docker-compose) below —
kept as-is as a historical record of the constraints that shaped the app
early on (the 512MB OOM incident in particular is referenced elsewhere in
the docs and shouldn't be deleted out from under those references).

**Backend → Render (Docker). Frontend → a static host (Vercel or
Cloudflare Pages), not a container.** `backend/Dockerfile` already builds
and runs correctly via `docker-compose.yml` (verified locally end-to-end:
healthcheck passing, `/health` and `/docs` responding) — Render deploys
that same image, not a rewrite. `frontend/Dockerfile`'s nginx stage stays
for local `docker-compose.yml` parity, but production frontend serving
moves off it: it's a static Vite build, and paying for container compute
(or routing it through `nginx`) to serve static files has no upside once a
CDN-backed static host does it for free with less to maintain.

Render over Cloud Run: **Cloud Run requires a GCP billing account (a
payment method on file) even to use its free-tier resources.** That's a
hard blocker, not a preference — Render's free tier needs no card at all.
The real cost of that choice is memory: Render's free web service instance
caps at **512MB RAM**, tight for `sentence-transformers` + `torch` + a
loaded FAISS index alongside FastAPI. See "Memory" below for how that's
handled.

### The constraint that shapes everything below: an ephemeral filesystem

Render's free tier wipes the entire local filesystem on every spin-down —
not just on redeploy. A free service spins down after **15 minutes of no
traffic** and spinning back up is a full rebuild, not a resumed container
(confirmed against Render's own docs). Persistent disks exist on Render,
but only on paid tiers.

Practically: anything uploaded through `/upload` — and the FAISS index it
builds — is gone the next time the service goes idle for 15 minutes. This
isn't a corruption risk (unlike the concurrent-writer problem
`threading.Lock` in `faiss_vector_store.py` already closes) — it's that the
app's core "upload once, ask questions later" loop doesn't survive a normal
idle gap on this tier at all.

**Handled by auto-seeding a bundled demo document on cold start.**
`backend/demo_corpus/pmp_key_concepts.pdf` is baked into the image;
`app/services/demo_seed_service.py`, wired into `main.py`'s FastAPI
`lifespan` handler, ingests it automatically on startup *only if the vector
store is empty* — a genuine cold start (Render) seeds fresh, a warm process
with real data (local dev, or any platform with real persistent storage)
is untouched. This means the live demo always has something to query
against out of the box. It does **not** solve persistence for a real
uploaded document — that still won't survive an idle gap on the free tier,
which is a known, documented limitation, not something silently papered
over.

### Memory

The embedding model is pre-downloaded into the image at build time (see
the Dockerfile's `RUN python -c "from sentence_transformers import
SentenceTransformer; SentenceTransformer(...)"` step) so a cold start
doesn't also pay for a Hugging Face download on top of loading it — that
step alone measured at roughly a minute over a slow/unauthenticated
connection locally.

`render.yaml` ships with **`RERANKING_ENABLED: false`**. Before the two
fixes below, `docker run --memory=512m` against the built image locally
peaked at ~497MB (97% of the 512MB limit) and survived — but **the first
real Render deploy OOM-killed anyway** ("Ran out of memory (used over
512MB)"), confirming the local Docker Desktop VM used to build/test this
doesn't reproduce Render's actual memory accounting closely enough to
trust a locally-passing test as sufficient on its own.

Two fixes closed most of that gap, verified with a second round of the
same local test:

- **Thread pools constrained** (`Dockerfile`:
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false`). torch/numpy/faiss default their
  BLAS/OMP thread pool to the *visible* CPU count, which in a container
  often reflects the host's full core count rather than the fraction
  actually allocated (Render's free tier: 0.1 CPU) — each extra thread
  allocates its own buffers, real memory, not just wasted scheduling.
- **Batched embedding encoding** (`Settings.embedding_batch_size`,
  default 8, `embedding_service.py`). Encoding an entire document's
  chunks in one `encode()` call was a real, measured transient memory
  spike; smaller batches trade a little throughput for a materially
  lower peak.

Together: peak dropped from ~500MB (97.7%) to **~480MB (93.8%)** —
real headroom gained, not just theorized, though still not a
comfortable margin. **Reranking stays off**: it loads a second model (a
cross-encoder) on top of the embedding model, and there's no headroom
in that peak to absorb it. `HYBRID_SEARCH_ENABLED` stays `true` — pure
CPU (BM25), no second model (see "Retrieval ablation" above). If a real
Render deploy still OOMs after these fixes, the next lever is
`EMBEDDING_BATCH_SIZE` lower than 8, then `HYBRID_SEARCH_ENABLED=false`
as a last resort before considering a paid tier.

### Deploying

No `gcloud`-equivalent CLI needed — Render deploys straight from a
connected GitHub repo:

1. Sign up at [render.com](https://render.com) (no card required for the
   free tier).
2. **New → Blueprint**, connect the GitHub repo. Render reads `render.yaml`
   at the repo root and provisions the service it describes.
3. During the Blueprint creation flow, Render prompts for every `envVar`
   marked `sync: false` in `render.yaml`: `GEMINI_API_KEY`, `API_KEY`,
   `GROQ_API_KEY`, and `FRONTEND_URL` (leave `FRONTEND_URL` as a placeholder
   until the frontend's real URL exists — step below — then update it in
   the dashboard; CORS will reject requests from the real frontend until
   this matches exactly).
4. Deploy. Render builds `backend/Dockerfile` and serves on the URL it
   assigns (`https://agrosense-rag-backend.onrender.com`-shaped).

### Deploying the frontend

Vercel or Cloudflare Pages, whichever's preferred — genuinely
interchangeable for a static Vite build, and neither needs a
platform-specific config file. Both auto-deploy on push via their own git
integration, no GitHub Actions step needed for this half of the app.
Build settings, identical on either platform:

- Build command: `npm run build`
- Output directory: `dist`
- Env vars (build-time — Vite inlines `VITE_*` at build time, not runtime,
  see `frontend/Dockerfile`'s existing comment on why): `VITE_API_BASE_URL`
  = the Render backend's URL. No API key needed — the web app authenticates
  via individual user login (JWT) at runtime, not a build-time secret.

Once the frontend has a real URL, go back to the Render dashboard and set
`FRONTEND_URL` to it — this is the CORS allowlist `main.py` checks against.

### Before making this reachable by anyone but you

`docs/OPERATIONS.md`'s A/B section above already documents hitting Gemini's
free-tier cap (20 requests/day) mid-eval-run — why `render.yaml` defaults
`LLM_PROVIDER` to `groq`. If a paid Gemini tier is available instead, switch
it back in the dashboard; either way this is a config decision already
supported today, not new engineering.

**This was deployed and live on Render/Vercel historically.** The pieces
described above (`Dockerfile`, `render.yaml`, `demo_seed_service.py`)
still work exactly as documented — `render.yaml` and the Render/Vercel
projects are what got removed, not the app's own portability. See
"Deploying to EC2" below for the current self-hosted path.

## Deploying to EC2 (Docker Compose)

The current self-hosted path, alongside Render (historical) above: a
single plain EC2 instance running the same `docker-compose.yml` already
used for local dev, plus two additive overlay files —
`docker-compose.prod.yml` (restart policies, Postgres no longer published
publicly) and `docker-compose.caddy.yml` (TLS reverse proxy — the
recommended default once a domain is pointed at the instance; see
"Security group" below for the no-domain fallback). No IaC for this
path — deliberately manual, since a single long-lived instance doesn't
carry the same case for it that a many-moving-pieces cloud deployment
would.

**Why this is genuinely simpler than the Render path, not just different:**
EC2's root volume is real, persistent disk — the single constraint that
shaped the Render section above ("The constraint that shapes everything
below: an ephemeral filesystem") simply doesn't exist here. The
named volumes in `docker-compose.yml` (`backend_uploads`,
`backend_vector_store`) sit on real disk across restarts and redeploys,
with no sync service, no read-only-filesystem workaround, and no
demo-seed-driven durability story. `demo_seed_service.py` still runs
harmlessly on first boot (empty vector store), same as always — it's just
no longer load-bearing for every restart the way it was on Render.

**What does *not* change:** this is still a single instance, so the same
single-process assumptions documented in `docs/DESIGN_REVIEW.md` Q9 still
apply — the in-memory session store and `FAISSVectorStore`'s
`threading.Lock` are correct only because there's exactly one backend
process. Docker Compose here never scales the backend service beyond
`replicas: 1` (the implicit default), so there's nothing to accidentally
turn on.

### Instance sizing

`docs/OPERATIONS.md`'s own measured numbers from the Render deploy apply
directly here: the backend alone (torch + sentence-transformers + a loaded
FAISS index) peaked at **~480MB** even after the thread-pool and
batch-size fixes. On this stack, Postgres, the frontend's nginx, and
(optionally) Caddy each add their own share on top of that — a **1GB
instance (t2.micro/t3.micro, free-tier eligible) is genuinely tight**,
not comfortable; the 2GB swapfile added in step 2 below exists
specifically to absorb bursts rather than let the OOM killer take the
backend down. **t3.small (2GB RAM)** is the more honest recommendation if
this needs to stay reliably up rather than free — same "state it plainly,
no asterisk" approach used elsewhere in this doc. `RERANKING_ENABLED=false`
(the default) still applies here for the same memory reason as Render.

### Security group

**Recommended (a domain is pointed at the instance): TLS via the Caddy
overlay.** Point an A record at the instance's Elastic IP before you
start (DNS propagation can take a few minutes to a few hours, so do this
first) — Let's Encrypt needs a real, publicly-resolvable hostname to
issue a certificate, which is why this can't just be turned on
unconditionally with no domain configured.

| Port | Source | Purpose |
|---|---|---|
| 22 | Your IP only | SSH |
| 80, 443 | 0.0.0.0/0 | Caddy — HTTP→HTTPS redirect and TLS termination, `docker-compose.caddy.yml` |

**Fallback (no domain yet, or genuinely don't want TLS): direct IP
access.** Traffic is plaintext HTTP in this mode — fine for a quick demo
or throwaway instance, not for anything handling real user data or
credentials over the network.

| Port | Source | Purpose |
|---|---|---|
| 22 | Your IP only | SSH |
| 8000 | 0.0.0.0/0 | Backend, direct IP access |
| 8080 | 0.0.0.0/0 | Frontend, direct IP access |

Never open 5432 (Postgres) or 8001 (LeafSense) in either mode —
`docker-compose.prod.yml` already stops publishing both (`!reset []`),
keeping them reachable only from other containers on the Compose
network. LeafSense is only ever called by AgroSense-RAG's own backend
(`http://leafsense:8001`, wired via `docker-compose.yml`'s
`VISION_SERVICE_URL` override — no manual `.env` edit needed for the
Compose path), never directly by a client, so it has no reason to be
internet-facing at all.

### Secrets

**Recommended: AWS SSM Parameter Store**, not plain `.env` files. Standard
`SecureString` parameters (the default AWS-managed KMS key, no custom CMK
needed) cost nothing extra — same free tier as any other Standard
parameter — so there's no cost trade-off for using it, only a few extra
setup steps. `app/core/config.py`'s `_load_secrets_from_ssm()` already
implements the resolution side: given `SECRETS_SSM_PREFIX`, it reads
`GEMINI_API_KEY`/`API_KEY`/`GROQ_API_KEY`/`DATABASE_URL`/`JWT_SECRET_KEY`
from `<prefix>/<lowercased-name>` as `SecureString` parameters into the
process environment before `Settings()` reads it, and never overwrites a
value that's already set directly — so this is purely additive to the
`.env` path below, not a replacement that breaks anything.

To use it:

1. Create the parameters (once, from your local machine or the instance,
   wherever you have AWS credentials configured):
   ```
   aws ssm put-parameter --name /agrosense/prod/gemini_api_key --type SecureString --value "<your key>"
   aws ssm put-parameter --name /agrosense/prod/api_key --type SecureString --value "<your key>"
   aws ssm put-parameter --name /agrosense/prod/jwt_secret_key --type SecureString --value "<a long random value>"
   # optional: groq_api_key, database_url, following the same pattern
   ```
2. On the instance, set `SECRETS_SSM_PREFIX=/agrosense/prod` in
   `backend/.env` (this one value is fine to keep in plaintext — it's a
   path, not a credential) instead of the real secret values.
3. Give the instance's IAM role (or its EC2 instance profile)
   `ssm:GetParameter` on that prefix, plus `kms:Decrypt` on the KMS key
   used to encrypt the parameters (`alias/aws/ssm`, the default, unless a
   custom key was used) — `SecureString` needs both, not just the SSM
   permission alone. Least-privilege, not the broad SSM/KMS access the
   AWS CLI session used to create the parameters needed.
4. Deploy as usual (step 6 below) — `_load_secrets_from_ssm()` runs
   before `Settings()` is ever constructed, so this needs no application
   code change, only the environment variable above.

**Fallback: plain `.env` files** (`backend/.env`, root `.env`) — still
fully supported, and the simpler choice if this isn't running on AWS or
the extra setup isn't worth it at this project's scale. The trade-off is
real either way it's used: secrets live in plaintext on disk unless SSM
(or an equivalent) is wired up. Steps 4-5 below show the plain-`.env`
version; swap in the SSM steps above wherever they diverge.

### First-time deploy

0. **Before launching anything**: if you're going the recommended
   TLS route, buy/choose a domain and point an A record at the Elastic IP
   you're about to attach in step 1 (you can attach the IP first, then
   update the DNS record — just start DNS propagating as early as
   possible, since it's the slowest step here). Skip this if you're
   deliberately using the no-domain fallback instead.
1. Launch an Ubuntu 22.04/24.04 EC2 instance (t3.small recommended, see
   above), attach an Elastic IP so the address survives a stop/start, and
   open the security group ports above (TLS mode: 22/80/443; fallback
   mode: 22/8000/8080).
2. SSH in and install Docker: `curl -fsSL https://get.docker.com | sudo sh`,
   then `sudo usermod -aG docker $USER` and re-login for it to take effect.
   Add a 2GB swapfile to absorb memory bursts (see "Instance sizing"
   above): `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile &&
   sudo mkswap /swapfile && sudo swapon /swapfile`, then append
   `/swapfile none swap sw 0 0` to `/etc/fstab` so it survives a reboot.
3. `git clone <repo-url> agrosense-rag && cd agrosense-rag`.
4. `cp backend/.env.example backend/.env` and set the required values —
   **recommended**: just `SECRETS_SSM_PREFIX` (see "Secrets" above, and
   create the actual parameters before this step); **fallback**: the real
   `GEMINI_API_KEY`/`API_KEY`/`GROQ_API_KEY`/`JWT_SECRET_KEY` values
   directly (same variables the Render deploy already needs, see
   `backend/.env.example`'s own comments). Optionally set `DATABASE_URL`
   to the Compose-network Postgres
   (`postgresql+psycopg2://insightai:insightai-dev-password@postgres:5432/insightai`
   — change that dev password for anything beyond a throwaway demo) for
   persisted users/RBAC/sessions instead of the in-memory fallback.
5. `cp .env.example .env` — leave `VITE_API_BASE_URL` unset in both
   modes. Frontend and backend are served from **one link** either way:
   nginx (`frontend/nginx.conf`) proxies `/api/*` to the backend
   container internally, so the browser only ever talks to one origin —
   `https://yourdomain.com` (TLS mode) or
   `http://<instance-public-ip>:8080` (fallback mode). Only set it if
   you're deliberately running frontend and backend on genuinely
   different hosts instead.
6. If using TLS: edit `Caddyfile`'s placeholder hostname (`example.com`)
   to your real domain first. Then:
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml -f
   docker-compose.caddy.yml up -d --build` (TLS mode — the recommended
   default), or
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   --build` (fallback mode, no Caddy overlay).
7. Verify: TLS mode — load `https://yourdomain.com` (frontend) and
   `https://yourdomain.com/api/health` (backend, through the same
   origin). Fallback mode — load `http://<instance-public-ip>:8080`
   (frontend) and `http://<instance-public-ip>:8080/api/health`
   (backend, same nginx proxy, no separate port needed). The backend's
   own direct port (8000) still works in fallback mode too, if you need
   to reach it without going through nginx.

### Ongoing deploys and rollback

`git pull`, then re-run the same `docker compose -f docker-compose.yml -f
docker-compose.prod.yml up -d --build` from step 6 (add
`-f docker-compose.caddy.yml` if that overlay is in use), then
`docker image prune -f` to clear dangling images/build cache left over
from the previous build (if disk fills up faster than expected,
`docker system df` shows where). No auto-deploy-on-push here (no
GitHub Actions SSH step exists) — a deliberate scope cut, same
"manual today, automatable later" shape the A/B eval workflow above
already has.

Rollback follows the same git-tag procedure as the "Rollback plan"
section above, run manually over SSH: `git checkout <tag>`, then the same
`docker compose ... up -d --build` command — the persistent EBS volume
means the vector store/uploads survive the rollback untouched, unlike a
rollback on the ephemeral Render deployment.

### Cost

t3.small: **~$15/month** on-demand (or free-tier-eligible t2.micro/t3.micro
for the first 12 months on a new AWS account, at the memory-tightness cost
above) + ~$1-2/month for a small EBS volume (20-30GB) + ~$3.60/month for an
unattached-safe Elastic IP (free while attached to a running instance).
Stated plainly, the same way the 512MB OOM incident above is stated
plainly — in exchange for no cold starts and real persistent storage.

## LeafSense Vision Service Deployment & Operations

AgroSense-RAG integrates with **LeafSense** (a dedicated deep learning service
serving a Hybrid CBAM + ViT + EfficientNetB0 plant pathology model across 38
disease classes).

### Architecture & Service Isolation

- **Microservice Separation**: LeafSense runs in its own dedicated Python virtual
  environment (`LeafSense/backend/.venv`) on **port 8001** to prevent TensorFlow /
  PyTorch / CUDA dependency and port collisions with AgroSense-RAG (port 8000).
- **Communication Protocol**: AgroSense-RAG communicates with LeafSense over
  HTTP via `app/services/vision_client.py`.
- **Health & Diagnostic Probes**: `GET /health/vision` on AgroSense-RAG actively
  monitors LeafSense availability with a 5-second TTL cache, surfacing real-time
  service status to the `/diagnose` frontend UI.

### Key Performance Optimizations

1. **Sub-Second Vision Inference**:
   - LeafSense's `backend/main.py` uses direct tensor graph execution
     `MODEL(img_batch, training=False).numpy()` rather than `MODEL.predict()`,
     eliminating dataset iterator and tracing overhead (~0.3s vs 3.5s+).
   - Startup dummy tensor execution warms up the Keras graph during service
     boot, guaranteeing zero cold-start delay for the first diagnosis query.
2. **Direct IPv4 Binding**:
   - `Settings.vision_service_url` defaults to `http://127.0.0.1:8001` (avoiding
     Windows / Linux IPv6 `::1` socket timeout delays).
3. **HTTP Keep-Alive & Connection Pooling**:
   - `vision_client.py` uses a persistent `httpx.Client` session pool with
     keep-alive enabled, eliminating TCP/TLS handshake latency on repeated scans.

### Local Startup

Run all services together from the repository root:
```powershell
.\start-local.ps1
```
Or start LeafSense independently:
- **Windows**: `cd LeafSense/backend && .\start.ps1`
- **Linux / macOS**: `cd LeafSense/backend && ./start.sh`

### Production Deployment & Systemd Service

When deploying LeafSense on a production VM / EC2 instance alongside AgroSense-RAG:

1. **Systemd Service (`/etc/systemd/system/leafsense.service`)**:
   ```ini
   [Unit]
   Description=LeafSense Vision Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/LeafSense/backend
   ExecStart=/home/ubuntu/LeafSense/backend/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
   Restart=always
   RestartSec=5
   Environment="PYTHONUNBUFFERED=1"

   [Install]
   WantedBy=multi-user.target
   ```
   Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now leafsense
   ```

2. **Configure AgroSense-RAG Environment**:
   In `backend/.env`:
   ```ini
   VISION_SERVICE_URL=http://127.0.0.1:8001
   VISION_SERVICE_TIMEOUT_SECONDS=15
   VISION_CONFIDENCE_THRESHOLD=0.5
   ```
   *(For multi-container Docker deployments, set `VISION_SERVICE_URL=http://leafsense:8001`)*.

3. **Verify Deployment Health**:
   ```bash
   # 1. Check LeafSense directly
   curl http://127.0.0.1:8001/model-info

   # 2. Check Vision connectivity through AgroSense-RAG
   curl http://127.0.0.1:8000/health/vision
   ```

