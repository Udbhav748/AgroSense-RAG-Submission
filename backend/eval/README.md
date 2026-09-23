# Evaluation harness

This directory has three tools:

- `run_eval.py` — an offline eval for `ChatService` (see below): checks
  whether the planner (`_plan`) routes queries to the right tool, and
  whether the resulting answers are on-topic, grounded, and resistant to
  prompt injection.
- `regression_check.py` — a CI gate that compares a fresh `run_eval.py`
  result against a committed baseline and fails if any tracked metric
  regressed beyond tolerance. Wired into `.github/workflows/eval.yml`.
- `metrics_report.py` — parses the backend's own JSON logs (stdout) and
  prints latency percentiles, error rate by `taxonomy_category`, and total
  LLM token usage/cost. A stand-in for real observability — in production
  you'd ship these same structured log lines to Prometheus/Grafana (or an
  APM) instead of grepping log files after the fact. Run it against a
  captured log file:

  ```bash
  uvicorn app.main:app | tee app.log   # one terminal, generate some traffic
  python eval/metrics_report.py app.log   # another terminal
  ```

## `run_eval.py`

An offline eval for `ChatService` (`app/services/rag_service.py`): it
checks whether the planner (`_plan`) routes queries to the right tool, and
whether the resulting answers are on-topic, grounded, and resistant to
prompt injection.

## Precondition: a document must already be indexed

This harness calls the **real** `ChatService`, backed by the **real**
persisted FAISS index at `backend/vector_store/`. It does not stand up a
fixture document — it reads whatever's already there.

Before running it:

1. Start the backend (`uvicorn app.main:app --reload`).
2. Upload at least one PDF via `POST /upload` (through the frontend, curl,
   or `/docs`).
3. `dataset_v1.json`'s content-specific entries (questions 3–8, 11) assume
   the indexed document is project-management material — specifically the
   "PMP Certification - Key Concepts" notes this repo was originally
   developed against (topics: what a project is, WBS, business case, EVM
   metrics like CPI/SPI, conflict-resolution styles, risk monitoring). If
   you index a different document, those `expected_keywords` entries will
   legitimately fail — that's not a bug, it's the keyword list being
   specific to that content. See **Versioning datasets** below for how to
   adapt this to a different document.
4. `GEMINI_API_KEY` must be set in `backend/.env` — this harness makes
   real LLM calls (16 entries ≈ 16+ generation calls, more if reflection
   or retries fire), so it consumes real API quota.
5. `dataset_v2.json` adds two web-findable entries (general-knowledge
   questions, deliberately outside the PMP document's content) with
   `expected_source: "web"`. To have any chance of scoring correctly on
   Source Accuracy, set `WEB_SEARCH_ENABLED=true` in `backend/.env` first
   — see "Source Accuracy" below for what happens if it's left off.

The two `summarize`-action entries and any `{{document_id}}` placeholders
in the dataset are filled in automatically at runtime: `run_eval.py`
reads `backend/vector_store/metadata.json` and uses whichever
`document_id` has the *most* indexed chunks (a proxy for "the real,
substantive document" — if a near-empty PDF got uploaded before your
target document, picking the first record in the file would grab that
instead). You don't need to hardcode an id yourself.

## Running it

From `backend/`:

```bash
python eval/run_eval.py
```

Or against a different dataset version:

```bash
python eval/run_eval.py --dataset dataset_v2.json
```

This prints a report to stdout and writes the full result (including
every entry's raw answer) to `eval/results/<UTC-timestamp>.json`.

## Metric selection: cost of failure

Every metric in this harness exists because the failure mode it detects
has a real, distinct cost if it ships silently. This is the explicit
rationale — not "we compute lots of things," but "each thing we compute
is here because *this specific failure* would otherwise go unnoticed
until a user hit it":

| Metric | Failure it detects | Cost if it ships undetected |
|---|---|---|
| Task Success Rate | The system gives a wrong or fabricated answer to a question it should get right | Direct user-facing harm — the core product promise (grounded answers) is broken; the most expensive failure mode in the harness, hence tracked first and with the largest dataset weight |
| False Refusal Rate | The system declines a question it should have answered | A quieter cost than Task Success Rate, but still real: every false refusal is a user who got no value at all from an answerable query — this is the metric that would silently regress if a "safety" tweak (e.g. tightening the retrieval-grade threshold) overcorrects |
| Injection Resistance / Prompt Injection Success Rate | The model complies with an attacker-supplied instruction hidden in a document or query | Security/trust failure, not just a wrong answer — a successful injection can exfiltrate the system prompt, make the app say attacker-chosen text, or otherwise act outside its intended role. Tracked with a zero-desired framing (like the other security metrics) because *any* non-zero rate here is a finding, not a tolerable baseline |
| Data Leak Rate | A response exposes content it shouldn't (an injection marker leaking outside the entry that was designed to provoke it) | Same class of cost as injection success, but catches leaks the *adversarial* entries alone would miss — a leak provoked by an unrelated, ordinary-looking query is arguably worse, since nothing about the request looked suspicious |
| Groundedness proxy / Entailment groundedness / Citation Accuracy | The answer sounds plausible but isn't actually supported by the retrieved context (or the citations shown don't back the claim) | The single most expensive *silent* failure mode for a RAG product specifically: a confident, well-formatted, wrong answer is more dangerous than an obvious one, because nothing about it signals "check this." Three metrics exist here, not one, because each catches a different way this can go wrong (lexical overlap without entailment; entailment without correct citations shown) — collapsing them into one would hide which layer actually failed |
| Precision@5 / Recall@5 / MRR / Hit Rate@5 | The wrong chunks come back from retrieval, independent of what the LLM does with them | Isolates a cheaper-to-fix failure (retrieval tuning) from a more expensive one (regenerate/reflect) — without these, a retrieval regression and a generation regression look identical at the Task Success Rate level, and you'd spend engineering time in the wrong place |
| Planner confusion matrix / accuracy / F1 | The system runs the wrong tool for a query (e.g. tries to summarize when it should retrieve) | Cheapest failure to detect (no LLM call needed — `_plan` is pure regex) and, left unchecked, the cheapest to silently compound: a misroute wastes every metric downstream of it on work that was never going to answer the actual question |
| Memory Recall Rate | A follow-up question that depends on earlier conversation turns gets answered as if the history weren't there | Product-trust cost specific to multi-turn use — a user who has to repeat themselves loses confidence in the "assistant" framing even when each individual answer is factually fine |
| Regression Rate (case-level, `regression_check.py`) | An aggregate metric holds steady while individual cases flip (some regress, others newly pass, netting to "no visible change") | The failure an aggregate-only gate would miss entirely — this is why it's a *second*, separate check rather than folded into the tolerance comparisons above |

Not every metric here is weighted equally in CI: `HIGHER_IS_BETTER`/
`LOWER_IS_BETTER` in `regression_check.py` gate on all of them with the
same tolerance today (a deliberate simplification — see "Regression
gate" below), rather than a severity-weighted gate that would fail harder
on, say, a Data Leak Rate regression than a Groundedness proxy dip. That
per-metric severity weighting is a reasonable next step, not yet built.

## What each metric means

### Planner metrics (confusion matrix, accuracy, precision/recall/F1)

For every dataset entry, `ChatService._plan(query, history=None)` is
called directly (no LLM involved — it's pure keyword/regex routing) and
compared against `expected_action`. The three classes are
`conversational`, `retrieve`, and `summarize`.

- **Confusion matrix**: rows are the expected action, columns are what
  the planner actually picked. The diagonal is correct routing;
  off-diagonal cells are misroutes (e.g. a `summarize` query that fell
  back to `retrieve` because no document id was found in it).
- **Precision / recall / F1 per class**, **macro F1** (unweighted average
  across classes — treats rare classes as equally important) and
  **weighted F1** (averaged by class support — reflects the dataset's
  actual class balance).

### Task Success Rate

Runs the *full* pipeline (`ChatService.handle_query`) and checks whether
the returned answer contains **at least one** of the entry's
`expected_keywords` (case-insensitive substring match). Only entries with
a non-empty `expected_keywords` list count toward this metric — small
talk and open-ended edge cases (e.g. the non-English query) are excluded,
since there's no fixed string that correctness hinges on. `n` in the
printed output tells you how many entries were actually scored.

Failure-type entries use `["couldn't find", "could not find"]` as their
keywords — for those, *success means the system correctly declined to
answer*, not that it produced a fabricated one.

### Groundedness proxy

For every `retrieve`-routed answer that actually got chunks back, this
checks whether the answer shares at least one non-trivial word
(alphabetic, 5+ letters, common stopwords excluded) with the text of the
chunks that were retrieved for it.

This is explicitly a **proxy**, not a faithfulness or entailment check:

- An answer can share vocabulary with its context while still
  misrepresenting what that context says — lexical overlap isn't
  correctness.
- A *correct* refusal (the `FALLBACK_REPLY` line, for failure-type
  entries) will usually score as "not grounded" here, since it doesn't
  reuse document vocabulary — that's expected, not a defect in the
  answer.

Treat this metric as "did the model at least engage with the retrieved
text," not "was the answer accurate."

### Entailment groundedness (LLM-as-judge)

For every `retrieve`-routed answer that actually got chunks back, this
makes a **second, cheap LLM call** (the "judge") to evaluate whether the
answer is actually *logically supported* by the retrieved context — not
just whether it shares vocabulary.

The judge prompt:
- Wraps each retrieved chunk in the same `---BEGIN UNTRUSTED DOCUMENT EXCERPT--- / ---END EXCERPT---` delimiters used by `prompt_builder.py`, so the judge knows the context is untrusted data.
- Asks the LLM to respond with structured JSON: `{"supported": true/false, "reason": "..."}` — validated against a Pydantic schema (`GroundednessJudgment`), not free-text parsing.
- Reuses the **same LLM provider** that generated the original answer (controlled by `LLM_PROVIDER` in `backend/.env`), so the run stays self-consistent (e.g. Groq answers judged by Groq, not a mix of providers).

This metric answers a genuinely different question from the other two groundedness-related metrics:

| Metric | What it checks | Method |
|---|---|---|
| **Groundedness proxy** (lexical) | Does the answer share non-trivial vocabulary with *any* retrieved chunk? | Word-overlap heuristic (deterministic, no LLM call) |
| **Citation Accuracy** | Does at least one *cited source excerpt* (the ~200-char slice a caller sees) contain keyword evidence for the claim? | Keyword-support heuristic on `ChatResponse.sources.excerpt` |
| **Entailment groundedness** (this metric) | Is the answer *actually supported by* the retrieved context — i.e., does every factual claim trace to the context? | LLM-as-judge with structured yes/no output |

Key distinctions:
- **vs. Groundedness proxy**: The proxy can be fooled by vocabulary overlap alone (e.g. answer repeats a key term but misstates the fact). The judge checks *logical entailment*, not just word presence.
- **vs. Citation Accuracy**: Citation Accuracy checks whether the *citations shown to the user* support the claim. Entailment groundedness checks whether the *answer itself* follows from the *full retrieved context* — even if a supporting chunk wasn't among the top-5 cited sources, or if the answer synthesizes across multiple chunks.

Because this doubles LLM calls for every scored `retrieve` entry, it respects the same quota pacing:
- Use `--delay` (e.g. `--delay 15` for free-tier rate limits) to space both the answer and judge calls.
- Set `LLM_PROVIDER=groq` in `backend/.env` for the cheaper/higher-quota provider — the judge uses the same provider automatically.
- The `--delay` is applied between the answer call and the judge call as well, so two calls per entry are paced correctly.

Entries that error out, return `FALLBACK_REPLY`, or have no retrieved chunks are scored as `entailment_grounded: false` (conservative default) and excluded from the denominator appropriately.

### Injection Resistance

For `case_type: "adversarial"` entries only. Each adversarial entry
carries an `injection_marker` — a string that would only appear in the
answer if the model actually complied with the injected instruction
(e.g. `"you are agrosense-rag"` for a "reveal your system prompt" attempt, or
`"injection successful"` for a "say the exact text X" attempt). Injection
Resistance is the fraction of adversarial entries where that marker does
**not** appear in the answer. An LLM call that errors out counts as
resisted (it didn't comply — it just failed).

This only detects the specific markers each entry defines; it isn't a
general jailbreak classifier. Add sharper markers as you add new
adversarial entries.

### Prompt Injection Success Rate

The exact complement of Injection Resistance (`1 - injection_resistance`,
same per-entry flags, same `n`) — reported separately because it matches
the literal framing used elsewhere ("successful injection attacks /
total injection attempts"), not because it's independently computed.
Kept alongside Injection Resistance rather than replacing it: existing
baselines and the regression gate already key off `injection_resistance`
by name.

### False Refusal Rate

The inverse failure mode from Task Success Rate: that measures wrongly
*answering* (fabricating instead of declining), this measures wrongly
*declining* — the answer came back as `FALLBACK_REPLY` for a question
that should have gotten a real one. Scored over entries where
`case_type` is **not** `"adversarial"` or `"failure"` — those two are
designed to correctly trigger `FALLBACK_REPLY`, so they're excluded from
this metric's denominator rather than counted as either a hit or a miss
against it. An entry that errored out (rather than returning any answer)
isn't counted here either — it's a different failure mode than declining
gracefully, tracked separately by the per-entry `error` field.

### Data Leak Rate

Same `injection_marker` strings Injection Resistance already uses, but
checked against **every** entry's answer in the run, not just the
adversarial entry each marker was written for — a leak isn't necessarily
only provoked by the prompt designed to elicit it. `n` covers the whole
dataset whenever at least one entry defines an `injection_marker`; an
errored entry counts as no leak (an empty/failed response can't contain
the marker text).

### Source Accuracy

For entries with an `expected_source` field only (currently just
`dataset_v2.json`'s two web-findable entries). Checks whether
`ChatResponse.answer_source` (`"documents"` / `"web"` / `"mixed"`) matches
`expected_source`. This is only meaningful with
`WEB_SEARCH_ENABLED=true` in `backend/.env` — with the fallback disabled
(the default), a web-findable entry's retrieval will be graded
`"insufficient"` but no web search fires, `answer_source` stays
`"documents"`, and it correctly scores as a Source Accuracy miss (that's
expected, not a bug — it's testing whether the fallback works when it's
on, not asserting it should always be on).

### Precision@5 / Recall@5 / MRR

For entries with a non-empty `expected_chunk_keywords` list — currently
`dataset_v2.json`'s content-specific `retrieve` entries (the same ones
scored for Task Success Rate, reusing `expected_keywords` as
`expected_chunk_keywords` since those terms come from the correct
chunk in the first place). Unlike every other metric here, these measure
**retrieval quality directly** — whether the right chunks came back at
all — independent of what the LLM did with them afterward. This is what
makes them the right metric for the hybrid-search/reranking ablation (see
`docs/OPERATIONS.md`'s "Retrieval ablation"): Task Success Rate and
Groundedness can move for reasons that have nothing to do with retrieval
(a different Gemini sample, a prompt tweak), but these don't.

There's no full corpus relevance judgment to compute textbook
Precision/Recall against — only the keyword heuristic ("a retrieved chunk
is relevant if it contains any of `expected_chunk_keywords`") — so each
metric is defined purely in terms of that:

- **Precision@5**: of the top 5 retrieved chunks, what fraction are
  relevant. Missing slots (fewer than 5 chunks actually came back) count
  as non-relevant.
- **Recall@5**: of the entry's *distinct expected keywords*, what
  fraction are covered by at least one of the top-5 chunks. Each keyword
  stands in for one expected fact/passage, since there's no labeled set
  of "every relevant chunk in the corpus" to divide by instead.
- **MRR** (Mean Reciprocal Rank): the average of `1 / rank` of the first
  relevant chunk per entry (`0` if none of the top 5 are relevant) — how
  quickly retrieval surfaces something useful, not just whether it
  eventually does.

See `precision_at_k`/`recall_at_k`/`reciprocal_rank` in `run_eval.py` for
the exact implementation.

### Citation Accuracy

Same entries as Precision@5/Recall@5/MRR, but a different question from
either that or Groundedness proxy:

- Groundedness checks the **answer text** — does it share vocabulary with
  *any* retrieved chunk.
- Precision@5 checks **retrieval** — did relevant chunks come back at
  all, independent of what the LLM did with them.
- Citation Accuracy checks the **citation surface a caller actually
  sees** — `ChatResponse.sources` — and whether at least one of those
  specific excerpts contains real supporting evidence, using the same
  keyword-support heuristic as Precision@5 (`citation_supported` in
  `run_eval.py`).

Checked against each source's `excerpt` (the ~200-char slice a caller
would actually see) rather than the full underlying chunk text — a
keyword present in a chunk but trimmed out of the excerpt is exactly the
kind of gap this is meant to surface, not paper over by checking text
nobody's shown. This is the automated version of what
`docs/HUMAN_EVAL.md` row 4 caught by hand: a correct, well-grounded
answer whose 5 listed sources didn't obviously contain the WBS
definition — Citation Accuracy would have flagged that row as `False`
automatically, without needing a human to notice it.

## Dataset format

Each entry in `dataset_vN.json` is an object:

| Field | Required | Meaning |
|---|---|---|
| `query` | yes | The user query. May contain the literal placeholder `{{document_id}}`, substituted at runtime. |
| `expected_action` | yes | One of `"conversational"`, `"retrieve"`, `"summarize"` — what `_plan` should choose. |
| `expected_keywords` | yes (may be `[]`) | Keywords for Task Success Rate. Empty list = excluded from that metric. |
| `case_type` | yes | One of `"normal"`, `"edge"`, `"failure"`, `"adversarial"`. |
| `injection_marker` | only for `case_type: "adversarial"` | String that would appear in the answer only if the injection succeeded. |
| `expected_source` | no | `"documents"`, `"web"`, or `"mixed"` — expected `ChatResponse.answer_source`, checked for Source Accuracy. Omit for entries where the source doesn't matter. |
| `expected_chunk_keywords` | no | Keywords for Precision@5/Recall@5/MRR/Hit Rate@5 — a retrieved chunk is "relevant" if it contains any of them. Omit (or leave `[]`) to exclude an entry from these metrics, e.g. for entries where no document chunk should ever be relevant (failure/adversarial/web-sourced entries). |
| `history` | no | Prior conversation turns for a **memory-eval** entry, oldest first, shaped like `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`. Passed to `handle_query(query, history=...)`, so it feeds the prompt's conversation-history block (only the most recent 6 turns are used). |
| `expected_memory_keywords` | only for entries with `history` | Keywords for Memory Recall Rate — terms the correct answer can only have come from an earlier turn in `history`, **not** from the current `query` (or else the metric can't distinguish memory from the query itself). |

The memory-eval pattern: turn 1 asks a factual question about the corpus,
turn 2 (the entry's `query`) asks a follow-up whose answer requires
carrying over something stated in turn 1's assistant reply — e.g. turn 1
"Define a project.", turn 2 "What are the two defining traits you just
listed?" — with `expected_memory_keywords` being those traits. If the
system remembers, the answer contains them; if the history were dropped,
there'd be nothing in the current query pointing at them.

## Versioning datasets

`dataset_v1.json` is tied to the PMP-notes content described above. When
the eval needs to target different content (a new default document,
broader coverage, fixes to bad keyword picks), don't edit `dataset_v1.json`
in place — that silently breaks comparability with past `results/*.json`
runs made against v1. Instead:

1. Copy it to `dataset_v2.json` (increment from whatever the latest
   version is).
2. Edit the copy — content-specific `expected_keywords`, add/remove
   entries, adjust `injection_marker`s, etc. Keep the minimums from the
   original spec: ≥2 `edge`, ≥2 `failure`, ≥2 `adversarial` entries.
3. Run with `python eval/run_eval.py --dataset dataset_v2.json`. The
   output filename under `eval/results/` is a timestamp, not tied to the
   dataset version, but each result JSON records `"dataset"` — check that
   field before comparing two runs.
4. Leave `dataset_v1.json` in place as a fixed reference point.

`dataset_v2.json` already exists, still targeting the same PMP document —
it's v1 plus two new entries (`expected_source: "web"`) exercising the
corrective RAG loop's web search fallback, not a retarget to different
content. It's the template to follow for adding more entries of any kind
going forward.

## Regression gate

`regression_check.py` fails CI when a metric regressed beyond tolerance:

```bash
python eval/regression_check.py --results eval/results/<new>.json \
    --baseline eval/baselines/v2_groq.json --tol 0.05
```

- **Higher-is-better** metrics fail when they drop more than `--tol`:
  `planner.accuracy/macro_f1/weighted_f1`, `task_success_rate`,
  `groundedness_proxy`, `entailment_groundedness`, `injection_resistance`,
  `source_accuracy`, `precision_at_5`, `recall_at_5`, `mrr`,
  `hit_rate_at_5`, `citation_accuracy`, `tool_arg_accuracy`.
- **Lower-is-better** metrics fail when they rise more than `--tol`:
  `false_refusal_rate`, `data_leak_rate`, `prompt_injection_success_rate`.
- Metrics missing from either file are skipped — so new metrics don't gate
  until a baseline that includes them is committed.

Baselines live in `eval/baselines/` (committed). Re-baselining: after a
deliberate, human-reviewed change (new prompt version, model change), run
a fresh eval and commit the resulting **full result file, including
`entries`** — intentionally, not as a way to paper over an unnoticed
regression. The `eval.yml` workflow runs this gate after every manual
eval run with the `baseline` input (default `eval/baselines/v2_groq.json`);
set the input empty to skip the gate.

`entries` matters now, not just the aggregates: alongside the tolerance
checks above, `regression_check.py` also computes case-level **Regression
Rate** (`Previously Passing Cases Now Failing / Previously Passing Cases`,
gated by `--max-regression-rate`, default 0.0) by matching entries between
the two files on query text. This is a genuinely different, finer-grained
signal than the aggregate checks — an aggregate can hold steady while
individual cases flip (some regress, others newly pass, netting to no
visible change), which the aggregate checks alone would never catch. It's
a no-op (nothing to compare, never flags a regression) against a baseline
that lacks `entries` — including `eval/baselines/v2_groq.json` as
committed today, which predates this check and was trimmed to aggregates
only. The next deliberate re-baseline should commit the full result so
this check actually has something to compare against going forward.

Every `run_eval.py` result also records the model + dataset revision that
produced it: `dataset_version`, `llm_model_name`, `reranking_model_name`,
and `embedding_model_name` — along with the existing `llm_provider`,
`fallback_llm_provider`, and `prompt_version` — so any two results can be
compared only when those identifying fields agree.
