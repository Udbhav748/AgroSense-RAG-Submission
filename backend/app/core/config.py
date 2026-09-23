"""Centralized application settings, loaded from environment variables / .env file.

See backend/.env.example for a description of each variable.
"""

import contextlib
import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Module 10 gap-closure: known placeholder/example secret values that show
# up in .env.example, READMEs, or tutorials -- if any of these are still
# in place on a DEBUG=false (production) run, that's a real misconfiguration
# worth failing loudly on rather than silently running insecure. Not an
# exhaustive deny-list (no such list could be), just the specific values
# this project's own docs/.env.example actually suggest.
_KNOWN_WEAK_SECRETS = {
    "local-dev-api-key-change-me",
    "change-me",
    "changeme",
    "change_me",
    "secret",
    "password",
    "test",
    "your-api-key-here",
    "your-secret-key-here",
    "insightai-dev-password",
    "",
}


def _load_secrets_from_ssm() -> None:
    """If SECRETS_SSM_PREFIX is set (only relevant on a deployment that
    resolves secrets from AWS SSM Parameter Store), resolve
    GEMINI_API_KEY/API_KEY/GROQ_API_KEY/DATABASE_URL/JWT_SECRET_KEY from
    SSM SecureString parameters under that prefix into the process
    environment, before Settings() below reads it. A no-op everywhere else
    (local dev, docker-compose, EC2, Render): SECRETS_SSM_PREFIX is never
    set there, so this returns immediately without importing boto3 or
    making any AWS call.

    Never overwrites a value that's already set directly (e.g. a local
    .env's GEMINI_API_KEY) — SSM only fills in what's actually missing.
    A parameter that doesn't exist under the prefix (e.g. DATABASE_URL
    when no DSN was configured) is silently skipped, not an error —
    Settings.database_url's own default ("") applies exactly as it does
    everywhere else.
    """
    ssm_prefix = os.environ.get("SECRETS_SSM_PREFIX")
    if not ssm_prefix:
        return

    import boto3

    client = boto3.client("ssm")
    for env_name in ("GEMINI_API_KEY", "API_KEY", "GROQ_API_KEY", "DATABASE_URL", "JWT_SECRET_KEY"):
        if os.environ.get(env_name):
            continue
        parameter_name = f"{ssm_prefix}/{env_name.lower()}"
        try:
            response = client.get_parameter(Name=parameter_name, WithDecryption=True)
        except client.exceptions.ParameterNotFound:
            continue
        os.environ[env_name] = response["Parameter"]["Value"]


def _sanitize_hf_cache_dirs() -> None:
    """Ensure HF_HOME, TRANSFORMERS_CACHE, and TORCH_HOME point to valid, existing local paths."""
    safe_cache = Path.home() / ".cache" / "huggingface"
    with contextlib.suppress(Exception):
        safe_cache.mkdir(parents=True, exist_ok=True)

    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        val = os.environ.get(key)
        if val:
            try:
                p = Path(val)
                drive = p.drive
                if drive and not Path(drive + "/").exists():
                    os.environ[key] = str(safe_cache)
                else:
                    p.mkdir(parents=True, exist_ok=True)
            except Exception:
                os.environ[key] = str(safe_cache)
        else:
            os.environ[key] = str(safe_cache)


_load_secrets_from_ssm()
_sanitize_hf_cache_dirs()


class Settings(BaseSettings):
    # Optional. PostgreSQL connection string (e.g.
    # postgresql+psycopg2://user:pass@host:5432/db). If set, the app uses
    # PostgreSQL-backed persistence (documents, sessions, API keys, usage
    # logs) via SQLAlchemy/Alembic — see app/core/database.py. If left
    # empty, the app falls back to the legacy in-memory/file stores (an
    # ephemeral deployment like Render's free tier that can't host a real
    # DB keeps working exactly as before).
    database_url: str = ""

    # Optional. Redis connection string for services/cache_service.py
    # (CacheService — not currently wired into any route/service; this
    # field just lets it construct without raising when someone does wire
    # it in). Empty disables caching, matching CacheService's own
    # try/except-degrade-to-no-cache behavior.
    redis_url: str = ""

    # Timeout, in seconds, for the startup database connection attempt
    # (passed as connect_args["connect_timeout"] to create_engine in
    # app/core/database.py). When Postgres is unreachable (Docker Desktop
    # not running, the insightai-postgres container down), startup used to
    # hang on an unbounded OS-level connect attempt; this bounds it so
    # init_db() fails fast with an actionable error. Matches the per-service
    # timeout-field convention used across this file.
    database_connect_timeout_seconds: int = 5

    # Required. Google Gemini API key (used by the future RAG/generation pipeline).
    gemini_api_key: str

    # Which LLM provider generate() calls go to by default: "gemini" or
    # "groq". Both client classes implement the same LLMClient interface,
    # so switching this is the entire manual A/B-testing surface (see
    # model_routing_enabled below for automatic, per-request routing).
    llm_provider: str = "gemini"

    # Optional. If set to a different provider than llm_provider, that
    # provider is used as an automatic fallback: after the primary
    # provider's own retries are exhausted (LLMTimeoutError/LLMAPIError),
    # FallbackLLMClient retries once against this provider instead of
    # failing the request outright. Leave unset (None) to disable
    # fallback and fail after the primary's retries.
    fallback_llm_provider: str | None = None

    # When True, build_llm_client() (app/services/llm_provider.py) wraps
    # the primary client in a RoutingLLMClient that picks, per request,
    # between llm_provider (the default, "simple" path) and
    # model_routing_complex_provider (below) based on complexity/risk
    # signals already present in the built prompt — see
    # app/services/routing_llm_client.py. No new classification call: the
    # signals (has the corrective loop already retried once? is the
    # prompt unusually large?) are read straight off what the RAG
    # pipeline already assembled. Off by default — same reasoning as
    # every other routing/approval flag in this file: a deployment
    # policy, not assumed behavior.
    model_routing_enabled: bool = False

    # The provider used for requests RoutingLLMClient judges "complex"
    # when model_routing_enabled is True. Meant to be set to your
    # stronger/costlier model while llm_provider stays your cheap/fast
    # default — e.g. LLM_PROVIDER=groq + MODEL_ROUTING_COMPLEX_PROVIDER
    # left at this default. If this equals llm_provider, routing is a
    # deliberate no-op (nothing to route to) — build_llm_client() detects
    # that and skips wrapping, the same early-return shape
    # fallback_llm_provider already uses above.
    model_routing_complex_provider: str = "gemini"

    # Prompt length (characters), above which RoutingLLMClient treats a
    # request as "complex" even without a corrective-loop retry — a
    # proxy for both reasoning difficulty (more context to synthesize)
    # and cost (more tokens billed either way). A single-chunk answer
    # (one ~1000-char excerpt, chunk_size) plus instructions/query sits
    # nowhere near this; it takes several chunks, or web results stacked
    # alongside document chunks, to cross it — configurable per
    # deployment since "complex" is inherently a judgment call, not a
    # fixed property of this app.
    model_routing_complex_prompt_chars: int = 6000

    # Groq API key. Only required if llm_provider or fallback_llm_provider
    # is "groq".
    groq_api_key: str = ""

    # Groq model used for text generation.
    groq_model_name: str = "llama-3.3-70b-versatile"

    # Timeout, in seconds, for Groq API calls.
    groq_timeout_seconds: int = 30

    # Estimated USD cost per 1,000 tokens for Groq, used the same way as
    # cost_per_1k_tokens below but with Groq's (much cheaper) pricing.
    groq_cost_per_1k_tokens: float = 0.0006

    # Required. Shared secret clients must send in the X-API-Key header to
    # reach the documents/query routers. See core/auth.py.
    api_key: str

    # Optional. JSON string mapping client identifiers to their API keys.
    # Example: '{"client-a": "key1", "client-b": "key2"}'
    # If provided, this supersedes api_key and enables per-client auth.
    # Keys are hashed at startup (SHA-256); only hashes are kept in memory.
    api_keys: str = ""

    # Optional. Comma-separated client names (the same names used as keys
    # in API_KEYS, or "default" for the single-api_key fallback) granted
    # the "admin" role — see app/services/tenant_service.py's
    # resolve_tenant(). Checked live on every request rather than baked
    # into the DB at tenant-creation time, so promoting/demoting a client
    # is just an env var change and a redeploy, not a migration or manual
    # SQL. Only takes effect when DATABASE_URL is set — without a DB there
    # is nowhere to persist per-tenant role, so role-gated actions (e.g.
    # DELETE /documents/{id}) fall back to their pre-RBAC, ungated
    # behavior, same as tenant-scoping already does.
    admin_client_names: str = ""

    # Optional. When set, GET /metrics requires this token via the
    # Authorization: Bearer header (compared constant-time; see
    # app/api/v1/routes/metrics.py). Left empty (the default), /metrics is
    # unauthenticated like /health — metrics carry no payload data and the
    # route cost is trivial, but a publicly-exposed deployment can tighten
    # this without handing out a full API key to its scrapers.
    metrics_bearer_token: str = ""

    # --- Individual user login (JWT), alongside API-key auth ----------
    # Secret used to sign/verify JWTs issued by POST /auth/login and
    # POST /auth/signup (see app/api/v1/routes/auth.py). Resolved through
    # _load_secrets_from_ssm above on the Lambda deployment, exactly like
    # GEMINI_API_KEY/DATABASE_URL — never a separate secret-delivery
    # mechanism. Empty by default: the auth routes raise a clear
    # LLMConfigurationError-style config error if a token is requested
    # while this is unset, rather than signing with an empty/weak secret
    # silently (same "fail loud, not silently insecure" posture
    # groq_api_key's own missing-key check already uses).
    jwt_secret_key: str = ""

    # Signing algorithm for the JWT above. HS256 (symmetric) needs no key
    # pair to manage, appropriate for a single backend that both issues
    # and verifies its own tokens.
    jwt_algorithm: str = "HS256"

    # How long an issued JWT stays valid, in minutes. Default 24h — long
    # enough that a web session survives a normal browsing gap without
    # needing a refresh-token flow, which this app deliberately doesn't
    # have yet. Revocation for the full 24h window is handled separately
    # by POST /auth/logout (see user_service.revoke_all_tokens) rather
    # than by shortening this, so a token stays convenient to use but is
    # still killable on demand.
    jwt_expiry_minutes: int = 1440

    # Max failed login attempts for one email within
    # login_lockout_window_seconds before further attempts are rejected
    # outright (AccountLockedError) — checked before the password is even
    # compared. Keyed by the *account* being attacked
    # (user_service.py's _is_locked_out), not the caller's IP, so it
    # closes the gap the IP-only limiters (core/auth.py's per-identity
    # limiter, core/security.py's per-IP one) leave open: an attacker who
    # rotates IPs can otherwise brute-force one account indefinitely as
    # long as each individual IP stays under its own limit.
    login_lockout_max_attempts: int = 5

    # Sliding window (seconds) the attempt count above is measured over.
    # A locked-out account becomes attemptable again as its oldest
    # recorded failure ages out of this window — no separate unlock step
    # needed. 15 minutes.
    login_lockout_window_seconds: int = 900

    # Per-IP rate limit for unauthenticated endpoints (auth/signup,
    # auth/login — the brute-force surface), and the per-identity limit
    # enforced in core/auth.py for authenticated paths (API keys and JWT
    # users alike). Enforced in-process (a sliding window per identity),
    # not persisted — resets on restart, acceptable for a free-tier demo.
    rate_limit_per_minute: int = 60
    rate_limit_window_seconds: int = 60

    # When True, the per-IP limiter is applied to the unauthenticated
    # auth routes (login/signup) via security.py's dependency. The
    # per-identity limiter in core/auth.py is always on for authenticated
    # paths — this flag only controls the IP-based brute-force guard on
    # the routes that have no identity to key on yet.
    rate_limit_enabled: bool = True

    # Optional. Comma-separated IPs of reverse proxies this app trusts to
    # set X-Forwarded-For honestly (e.g. the Caddy/nginx/load-balancer
    # hop directly in front of this process — see Caddyfile). security.py's
    # get_client_ip() only reads X-Forwarded-For when request.client.host
    # (the actual TCP peer) is in this set; otherwise it uses that peer
    # address directly. Without this, X-Forwarded-For is attacker-
    # controlled on any request that reaches this process directly (or
    # through an untrusted intermediary), letting a client rotate a fake
    # header value to defeat the per-IP brute-force limiter on
    # /auth/login and /auth/signup. Empty by default (no proxy trusted,
    # X-Forwarded-For ignored) — set this explicitly once a real reverse
    # proxy sits in front of the app.
    trusted_proxy_ips: str = ""

    # Origin of the frontend app; used to configure CORS.
    frontend_url: str = "http://localhost:5173"

    # Application metadata, surfaced in the FastAPI docs.
    app_name: str = "AgroSense-RAG"
    app_version: str = "0.1.0"

    # Enables debug behavior (e.g. verbose errors).
    debug: bool = False

    # Directory (relative to backend/) where uploaded files are stored.
    upload_dir_name: str = "uploads"

    # Directory (relative to backend/) where chat feedback events are appended.
    feedback_dir_name: str = "feedback"

    # Filename for the feedback JSONL file, inside feedback_dir_name.
    feedback_filename: str = "feedback.jsonl"

    # Maximum accepted upload size, in megabytes.
    max_upload_size_mb: int = 20

    # MIME types accepted by the upload endpoint.
    allowed_upload_mime_types: list[str] = ["application/pdf"]

    # Target size of each document chunk, in characters.
    chunk_size: int = 1000

    # Character overlap between consecutive chunks.
    chunk_overlap: int = 200

    # DPI used to rasterize a page for OCR (document_service.py), when the
    # page has no extractable text layer (scanned/image-only PDFs).
    # Higher improves OCR accuracy at the cost of extraction time; 200 is
    # tesseract's usual sweet spot.
    ocr_dpi: int = 200

    # Minimum characters PyMuPDF's own extraction must yield for a page to
    # be trusted as-is. Below this (including zero), OCR is attempted —
    # not just on truly blank pages. Real scans routinely carry a thin
    # native text layer (a header, a page number, a few garbled characters
    # from a prior bad OCR pass) that would otherwise pass the old
    # "any text at all" check and ship a near-useless chunk to the index.
    ocr_min_chars_per_page: int = 100

    # Sentence Transformers model used to generate chunk embeddings.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Chunks per encode() batch. Encoding a whole document's chunks in one
    # batch is fine on a normal machine, but on a memory-constrained
    # deployment (Render's free tier: 512MB) it's a real, measured
    # transient memory spike — confirmed against a live Render deploy that
    # OOM-killed on it. Smaller batches trade a little throughput for a
    # materially lower peak.
    embedding_batch_size: int = 8

    # Directory (relative to backend/) where the vector store is persisted.
    vector_store_dir_name: str = "vector_store"

    # Filename for the persisted FAISS index, inside vector_store_dir_name.
    vector_index_filename: str = "index.faiss"

    # Filename for the persisted chunk metadata (JSON), inside vector_store_dir_name.
    vector_metadata_filename: str = "metadata.json"

    # --- pgvector vector store (off by default) ---------------------------
    # When True (and DATABASE_URL is set — pgvector lives in Postgres, so
    # there's nowhere to store vectors without it), get_vector_store() in
    # query.py returns a PGVECTORVectorStore (services/pgvector_store.py)
    # instead of the FAISSVectorStore. Embeddings are stored in a
    # Postgres table with a pgvector column (see alembic migration 0006),
    # giving delete_document a native, scalable remove-by-id (no
    # index-rebuild), and letting the same tenant-scoped
    # WHERE tenant_id filtering used everywhere else apply directly to
    # vectors. FAISS remains the default and the migration path stays
    # opt-in: the pgvector table is created by Alembic regardless, but the
    # app only reads/writes it when this flag is on. Note: enabling it on a
    # deployment that already has a populated FAISS index starts with an
    # empty pgvector table — re-upload documents (or backfill the table)
    # before relying on it.
    pgvector_enabled: bool = False

    # Vector dimensions pgvector_store.py assumes for its embeddings table.
    # Must match the embedding model's output size (384 for the default
    # all-MiniLM-L6-v2); mismatch is caught at create_index() time and
    # reported as a clear error rather than a Postgres "wrong dimensions"
    # failure deep in a query.
    pgvector_dimensions: int = 384

    # The Postgres table pgvector_store.py stores vectors in. Single table
    # for every tenant, with tenant_id a column — not one table per
    # tenant — matching how the rest of the app's relational models do
    # multi-tenancy (see app/models/... tenant_id columns).
    pgvector_table_name: str = "document_embeddings"

    # Default number of chunks to return from retrieval.
    #
    # MODULE 10 FAITHFULNESS ROOT-CAUSE FIX (P2, 2026-09-20): was 5. Raised
    # to 8 after tracing a real Faithfulness-benchmark failure
    # (eval-potato-01: "What fungicides and bio-treatments effectively
    # manage potato early blight...") to a genuine retrieval-ranking miss,
    # not a generation defect -- the exact "Agricultural Treatment &
    # Dosage Reference: Potato - Early Blight" chunk (containing both the
    # organic remedy and chemical treatment the ground truth expects) was
    # confirmed present in the corpus and within the top-20 hybrid-search
    # candidate pool, but ranked #8, just outside the old top_k=5 cutoff --
    # verified directly against the live vector store, not assumed.
    # Enabling the existing cross-encoder reranking feature was tried
    # first and did NOT surface the chunk into the top-5 either (the
    # MS-MARCO-trained cross-encoder does not score this corpus's
    # pipe-delimited "Crop: X | Disease: Y | ..." dosage-table format as
    # highly relevant to a natural-language question -- a real, disclosed
    # limitation of that model on this corpus's formatting, not fixed
    # here). Widening the plain top-k window is the smaller, more
    # reliable fix: it directly and verifiably includes the missing chunk
    # for the failing case without depending on reranking's relevance
    # judgment for this corpus. See docs/MODULE10_RESULTS.md's
    # Faithfulness section for the full before/after measurement.
    retrieval_top_k: int = 8

    # When True, ChatService logs the exact prompt sent to the LLM on every
    # generation (see prompt_builder.build_prompt), not just its version —
    # closing the "prompt content not captured" debugging gap. Off by
    # default: prompts can embed retrieved document text, so capturing them
    # is a data-retention decision the deployment must make explicitly.
    log_prompt_content: bool = False

    # Upper bound on characters of the prompt logged when
    # log_prompt_content is True. Truncated with a marker, not silently
    # cut mid-word, so what's captured is unambiguously a prefix.
    log_prompt_max_chars: int = 2000

    # Default minimum similarity score a chunk must meet to be returned.
    # Chunks from a genuinely relevant match typically score ~0.45-0.55 with
    # all-MiniLM-L6-v2; 0.3 let weakly-related chunks through for off-topic/
    # conversational-ish queries that don't match any canned phrase in
    # rag_service.py, producing technically-grounded but irrelevant answers.
    retrieval_min_score: float = 0.4

    # Minimum top-chunk similarity score for retrieval to be graded "good"
    # (see ChatService._grade_retrieval). Between retrieval_min_score and
    # this threshold, retrieval is graded "weak" — chunks cleared the score
    # floor but aren't confidently on-topic, which is what triggers the web
    # search fallback (if enabled).
    retrieval_grade_threshold: float = 0.5

    # Enables the web-search fallback tool (services/web_search_service.py)
    # for "weak"/"insufficient" retrieval grades. Off by default so existing
    # behavior — and the lack of any outbound network call beyond Gemini —
    # is unchanged unless explicitly opted in.
    web_search_enabled: bool = False

    # Which provider web_search_service.search_web dispatches to. The
    # DuckDuckGo provider is keyless (duckduckgo-search SDK); Brave and
    # Bing require WEB_SEARCH_API_KEY. Off-by-default web_search_enabled is
    # the master switch; asking for a provider whose key isn't configured
    # force-disables the tool (degrades to "no results", never a crash)
    # with a distinct "web_search_unavailable" log event.
    web_search_provider: str = "duckduckgo"

    # API key for providers that require one (brave, bing). DuckDuckGo
    # ignores it. When web_search_provider is one of the keyed providers
    # and this is empty while web_search_enabled is True, the tool is
    # treated as disabled rather than erroring — "force-disable without
    # key", per Feature #7.
    web_search_api_key: str = ""

    # When True, the web-search tool is a human-in-the-loop action: it only
    # fires when the client explicitly sends confirm_web_search=true on the
    # /chat request (the "human approval" gate). Off by default — the
    # approval requirement is a deployment policy, not a behavior the app
    # should assume.
    web_search_requires_approval: bool = False

    # Same human-in-the-loop shape as web_search_requires_approval above,
    # applied to document deletion: when True, DELETE /documents/{id}
    # requires the caller to also send approved=true, in addition to
    # confirm=true. Distinct from both existing gates on that route:
    # confirm=true is mistake-prevention (did you mean to delete this at
    # all?) and the admin/member role check is access control (are you
    # allowed to delete anything?) — this is a deployment policy that can
    # require an explicit extra step before the action executes, on top
    # of either. Off by default, same reasoning as web search's flag: the
    # approval requirement is a deployment choice, not an assumed default.
    document_delete_requires_approval: bool = False

    # Enables JSON-mode structured output for the LLM: when True and the
    # client sends structured_response=true, ChatService asks the provider
    # for a strict JSON answer (via response_mime_type / response_format)
    # and validates it against the StructuredAnswer schema, falling back to
    # plain text if the model's output doesn't parse.
    #
    # Module 10 gap-closure (2026-09-21): this is now True by default. The
    # standard free-text path is still the default per-request behavior --
    # structured mode only activates when a caller explicitly opts in via
    # ChatRequest.structured_response=true (POST /chat only; /chat/stream,
    # /chat/diagnose(/stream) never request it, since token-by-token SSE
    # and vision-diagnosis text are intentionally free-form contracts, not
    # structured ones). This flag being True just means the already-built,
    # already-tested opt-in path is actually reachable in production
    # instead of silently dead regardless of what a caller requests.
    structured_output_enabled: bool = True

    # --- Multi-agent features (off by default) -------------------------
    # When True, intent classification uses an LLM router agent
    # (services/router_agent.py) instead of the deterministic keyword
    # planner — it can route a query to "research" (multi-step web
    # research) that the regex planner can't recognize. Deterministic
    # fast paths (small talk, "summarize <uuid>") still short-circuit
    # without an LLM call; a router failure degrades to the keyword
    # planner, so routing is a quality improvement, never a new failure
    # mode. Off by default: unchanged behavior + no extra LLM call.
    agent_routing_enabled: bool = False

    # When True, a second agent — the Research agent
    # (services/research_agent.py) — owns the weak/insufficient-retrieval
    # path: it plans search queries, runs them, reads the top pages, and
    # synthesizes a grounded answer, instead of the corrective loop's
    # single search_web() call. Only effective when WEB_SEARCH_ENABLED is
    # also true (research needs the web-search tool); otherwise it
    # degrades to the normal retrieve path. Together with the router this
    # makes the app genuinely multi-agent (planner hands off to a
    # specialist), which is what the checklist's collaboration/handoff
    # rows measure. Off by default.
    research_agent_enabled: bool = False

    # Max sub-queries the Research agent's planning step may emit for one
    # user query.
    research_max_subqueries: int = 3

    # How many of the top web results' pages the Research agent actually
    # fetches and reads (its "read" step). The rest are used as snippets
    # only. Bounded so a research pass can't balloon into an unbounded
    # scrape.
    research_read_limit: int = 2

    # Per-page character cap for content the Research agent's read step
    # pulls into context, and per-page-fetch timeout in seconds.
    research_page_max_chars: int = 1500
    research_page_timeout_seconds: int = 15

    # Wall-clock budget for the whole plan->search->read pass (the LLM plan
    # and synthesis calls aren't counted — they have their own client-level
    # timeouts). Without this, worst case is fully additive: subqueries x
    # search timeout + read_limit x page timeout, which with defaults is
    # already 30s+30s before a single LLM call. Checked between searches
    # and between page reads; once exceeded, the agent stops collecting
    # more and synthesizes from whatever it already has (degrade, not
    # fail), same as every other bound in this module.
    research_total_timeout_seconds: int = 45

    # Number of web results fetched when the fallback fires.
    web_search_result_count: int = 3

    # Timeout, in seconds, for the web search call.
    web_search_timeout_seconds: int = 10

    # Enables hybrid search: fuse FAISS semantic search with a BM25 lexical
    # index (services/hybrid_search.py) instead of semantic search alone.
    # Defaults on: the ablation in docs/OPERATIONS.md's "Retrieval
    # ablation" showed it improving Precision@5/Recall@5/MRR over the
    # semantic-only baseline with no measured downside, and BM25 adds no
    # extra model/network dependency — cheap enough to ship as the
    # default rather than leave as an unproven opt-in. Still config-gated
    # so it can be A/B'd against the baseline (or turned off) via the
    # eval harness.
    hybrid_search_enabled: bool = True

    # Weight given to the (min-max normalized) semantic score in hybrid
    # fusion; BM25's weight is 1 - this value. Both scores are normalized
    # to [0, 1] before combining, so these weights are directly comparable.
    hybrid_semantic_weight: float = 0.6

    # Reciprocal Rank Fusion (RRF) smoothing parameter k. Default is 60.
    hybrid_rrf_k: int = 60

    # Checks for chemical safety cautions when chemical active ingredients are mentioned.
    chemical_safety_verification_enabled: bool = True

    # Enables cross-encoder re-ranking (services/reranking_service.py) of
    # retrieval's candidate pool. Off by default — unlike hybrid search,
    # this stays opt-in: the ablation showed a real Precision@5 gain
    # (0.37 -> 0.40) but only at n=7 queries, and re-ranking has a real
    # operational cost hybrid search doesn't (a second model load, plus
    # cross-encoder inference on every retrieve-action request) — not
    # enough evidence yet to default every request into paying that cost.
    # See docs/OPERATIONS.md's "Retrieval ablation" before flipping this.
    reranking_enabled: bool = False

    # Cross-encoder model used for re-ranking.
    reranking_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Candidate pool size for hybrid search (per retriever, before fusion)
    # and for reranking (before narrowing back to retrieval_top_k). A
    # wider pool than the final top_k gives fusion/reranking real material
    # to work with instead of just re-ordering an already-narrow shortlist.
    retrieval_candidate_k: int = 20

    # Timeout, in seconds, for a single retrieval call (search + optional
    # reranking). A hung vector search or cross-encoder inference otherwise
    # blocks the whole request indefinitely, so this is the "prevent
    # indefinite waiting" timeout the tool checklist asks for. 0 = no
    # timeout (unchanged legacy behavior).
    retrieval_timeout_seconds: float = 0.0

    # Gemini model used for text generation.
    gemini_model_name: str = "gemini-3.5-flash"

    # Timeout, in seconds, for Gemini API calls.
    gemini_timeout_seconds: int = 30

    # Estimated USD cost per 1,000 tokens (prompt + completion combined),
    # used only to log a rough cost estimate per generation — not actual
    # billed usage. Default is a placeholder; set to your model's real
    # blended rate.
    cost_per_1k_tokens: float = 0.00025

    # Encryption-at-rest key for app.core.encryption (AES-256-GCM), used by
    # app/services/postgres_session_store.py to encrypt/decrypt ChatTurn.content.
    # Loaded via Settings (like every other config value in this app) rather
    # than encryption.py's own os.environ fallback, since this codebase's
    # .env is parsed by pydantic-settings and never exported to the process
    # environment -- reading raw os.environ here would silently never see a
    # value set only in .env. None (unset) means encrypted session-content
    # read/write fails closed (EncryptionKeyMissingError), never falls back
    # to storing plaintext. Base64-encoded 32-byte key; generate one with
    # `python -c "from app.core.encryption import generate_key; print(generate_key())"`.
    encryption_key_b64: str | None = None

    # Base URL of the LeafSense vision service (a separate FastAPI process,
    # its own TensorFlow/Keras stack — see services/vision_client.py).
    # Defaults to 8001, NOT LeafSense's own default of 8000: LeafSense's
    # backend/main.py hardcodes port 8000 when run directly, which collides
    # with this backend's own default port. Start LeafSense with
    # `uvicorn main:app --port 8001` when running both services locally.
    vision_service_url: str = "http://127.0.0.1:8001"

    # Timeout, in seconds, for calls to the vision service.
    vision_service_timeout_seconds: int = 15

    # Optional shared secret for authenticating to the LeafSense vision
    # service. When set, diagnose_image() sends it as an "X-API-Key" header
    # on every call (mirroring this app's own inbound auth convention).
    # When empty, calls are unauthenticated — fine for local dev, a
    # deployment gap if LeafSense is publicly reachable.
    vision_service_api_key: str = ""

    # Below this confidence, diagnose_image() still returns its prediction
    # but flags it low_confidence=True (see models.document.VisionPrediction)
    # rather than silently presenting an uncertain guess as settled.
    vision_confidence_threshold: float = 0.5

    # Enables S3 sync of uploads/vector_store/feedback (services/
    # s3_sync_service.py) — a persistence mechanism for a hypothetical
    # ephemeral-filesystem deployment (e.g. AWS Lambda, whose filesystem is
    # read-only outside /tmp and wiped on every cold start). Off by
    # default: local dev, docker-compose, EC2, and any other deployment
    # with real persistent disk never touch S3 or need boto3 installed.
    s3_sync_enabled: bool = False

    # S3 bucket s3_sync_service.py reads/writes when s3_sync_enabled is
    # True. Required (non-empty) whenever sync is enabled.
    s3_sync_bucket_name: str = ""

    # Overrides where uploads/vector_store/feedback resolve on disk — normally
    # left empty, resolving relative to backend/ (Path(__file__).parents[2]
    # in each owning service module). Only relevant for a hypothetical
    # read-only-filesystem deployment target, which would set this to a
    # writable directory like "/tmp" before startup, and unlike a
    # symlink-at-runtime approach, this doesn't require deleting/replacing
    # the on-image directories at container start — impossible anyway,
    # since deleting anything under a read-only mount (including the empty
    # directories baked into the image) fails too.
    data_dir_override: str = ""

    # --- Multi-modal RAG (off by default) --------------------------------
    # Extracts embedded images from uploaded PDFs (see
    # document_service.extract_images_from_pdf) and stores them on disk
    # under image_storage_dir_name, so later phases (captioning, vision QA)
    # have the bytes to work with. Local PyMuPDF work only — no new
    # dependency and no external call. Off by default like every feature
    # here; image-caption chunks are only produced when the downstream
    # captioning flag is also on.
    image_extraction_enabled: bool = False

    # Captions each extracted image via Gemini's vision input
    # (gemini_client.generate_with_image) and indexes the captions into the
    # *existing* text vector store, metadata-tagged as image-derived
    # (source="image_caption") so citations can say "from a figure on page
    # N". Requires image_extraction_enabled and a vision-capable LLM
    # client; a caption failure degrades to "no caption chunk" for that
    # image, never a failed upload. Off by default — each caption is a paid
    # Gemini vision call.
    image_captioning_enabled: bool = False

    # Extracts tabular regions via PyMuPDF's built-in table detection
    # (page.find_tables) and feeds the tables into the same chunking/
    # embedding pipeline as structured markdown text (source="table").
    # Off by default — table detection is heuristic and opt-in.
    table_extraction_enabled: bool = False

    # Vision-grounded QA on full-page images: when a chat query's retrieval
    # is graded weak/insufficient, the low-text pages (the same
    # ocr_min_chars_per_page decision point extraction uses) of the most
    # relevant document are sent to Gemini directly instead of relying on
    # OCR'd text alone (see vision_qa_service.py). Requires
    # image_extraction_enabled (page rasters are produced there). Off by
    # default — another paid Gemini vision call per weak-retrieval query.
    vision_qa_enabled: bool = False

    # Directory (relative to backend/) where extracted images are persisted
    # (filename shape: {document_id}_{page_number}_{image_id}.{ext}).
    image_storage_dir_name: str = "extracted_images"

    # Images whose smaller side is below this many pixels are dropped
    # (icons, dividers, noise) rather than captioned or QA'd.
    image_min_side_px: int = 50

    # Upper bound on images extracted from one document — a pathological
    # PDF (thousands of vector tiles) must not balloon ingestion.
    image_max_count_per_document: int = 50

    # Hard cap on the characters of a caption that actually get chunked —
    # captions are short by nature; a runaway caption is truncated.
    image_caption_max_chars: int = 600

    # Most full-page images sent to Gemini in one vision-QA attempt.
    vision_qa_max_pages: int = 3

    # Upper bound on tables extracted from one document.
    table_max_count_per_document: int = 50

    # --- CLIP cross-modal retrieval (Phase 4, off by default) -----------
    # Enables CLIP-based image retrieval: uploaded documents' extracted
    # figures are embedded via the CLIP microservice (clip_service/, a
    # separate LeafSense-shaped process — see clip_client.py) into a second
    # FAISS index (image_index.faiss / image_metadata.json), and that image
    # similarity becomes a *third* signal in hybrid search alongside the
    # existing semantic + BM25 fusion. Off by default like every modal
    # flag: requires image_extraction_enabled (images must exist to embed)
    # and a running CLIP service. A CLIP service that is unreachable or
    # times out degrades to the two-signal hybrid path — cross-modal
    # retrieval is an enhancement, never a failure mode for retrieval.
    clip_embedding_enabled: bool = False

    # Base URL of the CLIP embedding microservice (a separate FastAPI
    # process — see clip_service/). Defaults to 8002, distinct from this
    # backend's 8000 and LeafSense's 8001.
    clip_service_url: str = "http://localhost:8002"

    # Timeout, in seconds, for calls to the CLIP service.
    clip_service_timeout_seconds: int = 15

    # Optional shared secret sent as an X-API-Key header to the CLIP
    # service (mirroring vision_service_api_key's convention). Empty =
    # unauthenticated, fine for local dev.
    clip_service_api_key: str = ""

    # Filenames (inside vector_store_dir_name) for the CLIP image index and
    # its metadata — separate from the text index so text and image vectors
    # live in their own spaces/persistence files.
    image_vector_index_filename: str = "image_index.faiss"
    image_vector_metadata_filename: str = "image_metadata.json"

    # Weight given to the (min-max normalized) CLIP image-similarity score
    # in the three-way fusion. Semantic and BM25 weights are scaled by
    # (1 - clip_weight) so all three sum to 1; clip_weight=0 reproduces the
    # current two-signal behavior exactly.
    hybrid_clip_weight: float = 0.2

    # --- Answer-quality / agentic / vector-store-hygiene flags ----------
    # (originally planned in _recycle_bin/docs-planning-history/FEATURE_PROMPTS.md's
    # 13-feature plan, archived during repo cleanup). Every one defaults False,
    # per this codebase's convention: a new capability is off until
    # explicitly enabled. Two exceptions ship without a flag — 1.1
    # (retrieval_confidence banner: pure info-surfacing of a grade the app
    # already computes internally, zero added cost) and 4.3 (keyword
    # highlighting: pure frontend rendering) — each calls that out where it
    # ships.

    # Agent 1.2 — When True, a follow-up question (one where conversation
    # history exists) is rewritten into a standalone question via one extra
    # LLM call before it's used for retrieval — the raw follow-up text alone
    # ("what about the other one?") often retrieves poorly since it's
    # missing the context a human reader would infer from prior turns.
    # Off by default: it's one additional LLM call, only on follow-up
    # turns (never on a first message, since there's no history to use).
    query_contextualization_enabled: bool = False

    # Agent 1.3 — When True, a generated answer's [N] citation markers are
    # checked against their cited chunks via one extra LLM call before the
    # answer is accepted — catching a claim that cites a real chunk but
    # misstates what it says (something _is_ungrounded's empty/fallback
    # check can't detect). Bounded to one combined call per answer, never
    # one call per citation. Off by default: real added cost on every
    # answer that has citations.
    citation_verification_enabled: bool = False

    # When True, ChatService._detect_hallucination runs a lexical
    # groundedness check after an answer is produced (Feature #4): the
    # fraction of the answer's content tokens that also appear in the
    # retrieved context (chunks + web snippets). An answer that shares
    # almost no vocabulary with the very context it claims to be grounded
    # on is flagged as a likely hallucination — surfaced via
    # ChatResponse.hallucination_detected and a "hallucination_detected"
    # log event / metric. It is a signal, never a blocker: the answer is
    # delivered unchanged so a correct-but-paraphrased answer can't be
    # degraded, and the flag is a pointer for the user to double-check.
    hallucination_detection_enabled: bool = True

    # Minimum fraction of the answer's content tokens that must appear in
    # the retrieved context for the answer NOT to be flagged. 0.25 is a
    # deliberately forgiving floor: a fully paraphrased answer typically
    # still reuses well over a quarter of the source's vocabulary, so a
    # below-floor answer is more likely invented from nothing than
    # rephrased.
    hallucination_grounding_threshold: float = 0.25

    # Shortest answer (in characters) that will be grounding-checked. The
    # number of tokens in shorter answers is too small for a ratio to be
    # statistically meaningful.
    hallucination_min_answer_chars: int = 40

    # Agent 1.4 — When True, if retrieval graded "insufficient" and the
    # corrective loop still couldn't produce a grounded answer, the app
    # asks the user one short clarifying question instead of returning the
    # fixed "couldn't find that" line — a better outcome when the real
    # problem is an ambiguous question, not missing content. Degrades to
    # today's exact fallback behavior on any LLM failure. Off by default:
    # one extra LLM call, only in this specific (rare) end state.
    clarifying_question_enabled: bool = False

    # Agent 2.2 — When True, one extra small LLM call after the main answer
    # suggests up to 3 short follow-up questions the user might ask next
    # (the "related questions" pattern). Parsed defensively — any failure
    # degrades to an empty list, never affects the main answer. Off by
    # default: added latency + one extra LLM call per request.
    follow_up_questions_enabled: bool = False

    # Agent 2.3 — When True, a weak/insufficient-retrieval query gets
    # handed to LocalResearchAgent (local_research_agent.py) instead of (or
    # ahead of) the web-search research agent: it decomposes the query into
    # sub-queries and runs multiple local retrieve() passes, merging the
    # results — useful for questions that need combining content from
    # different parts of a document. Off by default: adds LLM planning
    # + multiple retrieval calls per weak-retrieval query.
    local_research_agent_enabled: bool = False

    # Agent 2.3 — Max sub-queries LocalResearchAgent's planning step may emit.
    local_research_max_subqueries: int = 3

    # Agent 3.2 — When True, a new chunk whose embedding is a near-duplicate
    # (cosine similarity >= chunk_dedup_similarity_threshold) of an
    # already-indexed chunk from the SAME document is skipped rather than
    # indexed — this is what prevents the exact scoring bug observed this
    # session, where two near-identical chunks starve each other out via
    # hybrid search's min-max score normalization. Off by default: a
    # linear-scan comparison cost per new chunk within its own document.
    chunk_dedup_enabled: bool = False

    # Agent 3.2 — Cosine-similarity threshold above which two chunks from
    # the same document are treated as duplicates. High (0.97) — this
    # catches true near-duplicates, not just topically-similar chunks.
    chunk_dedup_similarity_threshold: float = 0.97

    # Agent 3.3 — When True, a newly-uploaded document's first chunk's
    # embedding is compared against other same-tenant documents'
    # first-chunk embeddings; if similarity >=
    # duplicate_document_similarity_threshold, the upload still succeeds
    # (never blocked) but the response names the document it resembles.
    # Off by default: adds one comparison pass per upload.
    duplicate_document_detection_enabled: bool = False

    # Agent 3.3 — Similarity threshold for the duplicate-document check
    # above.
    duplicate_document_similarity_threshold: float = 0.95

    # --- Agent memory (off by default) --------------------------------------
    # When True, ChatService keeps a bounded per-session working memory
    # (services/agent_memory.py): conversation turns plus durable key/value
    # facts extracted from answers, injected back into later router and
    # generation prompts as a "Remembered from earlier in this conversation"
    # block. This is what makes a follow-up like "and what about the other
    # document?" consistent with what was established earlier, independent
    # of how much live history survives trimming. In-memory and bounded
    # (same LRU discipline as the session store) — no new persistence
    # layer, so an ephemeral deployment keeps working exactly as before.
    agent_memory_enabled: bool = False

    # Bounds for the per-process agent-memory store, mirroring the session
    # store's own caps so memory can't grow without limit.
    agent_memory_max_sessions: int = 1000
    agent_memory_max_turns_per_session: int = 10
    agent_memory_max_facts_per_session: int = 20

    # When True (and agent_memory_enabled), one extra LLM call per answered
    # request extracts durable key/value facts from the answer and stores
    # them in the session's memory — the fact block a later question can
    # draw on. Off by default: added latency + one LLM call per request,
    # only worth it when the fact block is actually used.
    agent_memory_fact_extraction_enabled: bool = False

    # --- AgentExecutor orchestration (off by default) ----------------------
    # When True, ChatService routes eligible queries through
    # AgentExecutor (services/agent_executor.py) — a genuine ReAct loop:
    # PlanningAgent decides one next step at a time, reacting to what
    # earlier steps in the same run actually returned, over the dynamic
    # tool registry (services/tools/) instead of ChatService's inline
    # orchestration. Same interfaces, same grounding/citation guarantees
    # (the executor's synthesis reuses prompt_builder and the sources
    # machinery), just orchestration moved out of ChatService — the
    # architectural boundary the 10/10 upgrade asks for. Off by default:
    # unchanged inline behavior is the safe default; the executor is the
    # opt-in path. The executor still delegates to the specialized agents
    # (document_analyst/web_researcher/fact_checker/summarizer) when its
    # plan calls for them.
    agent_executor_enabled: bool = False

    # Max steps (tool invocations + LLM synthesis) one AgentExecutor run
    # may execute before it stops and synthesizes from what it has —
    # the executor's loop guard, mirroring the research agent's
    # research_total_timeout_seconds budget.
    agent_executor_max_steps: int = 6

    # --- Nightly eval job (off by default) ----------------------------------
    # When True, the startup/shutdown hook (see eval/nightly_eval.py) and
    # the .github/workflows/nightly-eval.yml scheduled workflow run the
    # eval harness and gate against a baseline. The workflow itself is
    # independent of this flag (it runs on its own schedule), but
    # NIGHTLY_EVAL_ENABLED=true lets a deployment opt the app's own
    # process into running a regression check on startup. Off by default —
    # the eval harness calls the live LLM and costs quota, so it must be
    # deliberately enabled.
    nightly_eval_enabled: bool = False

    # Dataset (inside backend/eval/) the nightly eval runs against, and
    # the committed baseline JSON (relative to backend/) it gates on. The
    # baseline is updated intentionally via eval/regression_check.py
    # re-baselining — see eval/README.md.
    nightly_eval_dataset: str = "dataset_v1.json"
    nightly_eval_baseline: str = "eval/baselines/v2_groq.json"

    # Absolute regression tolerance (0-1 scale) for the nightly gate,
    # passed through to regression_check.py --tol. 0.05 matches the eval
    # workflow's manual default.
    nightly_eval_tolerance: float = 0.05

    model_config = SettingsConfigDict(
        env_file=(".env", str(Path(__file__).resolve().parents[2] / ".env")),
        extra="ignore",
    )

    def data_dir(self, dir_name: str) -> Path:
        """Resolve a data directory (uploads/vector_store/feedback) under
        data_dir_override when set, else relative to backend/ as before."""
        base = (
            Path(self.data_dir_override)
            if self.data_dir_override
            else Path(__file__).resolve().parents[2]
        )
        return base / dir_name

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def api_key_table(self) -> dict[str, str]:
        """Return a dict of sha256_hash -> client_name for all configured keys.

        If api_keys (JSON) is set, use that. Otherwise fall back to the single
        api_key with a default client name "default". Keys are hashed with SHA-256
        so plaintext keys are never stored in memory beyond initial parsing.
        The dict is keyed by hash for O(1) lookup, avoiding a linear scan.
        """
        import hashlib
        import json

        if self.api_keys:
            try:
                raw = json.loads(self.api_keys)
            except json.JSONDecodeError as exc:
                raise ValueError(f"API_KEYS must be valid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError("API_KEYS must be a JSON object mapping client_name -> key")
            return {hashlib.sha256(key.encode()).hexdigest(): client for client, key in raw.items()}
        # Backward compatibility: single api_key becomes "default" client
        return {hashlib.sha256(self.api_key.encode()).hexdigest(): "default"}

    @property
    def admin_client_names_set(self) -> set[str]:
        """Parsed, whitespace-trimmed set of admin_client_names. Empty set
        (not an error) when unset — no client is admin by default."""
        return {name.strip() for name in self.admin_client_names.split(",") if name.strip()}

    @property
    def trusted_proxy_ips_set(self) -> set[str]:
        """Parsed, whitespace-trimmed set of trusted_proxy_ips. Empty set
        (not an error) when unset — no proxy is trusted by default, so
        X-Forwarded-For is ignored everywhere until this is configured."""
        return {ip.strip() for ip in self.trusted_proxy_ips.split(",") if ip.strip()}

    @model_validator(mode="after")
    def _reject_weak_secrets_in_production(self) -> "Settings":
        """Module 10 gap-closure: secrets management was previously
        documented (.env/.env.example, SSM path) but not code-enforced --
        nothing stopped a DEBUG=false (production) run from using the
        exact placeholder values .env.example ships with. This closes
        that gap: with DEBUG=true (the local-dev default), placeholder
        secrets are expected and left alone -- this validator only acts
        when DEBUG=false, i.e. a run explicitly claiming to be
        production. It fails fast at Settings() construction (before the
        app can serve a single request), matching the codebase's existing
        "fail loud, not silently insecure" posture (EncryptionKeyMissingError,
        AuthConfigurationError) rather than a runtime check elsewhere.
        """
        if self.debug:
            return self

        problems: list[str] = []

        def _is_weak(value: str, min_length: int) -> bool:
            return value.strip().lower() in _KNOWN_WEAK_SECRETS or len(value) < min_length

        if self.api_keys:
            # api_keys (plural) supersedes the single api_key entirely for
            # auth resolution once set (see api_key_hash_map) -- a weak
            # single api_key left over in the environment is then inert,
            # not a real exposure, so only the values actually used for
            # auth are validated here.
            import json

            try:
                parsed = json.loads(self.api_keys)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                for client_name, key_value in parsed.items():
                    if not isinstance(key_value, str) or _is_weak(key_value, 16):
                        problems.append(f"API_KEYS entry for client '{client_name}' is a placeholder or too short.")
        elif _is_weak(self.api_key, 16):
            problems.append(
                "API_KEY is missing, a known placeholder, or shorter than 16 characters."
            )

        # jwt_secret_key is genuinely optional (individual-user login is an
        # optional feature -- see create_access_token's own runtime check);
        # only validate it here if a value was actually supplied.
        if self.jwt_secret_key and _is_weak(self.jwt_secret_key, 32):
            problems.append(
                "JWT_SECRET_KEY is a known placeholder or shorter than 32 characters."
            )

        if self.database_url and "insightai-dev-password" in self.database_url:
            problems.append("DATABASE_URL still contains the local-dev default password.")

        if problems:
            raise ValueError(
                "Insecure secret configuration for a production run (DEBUG=false):\n- "
                + "\n- ".join(problems)
                + "\nGenerate strong secrets before deploying, e.g.: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return self


settings = Settings()
