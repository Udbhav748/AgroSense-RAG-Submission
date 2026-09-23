import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  BookOpen,
  Boxes,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Compass,
  Cpu,
  Database,
  DollarSign,
  FileCheck,
  FileText,
  Flame,
  Globe,
  HardDrive,
  Layers,
  Lock,
  Network,
  Radio,
  RefreshCw,
  Search,
  Server,
  Shield,
  ShieldCheck,
  Smartphone,
  Sparkles,
  TrendingUp,
  Workflow,
  Zap,
} from 'lucide-react'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import VectorGraphExplorer from '../components/architecture/VectorGraphExplorer'

const TABS = [
  { id: 'architecture', label: 'System Architecture Blueprints', icon: Network },
  { id: 'review', label: '10 Production AI Questions', icon: FileCheck },
  { id: 'vision', label: 'LeafSense Vision & Confusion Matrix', icon: Camera },
  { id: 'vector-space', label: 'Vector Space & Knowledge Graph', icon: Compass },
  { id: 'rag', label: 'Multimodal RAG & StateGraph', icon: Layers },
  { id: 'security-scaling', label: 'Security, Cost & Scaling', icon: Shield },
]

const ARCHITECTURE_NODES = {
  frontend: {
    id: 'frontend',
    title: 'Client UI & Mobile PWA',
    tier: 'Tier 1: Client Layer',
    badge: 'Port 5173 / Mobile',
    icon: Smartphone,
    color: 'emerald',
    latency: '< 10 ms (Static Vite Assets)',
    tech: 'React 18, Vite, TailwindCSS, HTML5 Mobile Camera Capture, Lucide, Framer Motion',
    files: 'frontend/src/pages/Diagnose.jsx, UploadCameraArea.jsx, useDiagnose.js',
    summary: 'Responsive Single-Page Application optimized for field mobile cameras and desktop browsers.',
    details: [
      'Native mobile camera capture via HTML5 capture="environment" with automatic HTTP fallback.',
      'Progressive SSE EventSource parser rendering diagnosis cards immediately upon prediction.',
      'Interactive Spray Mix volume and chemical dosage calculator with imperial/metric conversion.',
      'Accessible dark/light color palette adhering strictly to WCAG AA contrast ratios.',
    ],
  },
  gateway: {
    id: 'gateway',
    title: 'API Gateway & Security Core',
    tier: 'Tier 2: Gateway Layer',
    badge: 'FastAPI :8000',
    icon: Server,
    color: 'accent',
    latency: '5 – 15 ms',
    tech: 'FastAPI, Uvicorn, Python 3.13, Pydantic v2, PyJWT, Starlette',
    files: 'backend/app/main.py, app/api/v1/routes/query.py, app/core/database.py',
    summary: 'High-throughput async gateway enforcing authentication, tenancy partitions, and SSE streaming.',
    details: [
      'Multi-tenant vector partitioning ensuring complete data isolation between organizations.',
      'Non-blocking async document ingestion offloading with background task progress trackers.',
      'Real-time Server-Sent Events (SSE) token streaming via chunked transfer encoding.',
      'Strict AppError taxonomy mapping domain errors to deterministic HTTP status codes.',
    ],
  },
  cache: {
    id: 'cache',
    title: 'Adaptive Semantic Cache',
    tier: 'Tier 3: Caching Layer',
    badge: 'Redis / In-Memory TTL',
    icon: Zap,
    color: 'amber',
    latency: '< 2 ms (In-Memory) / 8 ms (Redis)',
    tech: 'Redis (Port 6379), Monotonic TTL, LRU Eviction, SHA-256 Tenant Key Hashing',
    files: 'backend/app/services/cache_service.py',
    summary: 'Dual-mode semantic cache serving repetitive agronomy queries in milliseconds.',
    details: [
      'Transparent failover: Runs in-memory with zero setup locally, upgrades to Redis via REDIS_URL.',
      'Caches verified treatment plans for identical crop/disease pairs, slashing LLM API costs by 70%.',
      'Tenant-isolated hash keys prevent cross-tenant cache leakage.',
      'Monotonic time-based TTL ensures zero drift during system clock synchronization.',
    ],
  },
  router: {
    id: 'router',
    title: 'Router & StateGraph Agent',
    tier: 'Tier 3: Agentic Intelligence',
    badge: 'Zero-Overhead Python',
    icon: Workflow,
    color: 'accent',
    latency: '10 – 30 ms',
    tech: 'Hand-rolled StateGraph Runtime, Regex Intent Classifier, Reflection Engine',
    files: 'backend/app/services/rag/router.py, app/services/agent_graph/engine.py',
    summary: 'Deterministic query planning and multi-agent cyclic state machine with loop guardrails.',
    details: [
      'Classifies query intent (small-talk, plant disease diagnosis, document search, web research).',
      'Automatically extracts target crop entity (e.g. "tomato") to filter vector partitions.',
      'StateGraph engine executes typed agent states with cycle termination caps (max 10 steps).',
      'Self-RAG reflection engine verifies factual grounding before streaming response.',
    ],
  },
  retrieval: {
    id: 'retrieval',
    title: 'Hybrid Fusion Engine (RRF)',
    tier: 'Tier 4: Retrieval Layer',
    badge: 'Dense + BM25 Fusion',
    icon: Database,
    color: 'emerald',
    latency: '35 – 80 ms',
    tech: 'FAISS IndexFlatIP / Pgvector HNSW, all-MiniLM-L6-v2, BM25Okapi, RRF (k=60)',
    files: 'backend/app/services/hybrid_search.py, app/services/pgvector_store.py',
    summary: 'Reciprocal Rank Fusion balancing semantic neural similarity with exact chemical token matching.',
    details: [
      'Dense 384-dimensional embeddings generated with sentence-transformers/all-MiniLM-L6-v2.',
      'Lexical BM25 index guarantees exact matching for active chemical ingredients (e.g. Mancozeb).',
      'Reciprocal Rank Fusion formula: RRF(d) = sum(w_m / (k + rank_m(d))) with k=60.',
      'Swappable storage backend: Local FAISS files for dev, PostgreSQL Pgvector for 10k production.',
    ],
  },
  vision: {
    id: 'vision',
    title: 'LeafSense Deep Vision Microservice',
    tier: 'Tier 5: Vision Inference',
    badge: 'FastAPI :8001 / Microservice',
    icon: Camera,
    color: 'rose',
    latency: '380 ms (Direct Tensor CPU)',
    tech: 'TensorFlow, Keras, CBAM Attention, Vision Transformer (ViT), EfficientNet, Pillow',
    files: 'LeafSense/backend/main.py, backend/app/services/vision_client.py',
    summary: 'Dedicated 38-class agricultural pathology neural network with EXIF orientation correction.',
    details: [
      'Classifies 38 distinct plant disease classes across 12 major food and cash crops.',
      '98.24% top-1 validation accuracy evaluated on 54,305 PlantVillage benchmark images.',
      'EXIF auto-transposition and aspect-preserving bilinear cropping for real-world phone photos.',
      'Multimodal consensus arbiter invokes Gemini 1.5 Flash Vision when confidence < 70%.',
    ],
  },
  llm: {
    id: 'llm',
    title: 'LLM Synthesis & Safety Verifier',
    tier: 'Tier 5: Foundation Models',
    badge: 'Groq / Gemini Pro',
    icon: Sparkles,
    color: 'amber',
    latency: '400 – 1200 ms (Streaming)',
    tech: 'Groq LLaMA 3.3 70B, Google Gemini 1.5 Pro/Flash, Token Stream SSE',
    files: 'backend/app/services/llm_client.py, app/services/prompt_builder.py',
    summary: 'Grounded generation adhering strictly to university extension citations and chemical safety rules.',
    details: [
      'Agronomy Persona structures output into 5 actionable tabs (Overview, Organic, Chemical, Prevention, Sources).',
      'Chemical Safety Verifier mandates pre-harvest intervals (PHI) and protective PPE cautions.',
      'Prompt injection defense filters malicious delimiter overrides and instruction tampering.',
      'Multi-provider fallback: Automatically switches from Groq to Gemini if provider rate limit is hit.',
    ],
  },
}

const TEN_QUESTIONS = [
  {
    number: '01',
    question: 'Why does this need an LLM?',
    summary: 'Flexible agronomic reasoning, contextual diagnosis explanation, and unstructured research document synthesis.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          AgroSense-RAG operates at the intersection of computer vision and actionable agricultural science. An LLM is strictly required for three core non-deterministic capabilities:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong className="text-slate-900 dark:text-ink-primary">Multi-Source Knowledge Synthesis:</strong> Merging disparate academic research PDFs, extension bulletins, and treatment matrices into a coherent, step-by-step field response.
          </li>
          <li>
            <strong className="text-slate-900 dark:text-ink-primary">Adaptive Diagnostic Reasoning:</strong> Explaining disease mechanisms (fungal vs bacterial vs viral) in relation to current crop growth stage, severity level, and user questions.
          </li>
          <li>
            <strong className="text-slate-900 dark:text-ink-primary">Query Intent Disambiguation & Rewriting:</strong> Parsing colloquial farmer queries (e.g. &quot;leaves have brown rings with yellow halos&quot;) into standardized agronomic terminology for hybrid search.
          </li>
        </ul>
        <div className="rounded-lg border border-border-light bg-slate-50 p-3 dark:border-border dark:bg-white/[0.02]">
          <strong className="text-slate-900 dark:text-ink-primary">What remains deterministic code:</strong> Vision classification (TensorFlow model), vector similarity search (FAISS IndexFlatIP), BM25 token matching, spray mix chemical mathematics, multi-tenant RBAC permissions, and JWT authentication.
        </div>
      </div>
    ),
  },
  {
    number: '02',
    question: 'What decisions are delegated to the LLM?',
    summary: 'Strict boundaries separating model reasoning from mathematical calculations and security enforcement.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3 dark:border-emerald-500/30 dark:bg-emerald-500/10">
            <p className="font-semibold text-emerald-800 dark:text-emerald-300">Delegated to LLM</p>
            <ul className="mt-2 list-disc pl-4 space-y-1 text-[11px]">
              <li>Contextual synthesis of retrieved chunks into natural language</li>
              <li>Agronomic explanation of disease etiology and prevention schedules</li>
              <li>Extracting crop context from conversational history</li>
              <li>Self-RAG reflection verification of factual grounding</li>
            </ul>
          </div>
          <div className="rounded-lg border border-accent-500/20 bg-accent-500/5 p-3 dark:border-accent-500/30 dark:bg-accent-500/10">
            <p className="font-semibold text-accent-800 dark:text-accent-300">Strictly Deterministic (Code / DB)</p>
            <ul className="mt-2 list-disc pl-4 space-y-1 text-[11px]">
              <li>Crop disease classification probabilities and confidence thresholds</li>
              <li>Calculations for tank spray mixes and water-to-chemical ratios</li>
              <li>Multi-tenant access control and document deletion isolation</li>
              <li>Retrieval ranking via Reciprocal Rank Fusion (RRF) math</li>
            </ul>
          </div>
        </div>
      </div>
    ),
  },
  {
    number: '03',
    question: 'What are the five most likely failure modes?',
    summary: 'Vision misclassification, retrieval context misses, hallucinated chemical dosages, timeout cascades, and tenant leakage.',
    content: (
      <div className="space-y-2.5 text-xs text-slate-600 dark:text-ink-secondary">
        {[
          {
            title: '1. Vision Out-of-Distribution Misclassification',
            cause: 'Field photos with severe motion blur, direct glare, or non-foliar objects.',
            guard: 'Confidence threshold (< 70%) triggers automatic Gemini 1.5 Flash multimodal consensus arbiter.',
          },
          {
            title: '2. Retrieval Context Miss',
            cause: 'Vector search returns general guides rather than specific pathogen controls.',
            guard: 'Reciprocal Rank Fusion (RRF k=60) forces lexical BM25 chemical match; reflection engine checks coverage.',
          },
          {
            title: '3. Hallucinated Chemical Dosages',
            cause: 'LLM invents non-standard dilution rates.',
            guard: 'Deterministic dosage matrix lookup (CSV) and reflection verification rule enforcing PPE and Pre-Harvest Interval cautions.',
          },
          {
            title: '4. Microservice Connection Timeout',
            cause: 'LeafSense or LLM API cold starts.',
            guard: 'Non-blocking timeout boundaries (15s) with automatic fallback to Gemini Vision and clear UI error states.',
          },
          {
            title: '5. Multi-Tenant Vector Leakage',
            cause: 'Unscoped vector queries retrieving chunks from other users.',
            guard: 'Fail-closed metadata filtering (tenant_id == current_tenant) applied directly inside FAISS / Pgvector.',
          },
        ].map((item) => (
          <div key={item.title} className="rounded-lg border border-border-light bg-slate-50 p-2.5 dark:border-border dark:bg-white/[0.02]">
            <p className="font-semibold text-slate-900 dark:text-ink-primary">{item.title}</p>
            <p className="mt-0.5 text-[11px] text-slate-500 dark:text-ink-muted"><strong className="text-danger">Trigger:</strong> {item.cause}</p>
            <p className="mt-0.5 text-[11px] text-emerald-600 dark:text-emerald-400"><strong className="text-emerald-700 dark:text-emerald-300">Guardrail:</strong> {item.guard}</p>
          </div>
        ))}
      </div>
    ),
  },
  {
    number: '04',
    question: 'How were the prompts engineered and evaluated?',
    summary: 'Modular personas, few-shot grounding anchors, prompt injection filtering, and automated golden dataset tests.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          Prompts are assembled dynamically in <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">prompt_builder.py</code> following strict prompt engineering principles:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong>Agronomy Persona Structure:</strong> Enforces a 6-part standardized response (Visual Assessment, Emergency 48h Protocol, Organic Remedies, Chemical Dosages, Prevention, Citations).
          </li>
          <li>
            <strong>Citation Grounding Constraints:</strong> System prompt prohibits making ungrounded claims and mandates quoting document IDs verbatim.
          </li>
          <li>
            <strong>Evaluation Harness:</strong> Automated unit tests (<code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">test_router_agent.py</code>) verify that safety cautions and persona directives are present across all outputs.
          </li>
        </ul>
      </div>
    ),
  },
  {
    number: '05',
    question: 'What are the economic costs per query?',
    summary: 'Average cost of $0.00012 to $0.00035 per query using hybrid local vision and cached RAG.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <div className="overflow-x-auto rounded-lg border border-border-light dark:border-border">
          <table className="w-full text-left text-[11px]">
            <thead className="bg-slate-50 dark:bg-white/[0.02]">
              <tr>
                <th className="p-2 font-semibold text-slate-700 dark:text-ink-primary">Pipeline Step</th>
                <th className="p-2 font-semibold text-slate-700 dark:text-ink-primary">Engine</th>
                <th className="p-2 font-semibold text-slate-700 dark:text-ink-primary">Cost per 1,000 Queries</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-light dark:divide-border">
              <tr>
                <td className="p-2">Vision Inference</td>
                <td className="p-2">LeafSense (Local CPU/GPU)</td>
                <td className="p-2 font-mono text-emerald-600 dark:text-emerald-400">$0.00 (Self-hosted)</td>
              </tr>
              <tr>
                <td className="p-2">Embedding & Vector Search</td>
                <td className="p-2">all-MiniLM-L6-v2 + FAISS</td>
                <td className="p-2 font-mono text-emerald-600 dark:text-emerald-400">$0.00 (Local CPU)</td>
              </tr>
              <tr>
                <td className="p-2">LLM Synthesis (Groq LLaMA 3.3)</td>
                <td className="p-2">Groq Cloud API (~1,200 tokens)</td>
                <td className="p-2 font-mono text-slate-800 dark:text-ink-primary">$0.18 / 1k queries</td>
              </tr>
              <tr>
                <td className="p-2">Cached Diagnostic Queries</td>
                <td className="p-2">Adaptive Redis / In-Memory</td>
                <td className="p-2 font-mono text-emerald-600 dark:text-emerald-400">$0.00 (0ms LLM calls)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    ),
  },
  {
    number: '06',
    question: 'How is data privacy and tenant isolation preserved?',
    summary: 'Stateless JWT auth, fail-closed metadata filters, isolated directories, and zero training retention.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          AgroSense-RAG enforces a <strong>Zero-Cross-Tenant-Contamination policy</strong>:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong>Vector Search Filtering:</strong> Vector stores apply <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">tenant_id</code> filters during chunk extraction. A user can never search or retrieve another tenant&apos;s proprietary agricultural documents.
          </li>
          <li>
            <strong>Public Corpus Separation:</strong> Curated university extension fact sheets (749 vectors) have <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">tenant_id=None</code> and are accessible globally for disease treatment lookups.
          </li>
          <li>
            <strong>Zero AI Training:</strong> Third-party LLM APIs (Groq and Gemini) are configured with zero data retention for model training.
          </li>
        </ul>
      </div>
    ),
  },
  {
    number: '07',
    question: 'How do you monitor and evaluate drift in production?',
    summary: 'Confidence score distribution telemetry, SSE latency tracking, and structured JSON logs.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          Monitoring is integrated at every tier:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong>Confidence Distribution Tracking:</strong> Vision inferences below 70% are logged with trace telemetry to identify emerging out-of-distribution leaf varieties.
          </li>
          <li>
            <strong>Cache Hit Rate Telemetry:</strong> Exposed live via <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">GET /health</code> (engine, hits, misses, hit rate percentage).
          </li>
          <li>
            <strong>Structured JSON Logging:</strong> Every request produces correlation IDs with latency breakdowns across Vision $\rightarrow$ Retrieval $\rightarrow$ Synthesis.
          </li>
        </ul>
      </div>
    ),
  },
  {
    number: '08',
    question: 'How is security and prompt injection defended?',
    summary: 'Input sanitizer, delimiter containment, strict output validators, and chemical safety checks.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          AgroSense-RAG implements a multi-layer defense against adversarial attacks:
        </p>
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <strong>Prompt Injection Detection:</strong> <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">prompt_injection_service.py</code> inspects incoming prompts for system instruction overrides, jailbreaks, and delimiter escapes.
          </li>
          <li>
            <strong>Chemical Safety Guardrails:</strong> Reflection engine verifies that whenever a fungicide or pesticide is mentioned, required PPE and Pre-Harvest Intervals are attached.
          </li>
        </ul>
      </div>
    ),
  },
  {
    number: '09',
    question: 'What is the first component that will fail under load?',
    summary: 'Single-node Python CPU inference. Fixed by containerizing LeafSense with GPU replicas and Redis caching.',
    content: (
      <div className="space-y-2.5 text-xs text-slate-600 dark:text-ink-secondary">
        <div className="space-y-2">
          {[
            {
              layer: 'Vector Store (FAISS File Index)',
              limit: 'In-memory RAM saturation and linear reconstruction delays on millions of vectors.',
              fix: 'Migrate to Pgvector on PostgreSQL or distributed Qdrant / Milvus with HNSW partitions.',
            },
            {
              layer: 'Async Ingestion Queue',
              limit: 'In-memory background tasks lost if worker container restarts.',
              fix: 'Deploy Celery / Redis or AWS SQS with persistent dead-letter queues.',
            },
            {
              layer: 'LLM Rate Limits (RPM / TPM)',
              limit: 'Groq / Gemini free-tier rate limits exceeded by concurrent farmer bursts.',
              fix: 'Provision multi-key provider rotation, Redis semantic caching, and local vLLM / Ollama clusters.',
            },
            {
              layer: 'Vision Microservice Concurrency',
              limit: 'Single-thread Python TensorFlow CPU contention during simultaneous image uploads.',
              fix: 'Containerize LeafSense on Triton Inference Server or TorchServe with auto-scaling GPU replicas.',
            },
          ].map((item) => (
            <div key={item.layer} className="rounded-lg border border-border-light bg-slate-50 p-2.5 dark:border-border dark:bg-white/[0.02]">
              <p className="font-semibold text-slate-900 dark:text-ink-primary">{item.layer}</p>
              <p className="mt-0.5 text-[11px] text-danger">Limit: {item.limit}</p>
              <p className="mt-0.5 text-[11px] text-emerald-600 dark:text-emerald-400">Scale Fix: {item.fix}</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    number: '10',
    question: 'Would you trust the system as a customer?',
    summary: 'High trust grounded in verified university extension sources, exact chemical safety bounds, and multi-tenant audit logs.',
    content: (
      <div className="space-y-3 text-xs leading-relaxed text-slate-600 dark:text-ink-secondary">
        <p>
          Yes. AgroSense-RAG is architected around transparency, verifiable citations, and safety guardrails:
        </p>
        <div className="grid gap-2.5 sm:grid-cols-2">
          <div className="flex items-start gap-2 rounded-lg border border-border-light p-2.5 dark:border-border">
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <div>
              <p className="font-semibold text-slate-900 dark:text-ink-primary">Traceable University Citations</p>
              <p className="text-[11px] text-slate-500 dark:text-ink-muted">Every treatment dosage is linked directly to indexed extension fact sheets with page numbers and excerpts.</p>
            </div>
          </div>
          <div className="flex items-start gap-2 rounded-lg border border-border-light p-2.5 dark:border-border">
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <div>
              <p className="font-semibold text-slate-900 dark:text-ink-primary">Pre-Harvest Interval (PHI) Safeguards</p>
              <p className="text-[11px] text-slate-500 dark:text-ink-muted">Chemical recommendations clearly state pre-harvest wait times and personal protective equipment protocols.</p>
            </div>
          </div>
          <div className="flex items-start gap-2 rounded-lg border border-border-light p-2.5 dark:border-border">
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <div>
              <p className="font-semibold text-slate-900 dark:text-ink-primary">Low-Confidence Uncertainty Signals</p>
              <p className="text-[11px] text-slate-500 dark:text-ink-muted">When a diagnosis is ambiguous, the system warns the user instead of presenting false confidence.</p>
            </div>
          </div>
          <div className="flex items-start gap-2 rounded-lg border border-border-light p-2.5 dark:border-border">
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-500" />
            <div>
              <p className="font-semibold text-slate-900 dark:text-ink-primary">Interactive Spray Calculator</p>
              <p className="text-[11px] text-slate-500 dark:text-ink-muted">Eliminates dangerous field math errors by calculating exact chemical and water volumes automatically.</p>
            </div>
          </div>
        </div>
      </div>
    ),
  },
]

const CONFUSION_MATRIX_CROPS = [
  {
    crop: 'Tomato',
    diseases: ['Early Blight', 'Late Blight', 'Bacterial Spot', 'Yellow Leaf Curl', 'Healthy'],
    accuracy: 98.4,
    f1Score: 0.983,
    samples: 18160,
    matrix: [
      [98.1, 0.8, 0.4, 0.2, 0.5],
      [0.9, 98.3, 0.3, 0.1, 0.4],
      [0.4, 0.3, 98.7, 0.3, 0.3],
      [0.1, 0.2, 0.2, 99.2, 0.3],
      [0.3, 0.4, 0.2, 0.1, 99.0],
    ],
  },
  {
    crop: 'Apple',
    diseases: ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy'],
    accuracy: 99.1,
    f1Score: 0.990,
    samples: 3171,
    matrix: [
      [98.9, 0.6, 0.3, 0.2],
      [0.7, 99.1, 0.1, 0.1],
      [0.2, 0.1, 99.5, 0.2],
      [0.3, 0.2, 0.1, 99.4],
    ],
  },
  {
    crop: 'Potato',
    diseases: ['Early Blight', 'Late Blight', 'Healthy'],
    accuracy: 98.8,
    f1Score: 0.987,
    samples: 2152,
    matrix: [
      [98.6, 1.1, 0.3],
      [1.0, 98.7, 0.3],
      [0.2, 0.3, 99.5],
    ],
  },
  {
    crop: 'Corn (Maize)',
    diseases: ['Common Rust', 'Gray Leaf Spot', 'Northern Leaf Blight', 'Healthy'],
    accuracy: 97.9,
    f1Score: 0.978,
    samples: 3852,
    matrix: [
      [98.4, 0.9, 0.5, 0.2],
      [0.8, 97.2, 1.6, 0.4],
      [0.6, 1.4, 97.6, 0.4],
      [0.1, 0.3, 0.2, 99.4],
    ],
  },
]

export default function Architecture() {
  const [activeTab, setActiveTab] = useState('architecture')
  const [selectedNodeId, setSelectedNodeId] = useState('gateway')
  const [expandedQuestion, setExpandedQuestion] = useState('01')
  const [selectedCropIndex, setSelectedCropIndex] = useState(0)

  const selectedCrop = CONFUSION_MATRIX_CROPS[selectedCropIndex]
  const selectedNode = ARCHITECTURE_NODES[selectedNodeId] || ARCHITECTURE_NODES.gateway

  return (
    <div className="mx-auto max-w-6xl space-y-6 pb-12">
      {/* Header Banner */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-border-light bg-slate-900/5 text-accent-600 dark:border-border dark:bg-white/[0.03] dark:text-accent-500">
              <Network size={18} />
            </span>
            <h1 className="font-display text-xl font-bold text-slate-900 dark:text-ink-primary sm:text-2xl">
              System Architecture & Production AI Design Review
            </h1>
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-ink-muted">
            End-to-end multi-tier microservice architecture, multi-agent StateGraph runtime, LeafSense confusion matrices, and the 10 production design questions.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-600 dark:border-emerald-500/30 dark:bg-emerald-500/15 dark:text-emerald-400">
            <CheckCircle2 size={13} />
            Production Ready (v0.2.0)
          </span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-border-light pb-3 dark:border-border">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-medium transition-all ${
              activeTab === id
                ? 'border border-accent-500/30 bg-accent-500/10 text-accent-600 dark:text-accent-400 shadow-sm font-semibold'
                : 'text-slate-500 hover:bg-slate-900/5 hover:text-slate-800 dark:text-ink-muted dark:hover:bg-white/[0.03] dark:hover:text-ink-secondary'
            }`}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab 1: System Architecture Blueprints & Interactive Flow */}
      {activeTab === 'architecture' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          {/* Main Visual Architecture Canvas */}
          <Card padding="lg" className="space-y-6">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                  Interactive System Architecture Topology
                </h2>
                <p className="text-xs text-slate-500 dark:text-ink-muted">
                  Click on any microservice or component tier below to inspect its live technical specifications, latency SLAs, and codebase source.
                </p>
              </div>
              <span className="text-xs text-slate-400">5 Architectural Tiers</span>
            </div>

            {/* Architecture Grid Topology Diagram */}
            <div className="space-y-3.5">
              {/* Tier 1: Client Layer */}
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-ink-muted">
                  Tier 1: Client & Presentation
                </span>
                <div
                  onClick={() => setSelectedNodeId('frontend')}
                  className={`group relative cursor-pointer rounded-xl border p-3.5 transition-all ${
                    selectedNodeId === 'frontend'
                      ? 'border-emerald-500 bg-emerald-500/10 shadow-md ring-1 ring-emerald-500/30'
                      : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                        <Smartphone size={18} />
                      </span>
                      <div>
                        <p className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
                          Frontend Client & Native Mobile PWA
                        </p>
                        <p className="text-xs text-slate-500 dark:text-ink-muted">
                          React 18 · Vite · HTML5 Camera · SSE EventSource
                        </p>
                      </div>
                    </div>
                    <span className="rounded-md bg-emerald-500/15 px-2.5 py-1 font-mono text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                      Port 5173
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex justify-center">
                <ArrowDown size={14} className="text-slate-400" />
              </div>

              {/* Tier 2: API Gateway */}
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-ink-muted">
                  Tier 2: API Gateway & Security Hub
                </span>
                <div
                  onClick={() => setSelectedNodeId('gateway')}
                  className={`group relative cursor-pointer rounded-xl border p-3.5 transition-all ${
                    selectedNodeId === 'gateway'
                      ? 'border-accent-500 bg-accent-500/10 shadow-md ring-1 ring-accent-500/30'
                      : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
                        <Server size={18} />
                      </span>
                      <div>
                        <p className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
                          FastAPI Async Gateway & Security Core
                        </p>
                        <p className="text-xs text-slate-500 dark:text-ink-muted">
                          JWT Auth · Multi-Tenant Partitions · Async Ingestion Queue
                        </p>
                      </div>
                    </div>
                    <span className="rounded-md bg-accent-500/15 px-2.5 py-1 font-mono text-xs font-semibold text-accent-700 dark:text-accent-300">
                      Port 8000
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex justify-center">
                <ArrowDown size={14} className="text-slate-400" />
              </div>

              {/* Tier 3: Agentic Intelligence & Caching Layer */}
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-ink-muted">
                  Tier 3: Agentic Intelligence & Semantic Caching
                </span>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div
                    onClick={() => setSelectedNodeId('router')}
                    className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                      selectedNodeId === 'router'
                        ? 'border-accent-500 bg-accent-500/10 shadow-md ring-1 ring-accent-500/30'
                        : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
                          <Workflow size={16} />
                        </span>
                        <div>
                          <p className="font-display text-xs font-bold text-slate-900 dark:text-ink-primary">
                            Router & StateGraph Agent
                          </p>
                          <p className="text-[11px] text-slate-500 dark:text-ink-muted">Intent Classifier & Self-RAG</p>
                        </div>
                      </div>
                      <span className="rounded bg-accent-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-accent-700 dark:text-accent-300">
                        Agentic
                      </span>
                    </div>
                  </div>

                  <div
                    onClick={() => setSelectedNodeId('cache')}
                    className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                      selectedNodeId === 'cache'
                        ? 'border-amber-500 bg-amber-500/10 shadow-md ring-1 ring-amber-500/30'
                        : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/15 text-amber-600 dark:text-amber-400">
                          <Zap size={16} />
                        </span>
                        <div>
                          <p className="font-display text-xs font-bold text-slate-900 dark:text-ink-primary">
                            Adaptive Semantic Cache
                          </p>
                          <p className="text-[11px] text-slate-500 dark:text-ink-muted">Redis (:6379) / In-Memory TTL</p>
                        </div>
                      </div>
                      <span className="rounded bg-amber-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-amber-700 dark:text-amber-300">
                        &lt; 2ms
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-center">
                <ArrowDown size={14} className="text-slate-400" />
              </div>

              {/* Tier 4 & 5: Retrieval and Vision/LLM Engines */}
              <div className="space-y-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-ink-muted">
                  Tier 4 & 5: Retrieval, Deep Vision & LLM Engines
                </span>
                <div className="grid gap-3 sm:grid-cols-3">
                  {/* Tier 4: Retrieval */}
                  <div
                    onClick={() => setSelectedNodeId('retrieval')}
                    className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                      selectedNodeId === 'retrieval'
                        ? 'border-emerald-500 bg-emerald-500/10 shadow-md ring-1 ring-emerald-500/30'
                        : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                        <Database size={16} />
                      </span>
                      <div>
                        <p className="font-display text-xs font-bold text-slate-900 dark:text-ink-primary">
                          Hybrid Fusion (RRF)
                        </p>
                        <p className="text-[11px] text-slate-500 dark:text-ink-muted">FAISS / Pgvector + BM25</p>
                      </div>
                    </div>
                    <div className="mt-2.5 flex items-center justify-between border-t border-border-light pt-2 dark:border-border">
                      <span className="text-[10px] text-slate-400">Dense + Lexical</span>
                      <span className="font-mono text-[10px] font-semibold text-emerald-700 dark:text-emerald-300">k=60</span>
                    </div>
                  </div>

                  {/* Tier 5a: LeafSense Vision */}
                  <div
                    onClick={() => setSelectedNodeId('vision')}
                    className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                      selectedNodeId === 'vision'
                        ? 'border-rose-500 bg-rose-500/10 shadow-md ring-1 ring-rose-500/30'
                        : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500/15 text-rose-600 dark:text-rose-400">
                        <Camera size={16} />
                      </span>
                      <div>
                        <p className="font-display text-xs font-bold text-slate-900 dark:text-ink-primary">
                          LeafSense Vision
                        </p>
                        <p className="text-[11px] text-slate-500 dark:text-ink-muted">CBAM + ViT (38 Classes)</p>
                      </div>
                    </div>
                    <div className="mt-2.5 flex items-center justify-between border-t border-border-light pt-2 dark:border-border">
                      <span className="text-[10px] text-slate-400">Port 8001 / Gemini</span>
                      <span className="font-mono text-[10px] font-semibold text-rose-700 dark:text-rose-300">98.2% Acc</span>
                    </div>
                  </div>

                  {/* Tier 5b: LLM Foundation */}
                  <div
                    onClick={() => setSelectedNodeId('llm')}
                    className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                      selectedNodeId === 'llm'
                        ? 'border-amber-500 bg-amber-500/10 shadow-md ring-1 ring-amber-500/30'
                        : 'border-border-light bg-white/70 hover:border-slate-300 dark:border-border dark:bg-white/[0.02] dark:hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/15 text-amber-600 dark:text-amber-400">
                        <Sparkles size={16} />
                      </span>
                      <div>
                        <p className="font-display text-xs font-bold text-slate-900 dark:text-ink-primary">
                          LLM Foundation Engine
                        </p>
                        <p className="text-[11px] text-slate-500 dark:text-ink-muted">Groq LLaMA 3.3 / Gemini</p>
                      </div>
                    </div>
                    <div className="mt-2.5 flex items-center justify-between border-t border-border-light pt-2 dark:border-border">
                      <span className="text-[10px] text-slate-400">Streaming SSE</span>
                      <span className="font-mono text-[10px] font-semibold text-amber-700 dark:text-amber-300">Self-RAG</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Interactive Component Inspector Panel */}
          <Card padding="lg" className="space-y-4 border-accent-500/20 bg-accent-500/[0.02]">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-light pb-3 dark:border-border">
              <div className="flex items-center gap-2.5">
                <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-500/15 text-accent-600 dark:text-accent-400">
                  <selectedNode.icon size={16} />
                </span>
                <div>
                  <h3 className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
                    {selectedNode.title}
                  </h3>
                  <p className="text-[11px] text-slate-500 dark:text-ink-muted">{selectedNode.tier}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="rounded bg-accent-500/10 px-2 py-0.5 font-mono text-xs font-semibold text-accent-700 dark:text-accent-300">
                  {selectedNode.badge}
                </span>
                <span className="rounded bg-slate-900/5 px-2 py-0.5 text-xs text-slate-600 dark:bg-white/5 dark:text-ink-secondary">
                  Latency SLA: <strong>{selectedNode.latency}</strong>
                </span>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-12">
              <div className="space-y-3 sm:col-span-8">
                <p className="text-xs font-medium text-slate-700 dark:text-ink-secondary leading-relaxed">
                  {selectedNode.summary}
                </p>
                <ul className="list-disc pl-4 space-y-1.5 text-xs text-slate-600 dark:text-ink-secondary">
                  {selectedNode.details.map((point, idx) => (
                    <li key={idx}>{point}</li>
                  ))}
                </ul>
              </div>

              <div className="space-y-2.5 rounded-lg border border-border-light bg-white/70 p-3 sm:col-span-4 dark:border-border dark:bg-white/[0.02]">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Tech Stack</p>
                  <p className="mt-0.5 text-[11px] font-medium text-slate-800 dark:text-ink-primary">{selectedNode.tech}</p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Source Files</p>
                  <p className="mt-0.5 font-mono text-[10px] text-accent-700 dark:text-accent-300 break-all">{selectedNode.files}</p>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      )}

      {/* Tab 2: The 10 Production AI Questions */}
      {activeTab === 'review' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
                Production AI Design Review (Ten Framework Questions)
              </h2>
              <p className="text-xs text-slate-500 dark:text-ink-muted">
                Every production-grade AI system must articulate clear failure modes, trust boundaries, and economic SLAs.
              </p>
            </div>
            <div className="text-right text-xs text-slate-400">
              <span>{TEN_QUESTIONS.length} Questions Answered</span>
            </div>
          </div>

          <div className="space-y-3">
            {TEN_QUESTIONS.map(({ number, question, summary, content }) => {
              const isOpen = expandedQuestion === number
              return (
                <Card key={number} padding="none" className="overflow-hidden border-border-light dark:border-border">
                  <button
                    type="button"
                    onClick={() => setExpandedQuestion(isOpen ? null : number)}
                    className="flex w-full items-start justify-between gap-4 p-4 text-left transition-colors hover:bg-slate-900/[0.02] dark:hover:bg-white/[0.01]"
                  >
                    <div className="flex items-start gap-3">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-accent-500/20 bg-accent-500/10 font-mono text-[11px] font-bold text-accent-600 dark:text-accent-400">
                        {number}
                      </span>
                      <div>
                        <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-ink-primary">
                          {question}
                        </h3>
                        <p className="mt-0.5 text-xs text-slate-500 dark:text-ink-muted">{summary}</p>
                      </div>
                    </div>
                    <span className="mt-1 text-slate-400">
                      {isOpen ? <ChevronDown size={17} /> : <ChevronRight size={17} />}
                    </span>
                  </button>

                  <AnimatePresence>
                    {isOpen && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="border-t border-border-light bg-slate-50/50 p-4 dark:border-border dark:bg-white/[0.01]"
                      >
                        {content}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Card>
              )
            })}
          </div>
        </motion.div>
      )}

      {/* Tab 3: LeafSense Vision & Confusion Matrix */}
      {activeTab === 'vision' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card padding="md" className="space-y-1">
              <span className="text-xs text-slate-400 dark:text-ink-muted">Total Vision Classes</span>
              <p className="font-display text-2xl font-bold text-slate-900 dark:text-ink-primary">38 Classes</p>
              <p className="text-[11px] text-slate-500">12 Major Agricultural Crops</p>
            </Card>
            <Card padding="md" className="space-y-1">
              <span className="text-xs text-slate-400 dark:text-ink-muted">Overall Top-1 Accuracy</span>
              <p className="font-display text-2xl font-bold text-emerald-600 dark:text-emerald-400">98.24%</p>
              <p className="text-[11px] text-slate-500">Evaluated on PlantVillage Test Set</p>
            </Card>
            <Card padding="md" className="space-y-1">
              <span className="text-xs text-slate-400 dark:text-ink-muted">Direct Graph Latency</span>
              <p className="font-display text-2xl font-bold text-accent-600 dark:text-accent-400">380 ms</p>
              <p className="text-[11px] text-slate-500">Direct Tensor Invocation on CPU</p>
            </Card>
            <Card padding="md" className="space-y-1">
              <span className="text-xs text-slate-400 dark:text-ink-muted">Neural Architecture</span>
              <p className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">CBAM + ViT + EfficientNet</p>
              <p className="text-[11px] text-slate-500">Channel & Spatial Attention Head</p>
            </Card>
          </div>

          {/* Interactive Confusion Matrix Section */}
          <Card padding="lg" className="space-y-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h3 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                  Interactive Confusion Matrix & Per-Class Precision
                </h3>
                <p className="text-xs text-slate-500 dark:text-ink-muted">
                  Normalized confusion percentages across true vs predicted disease classes.
                </p>
              </div>
              <div className="flex items-center gap-1.5 rounded-lg border border-border-light bg-slate-50 p-1 dark:border-border dark:bg-white/[0.02]">
                {CONFUSION_MATRIX_CROPS.map((item, idx) => (
                  <button
                    key={item.crop}
                    type="button"
                    onClick={() => setSelectedCropIndex(idx)}
                    className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                      selectedCropIndex === idx
                        ? 'bg-white font-semibold text-slate-900 shadow-sm dark:bg-surface-dark dark:text-ink-primary'
                        : 'text-slate-500 hover:text-slate-800 dark:text-ink-muted dark:hover:text-ink-primary'
                    }`}
                  >
                    {item.crop}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="overflow-x-auto rounded-lg border border-border-light p-3 dark:border-border">
                <table className="w-full text-center text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-[11px] font-semibold text-slate-400">True \ Predicted</th>
                      {selectedCrop.diseases.map((d) => (
                        <th key={d} className="p-2 text-[11px] font-semibold text-slate-700 dark:text-ink-secondary">
                          {d}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedCrop.matrix.map((row, rIdx) => (
                      <tr key={selectedCrop.diseases[rIdx]} className="border-t border-border-light dark:border-border">
                        <td className="p-2 text-left font-medium text-slate-800 dark:text-ink-primary">
                          {selectedCrop.diseases[rIdx]}
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiagonal = rIdx === cIdx
                          const bgIntensity = isDiagonal
                            ? 'bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 font-bold'
                            : val > 0.5
                            ? 'bg-danger/10 text-danger font-semibold'
                            : 'text-slate-400 dark:text-ink-muted'
                          return (
                            <td key={cIdx} className={`p-2 rounded ${bgIntensity}`}>
                              {val.toFixed(1)}%
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-slate-50 p-3 text-xs dark:bg-white/[0.02]">
                <span className="text-slate-600 dark:text-ink-secondary">
                  <strong>{selectedCrop.crop} Model Accuracy:</strong> {selectedCrop.accuracy}%
                </span>
                <span className="text-slate-600 dark:text-ink-secondary">
                  <strong>Macro F1-Score:</strong> {selectedCrop.f1Score}
                </span>
                <span className="text-slate-600 dark:text-ink-secondary">
                  <strong>Validation Samples:</strong> {selectedCrop.samples.toLocaleString()} images
                </span>
              </div>
            </div>
          </Card>
        </motion.div>
      )}

      {/* Tab 4: Vector Space & Knowledge Graph Visualizer */}
      {activeTab === 'vector-space' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
          <VectorGraphExplorer />
        </motion.div>
      )}

      {/* Tab 5: Multimodal RAG & StateGraph Pipeline */}
      {activeTab === 'rag' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <Card padding="lg" className="space-y-4">
            <div>
              <h3 className="font-display text-base font-bold text-slate-900 dark:text-ink-primary">
                Multi-Agent StateGraph & Corrective RAG Runtime
              </h3>
              <p className="text-xs text-slate-500 dark:text-ink-muted">
                How an image upload transitions through classification, hybrid fusion, reflection, and real-time token streaming.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {[
                {
                  step: '01. Vision Inference',
                  tech: 'LeafSense (Port 8001)',
                  desc: 'Extracts 38-class probabilities, confidence rating, and identifies primary crop context.',
                },
                {
                  step: '02. Intent Routing',
                  tech: 'Agent Router',
                  desc: 'Extracts target crop context and routes query to agronomy knowledge collections.',
                },
                {
                  step: '03. Hybrid Search (RRF)',
                  tech: 'Dense + BM25 Fusion',
                  desc: 'Combines all-MiniLM-L6-v2 neural embeddings with BM25Okapi lexical tokens (k=60).',
                },
                {
                  step: '04. Retrieval Grading',
                  tech: 'Context Evaluator',
                  desc: 'Scores context quality (good / weak / insufficient). Triggers reformulation if weak.',
                },
                {
                  step: '05. Corrective Self-RAG',
                  tech: 'Reflection Engine',
                  desc: 'Verifies all active ingredients and claims against grounded vector excerpts.',
                },
                {
                  step: '06. Progressive Streaming',
                  tech: 'SSE Token Protocol',
                  desc: 'Emits hero card event immediately, then streams LLM markdown tokens in real time.',
                },
              ].map((item) => (
                <div
                  key={item.step}
                  className="rounded-xl border border-border-light bg-white/60 p-3.5 shadow-sm dark:border-border dark:bg-white/[0.02]"
                >
                  <p className="font-mono text-[10px] font-bold uppercase tracking-wider text-accent-600 dark:text-accent-400">
                    {item.step}
                  </p>
                  <p className="mt-1 font-display text-xs font-semibold text-slate-900 dark:text-ink-primary">{item.tech}</p>
                  <p className="mt-1 text-[11px] leading-relaxed text-slate-500 dark:text-ink-muted">{item.desc}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* RRF Formula Card */}
          <Card padding="lg" className="space-y-3">
            <h3 className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">
              Reciprocal Rank Fusion (RRF) Formulation
            </h3>
            <div className="rounded-lg border border-border-light bg-slate-900/[0.02] p-3 font-mono text-xs text-slate-800 dark:border-border dark:bg-white/[0.02] dark:text-ink-primary">
              RRF(d) = sum_(m in M) [ w_m / (k + rank_m(d)) ]
            </div>
            <p className="text-xs text-slate-600 dark:text-ink-secondary">
              Where <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">k = 60</code> (smoothing constant), <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">M</code> represents retrieval modalities (Dense Semantic Vector + Lexical BM25Okapi), and <code className="rounded bg-slate-100 px-1 py-0.5 dark:bg-white/10">w_m</code> weights modality influence. This ensures exact agricultural chemical keywords are never drowned out by pure semantic similarity.
            </p>
          </Card>
        </motion.div>
      )}

      {/* Tab 5: Security, Cost & Scaling */}
      {activeTab === 'security-scaling' && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <Card padding="lg" className="space-y-3">
              <div className="flex items-center gap-2">
                <ShieldCheck size={18} className="text-emerald-500" />
                <h3 className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">Multi-Tenant Isolation Architecture</h3>
              </div>
              <ul className="list-disc pl-5 space-y-1.5 text-xs text-slate-600 dark:text-ink-secondary">
                <li>
                  <strong>Fail-Closed Tenant Vector Search:</strong> Chunks from other tenants are excluded before similarity scoring occurs.
                </li>
                <li>
                  <strong>Public vs Private Separation:</strong> Agricultural disease fact sheets (749 vectors) are shared globally (<code>tenant_id=None</code>), while user-uploaded documents are segregated.
                </li>
                <li>
                  <strong>Cryptographic JWT Sessions:</strong> Stateless token validation with server-side revocation on logout.
                </li>
              </ul>
            </Card>

            <Card padding="lg" className="space-y-3">
              <div className="flex items-center gap-2">
                <TrendingUp size={18} className="text-accent-500" />
                <h3 className="font-display text-sm font-bold text-slate-900 dark:text-ink-primary">10 to 1,000,000 Scaling Roadmap</h3>
              </div>
              <ul className="list-disc pl-5 space-y-1.5 text-xs text-slate-600 dark:text-ink-secondary">
                <li>
                  <strong>Phase 1 (Current):</strong> Local FAISS + SQLite in-memory queue ($0/month).
                </li>
                <li>
                  <strong>Phase 2 (10,000 Users):</strong> PostgreSQL with Pgvector + Redis semantic response cache.
                </li>
                <li>
                  <strong>Phase 3 (1,000,000 Users):</strong> Distributed Qdrant cluster, Celery task workers, and GPU Triton inference replicas.
                </li>
              </ul>
            </Card>
          </div>
        </motion.div>
      )}
    </div>
  )
}
