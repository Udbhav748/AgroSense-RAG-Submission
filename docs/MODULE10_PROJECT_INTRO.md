# Module 10 Project Introduction

## Project

AgroSense-RAG

## What it is

A document Q&A / agentic RAG system with a secondary agronomy-diagnosis
surface layered on top: upload a PDF (or a plant-leaf photo), ask
questions, get an answer grounded in retrieved passages with inline
citations. Concretely, it combines:

- Hybrid retrieval (BM25 + dense vector search, `hybrid_search.py`) with
  optional cross-encoder reranking (`reranking_service.py`)
- A corrective RAG loop — retrieval grading, a web-search/vision/local-
  research escalation when retrieval is weak, and bounded regeneration
  when the initial answer looks ungrounded (`rag_service.py`)
- An explicit, traced agent workflow graph (`agent_graph/`) that
  orchestrates the above as named nodes with typed state, human-approval
  gates, and Prometheus-style metrics (see `docs/ARCHITECTURE.md`'s
  "Explicit Agent Workflow" section)
- LeafSense: a plant-disease vision model (crop/disease classification
  from a leaf photo) whose prediction is turned into a query and run
  through the same RAG pipeline for a grounded agronomic explanation
- Session memory, PII detection, prompt-injection defenses, and a
  Prometheus `/metrics` endpoint

## Problem

The core problem is grounded question-answering over documents a user
actually uploaded, with citations a user can check — not a general
chatbot. A secondary, concrete instance of that same problem: a grower
photographs a diseased leaf and wants a diagnosis plus practical,
sourced treatment guidance, rather than a generic web search result. Both
share the same underlying requirement this project is built around:
**never answer past what the retrieved evidence actually supports**, and
make that boundary machine-checkable (retrieval scoring, hallucination
detection, structural citations) rather than just a prompt instruction.

No exaggerated claims: this is a portfolio-scale system running on a
single process against a FAISS index and (for LeafSense) an external
vision service — not a distributed, multi-tenant production platform,
though it does implement real multi-tenant primitives (API-key-scoped
tenants, RBAC) as part of exploring that direction.

## Architecture

See `docs/ARCHITECTURE.md` for the full diagram and component
descriptions, and its "Explicit Agent Workflow (Phase 1)" section
specifically for the agent-graph topology this Module 10 audit evaluates.
Summary:

```
Client (React SPA)
   -> FastAPI (auth, rate limit)
   -> agent_graph (validate -> plan -> cache -> retrieve -> grade
                    -> [augment if weak] -> generate -> reflect
                    -> validate -> finalize)
   -> FAISS + BM25 (hybrid_search.py) / cross-encoder reranker
   -> Gemini or Groq (configurable, with fallback)
   -> Prometheus-style /metrics
```

## Repository

<https://github.com/Udbhav748/AgroSense-RAG>

## Live/Demo

**Live demo: Not verified at audit time.** `docs/OPERATIONS.md` documents
that an earlier Render/Vercel deployment existed historically but that
path has since been removed from the repository; the current documented
path is manual, self-hosted Docker Compose on EC2, not a standing public
URL. No live URL is claimed here.
