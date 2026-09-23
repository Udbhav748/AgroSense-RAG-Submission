"""Real NLI-based faithfulness/groundedness scoring using a pretrained
cross-encoder entailment model -- NOT trained by this project. This
upgrades the disclosed lexical-overlap proxy (scripts/run_rag_eval.py's
compute_faithfulness) with genuine premise->hypothesis entailment
inference: premise = a retrieved chunk's text, hypothesis = one claim
(sentence) from the generated answer.

Why no training: NLI models trained on MNLI/SNLI (hundreds of thousands
of human-labeled premise/hypothesis pairs) already generalize well to
"does this text support this claim" -- exactly the RAG-groundedness
question. Training a new one from scratch would need a comparably large
labeled dataset this project doesn't have, GPU time, and would likely
underperform a model already trained at that scale. Loaded the same way
app/services/reranking_service.py's cross-encoder reranker is: lazy,
cached, no training step, using a small MiniLM-class NLI cross-encoder
(cross-encoder/nli-MiniLM2-L6-H768) to match this project's existing
model-size class (all-MiniLM-L6-v2 for embeddings, ms-marco-MiniLM-L-6-v2
for reranking) rather than a much larger model like nli-deberta-v3-base.

Per-chunk, not concatenated-context, scoring: a claim's entailment is
checked against EACH retrieved chunk individually and the MAXIMUM
entailment probability is kept -- "is this claim supported by ANY
retrieved chunk," the standard RAG-groundedness definition. Concatenating
all chunks into one giant premise would risk truncation past the
model's max sequence length (bounded, ~512 tokens for MiniLM-class
NLI models) and silently drop content that could support a claim; this
avoids that entirely.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
# A claim is "supported" if the best-matching chunk's entailment
# probability meets this threshold. 0.5 is the natural decision
# boundary for a 3-way softmax (entailment more likely than not,
# relative to contradiction+neutral combined) -- not tuned/cherry-picked
# against this project's own dataset.
ENTAILMENT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    """Load the NLI cross-encoder once and reuse it on every subsequent
    call -- same lru_cache rationale as reranking_service.get_reranker():
    serializes concurrent first-call loads instead of racing, and a
    failed load isn't cached, so the next call retries."""
    from sentence_transformers import CrossEncoder

    return cast("CrossEncoder", CrossEncoder(NLI_MODEL_NAME))


def entailment_probability(premise: str, hypothesis: str) -> float:
    """Real cross-encoder NLI inference for one (premise, hypothesis)
    pair. Returns P(entailment) via softmax over the model's raw
    3-class logits, in this model family's standard label order
    [contradiction, entailment, neutral] (verified empirically against
    known entailment/contradiction example pairs, not assumed from
    documentation alone)."""
    import numpy as np

    model = get_nli_model()
    logits = model.predict([(premise, hypothesis)])[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return float(probs[1])  # index 1 = entailment


def compute_nli_faithfulness(answer: str, retrieved_chunks: list[Any]) -> dict:
    """Real entailment-based faithfulness for one generated answer
    against its retrieved context -- reuses
    scripts.run_rag_eval.compute_faithfulness's own sentence-splitting
    logic (imported, not reimplemented) so the lexical and NLI scores
    are directly comparable claim-for-claim, from the same claim set.

    Returns the same shape family as compute_faithfulness
    ((score, supported, total)-equivalent as a dict) plus a per-claim
    breakdown with each claim's best-matching chunk and entailment
    probability, for transparency -- not just a bare number.
    """
    from scripts.run_rag_eval import _SENTENCE_SPLIT_RE, _get_chunk_text

    if not answer or not answer.strip():
        return {"nli_faithfulness": 0.0, "supported_claims": 0, "total_claims": 0, "per_claim": []}

    chunk_texts = [_get_chunk_text(c) for c in retrieved_chunks]
    chunk_texts = [t for t in chunk_texts if t and t.strip()]
    if not chunk_texts:
        return {"nli_faithfulness": 0.0, "supported_claims": 0, "total_claims": 1, "per_claim": []}

    raw_sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    claims = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
    if not claims:
        claims = [answer.strip()]

    per_claim = []
    supported = 0
    for claim in claims:
        best_prob = 0.0
        best_chunk_index = -1
        for i, chunk_text in enumerate(chunk_texts):
            prob = entailment_probability(chunk_text, claim)
            if prob > best_prob:
                best_prob = prob
                best_chunk_index = i
        is_supported = best_prob >= ENTAILMENT_THRESHOLD
        if is_supported:
            supported += 1
        per_claim.append(
            {
                "claim": claim,
                "best_entailment_probability": round(best_prob, 4),
                "best_matching_chunk_index": best_chunk_index,
                "supported": is_supported,
            }
        )

    score = supported / len(claims) if claims else 1.0
    return {
        "nli_faithfulness": round(max(0.0, min(1.0, score)), 4),
        "supported_claims": supported,
        "total_claims": len(claims),
        "per_claim": per_claim,
    }
