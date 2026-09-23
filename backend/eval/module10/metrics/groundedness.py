"""Groundedness evaluation: does the generated answer's factual claims
hold up against retrieved evidence?

Three distinct signals, all reported, never conflated:

1. `lexical_groundedness` — re-exports run_eval.py's existing is_grounded()
   lexical-overlap proxy. Cheap, deterministic, but (per its own docstring)
   not a faithfulness check.
2. `llm_judge_groundedness` — re-exports run_eval.py's existing
   judge_groundedness() (a structured LLM-as-judge call). Labeled
   explicitly as evaluator-based measurement, not ground truth — the
   judge model/version is recorded alongside every verdict.
3. `claim_decomposition_groundedness` — NEW in this module: splits the
   answer into individual factual claims (one per sentence, filtered to
   sentences that assert something rather than hedge/refuse/ask), then
   checks each claim against the retrieved evidence text via the same
   lexical-overlap primitive run_eval.py already uses (content-word
   overlap) — reporting supported/unsupported per claim rather than one
   answer-level boolean. This is still a lexical proxy, not a real NLI
   entailment model (this project has no NLI dependency), but decomposing
   to claim level is real additional signal over a single whole-answer
   flag: an answer with 4 correct claims and 1 fabricated one scores
   0.8 here instead of a single pass/fail that hides the one bad claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eval.run_eval import (
    GroundednessJudgment,
    _content_words,
    build_groundedness_judge_prompt,
    is_grounded,
    judge_groundedness,
)

__all__ = [
    "ClaimDecompositionResult",
    "GroundednessJudgment",
    "build_groundedness_judge_prompt",
    "claim_decomposition_groundedness",
    "is_grounded",
    "judge_groundedness",
    "lexical_groundedness",
]

lexical_groundedness = is_grounded
llm_judge_groundedness = judge_groundedness

# A "claim" sentence asserts something; these starts mark a sentence as a
# hedge, question, or refusal instead (nothing to fact-check against
# evidence) — excluded from the denominator rather than counted as an
# unsupported claim, since "I don't know" is not a factual assertion.
_NON_CLAIM_PREFIXES = (
    "i don't", "i do not", "i'm not sure", "i am not sure", "could you",
    "can you", "would you", "what do you mean", "sorry,", "i couldn't find",
    "i could not find",
)


def _split_claims(answer: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    return [
        s for s in sentences
        if not any(s.lower().startswith(prefix) for prefix in _NON_CLAIM_PREFIXES)
    ]


@dataclass
class ClaimDecompositionResult:
    total_claims: int
    supported_claims: int
    unsupported: list[str]
    groundedness_rate: float | None


def claim_decomposition_groundedness(answer: str, evidence_texts: list[str]) -> ClaimDecompositionResult:
    """Split `answer` into claim sentences, mark each supported if it
    shares at least one non-trivial content word with the combined
    evidence text (retrieved chunks + gold expected_facts, when
    available) — the same overlap primitive run_eval.py's is_grounded()
    uses, applied per-claim instead of per-answer.
    """
    claims = _split_claims(answer)
    if not claims:
        return ClaimDecompositionResult(0, 0, [], None)

    evidence_words: set[str] = set()
    for text in evidence_texts:
        evidence_words |= _content_words(text)

    unsupported: list[str] = []
    supported = 0
    for claim in claims:
        claim_words = _content_words(claim)
        if claim_words and (claim_words & evidence_words):
            supported += 1
        else:
            unsupported.append(claim)

    return ClaimDecompositionResult(
        total_claims=len(claims),
        supported_claims=supported,
        unsupported=unsupported,
        groundedness_rate=round(supported / len(claims), 4),
    )
