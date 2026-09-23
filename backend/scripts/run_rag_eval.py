"""Quantitative RAG Evaluation & Benchmarking Script for AgroSense-RAG.

Evaluates and benchmarks the hybrid RRF + Cross-Encoder retrieval and answer
generation pipeline against a golden dataset of 20 plant pathology Q&A pairs
covering major agricultural crops (Tomato, Potato, Apple, Corn, Grape, Orange, Pepper).

Standard RAG Metrics Computed:
1. Faithfulness Score (0.0 - 1.0):
   Evaluates whether claims in the generated answer are grounded in the retrieved context chunks.
2. Context Recall (0.0 - 1.0):
   Evaluates whether ground-truth active ingredients and organic remedies are present in retrieved chunks.
3. Context Precision (0.0 - 1.0):
   Computes rank-weighted precision (Average Precision @ K / Mean Reciprocal Rank) of relevant chunks.
4. Answer Relevance (0.0 - 1.0):
   Evaluates semantic embedding similarity and topic alignment between query and generated answer.
5. Harmonic Composite RAG Score (0.0 - 1.0):
   Weighted composite & harmonic blend across all active evaluation dimensions.

Usage:
    python backend/scripts/run_rag_eval.py
    python backend/scripts/run_rag_eval.py --limit 5
    python backend/scripts/run_rag_eval.py --no-llm --output data/eval_reports/ci_retrieval_report.json
    python backend/scripts/run_rag_eval.py --dataset path/to/custom_dataset.json --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_ROOT.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

logger = logging.getLogger("rag_eval")

# Default report output path
DEFAULT_OUTPUT_REPORT_PATH = _PROJECT_ROOT / "data" / "eval_reports" / "latest_eval_report.json"

# ---------------------------------------------------------------------------
# Golden Evaluation Dataset (20 Comprehensive Plant Pathology Q&A Pairs)
# ---------------------------------------------------------------------------

GOLDEN_DATASET: list[dict[str, Any]] = [
    # 1. Tomato - Early Blight
    {
        "id": "eval-tomato-01",
        "crop": "tomato",
        "disease": "early blight",
        "pathogen": "Alternaria solani",
        "query": "What fungicides and organic methods treat early blight on tomatoes caused by Alternaria solani?",
        "ground_truth_answer": (
            "Early blight on tomatoes caused by Alternaria solani is treated with chemical fungicides "
            "such as Chlorothalonil 75% WP, Azoxystrobin 23% SC, and Difenoconazole 25% EC. "
            "Organic remedies include Copper octanoate, Potassium bicarbonate, and Bacillus subtilis (Serenade ASO), "
            "along with pruning the lower 12 inches of foliage to eliminate soil-splash contact."
        ),
        "expected_active_ingredients": ["Chlorothalonil", "Azoxystrobin", "Difenoconazole"],
        "expected_organic_remedies": ["Copper octanoate", "Potassium bicarbonate", "Bacillus subtilis"],
    },
    # 2. Tomato - Late Blight
    {
        "id": "eval-tomato-02",
        "crop": "tomato",
        "disease": "late blight",
        "pathogen": "Phytophthora infestans",
        "query": "How do I control late blight on tomato plants and what chemical active ingredients work?",
        "ground_truth_answer": (
            "Late blight on tomatoes (Phytophthora infestans) is controlled with chemical active ingredients "
            "including Mandipropamid 23.4% SC (Revus), Cyazofamid 34.5% SC (Ranman), and Cymoxanil 60% WG. "
            "Organic controls include Copper hydroxide (2.5g/L), Bordeaux mixture 1%, and Potassium phosphonate salts. "
            "Infected plants must be rogued and bagged in plastic immediately; never compost infected tomato tissue."
        ),
        "expected_active_ingredients": ["Mandipropamid", "Cyazofamid", "Cymoxanil"],
        "expected_organic_remedies": ["Copper hydroxide", "Bordeaux mixture", "Potassium phosphonate"],
    },
    # 3. Tomato - Bacterial Spot
    {
        "id": "eval-tomato-03",
        "crop": "tomato",
        "disease": "bacterial spot",
        "pathogen": "Xanthomonas spp.",
        "query": "What spray treatments and active ingredients manage bacterial spot on tomato leaves and fruit?",
        "ground_truth_answer": (
            "Bacterial spot on tomatoes is treated with a synergistic tank mix of Copper hydroxide and "
            "Mancozeb 75% WP or Actigard (Acibenzolar-S-methyl). Organic options combine Copper hydroxide 53.8% WG "
            "with Bacillus amyloliquefaciens strain D747. Field workers should never work when plants are wet."
        ),
        "expected_active_ingredients": ["Copper hydroxide", "Mancozeb", "Acibenzolar-S-methyl"],
        "expected_organic_remedies": ["Copper hydroxide", "Bacillus amyloliquefaciens"],
    },
    # 4. Tomato - Septoria Leaf Spot
    {
        "id": "eval-tomato-04",
        "crop": "tomato",
        "disease": "septoria leaf spot",
        "pathogen": "Septoria lycopersici",
        "query": "How should Septoria leaf spot on tomatoes be treated with fungicides and organic controls?",
        "ground_truth_answer": (
            "Septoria leaf spot (Septoria lycopersici) is treated with Chlorothalonil 75% WP or Mancozeb 75% WP. "
            "Organic remedies include Copper hydroxide 53.8% WG and cold-pressed Neem oil, accompanied by heavy "
            "organic or plastic mulch to suppress spore splash from the soil."
        ),
        "expected_active_ingredients": ["Chlorothalonil", "Mancozeb"],
        "expected_organic_remedies": ["Copper hydroxide", "Neem oil"],
    },
    # 5. Tomato - Leaf Mold
    {
        "id": "eval-tomato-04b",
        "crop": "tomato",
        "disease": "leaf mold",
        "pathogen": "Passalora fulva",
        "query": "What is the recommended fungicide and organic management for tomato leaf mold in greenhouse tunnels?",
        "ground_truth_answer": (
            "Tomato leaf mold (Passalora fulva) is managed with Difenoconazole 25% EC or Chlorothalonil 75% WP. "
            "Organic remedies include Copper sulfate and Potassium bicarbonate, with high-tunnel exhaust "
            "ventilation to keep relative humidity strictly below 85%."
        ),
        "expected_active_ingredients": ["Difenoconazole", "Chlorothalonil"],
        "expected_organic_remedies": ["Copper sulfate", "Potassium bicarbonate"],
    },
    # 6. Tomato - Tomato Yellow Leaf Curl Virus
    {
        "id": "eval-tomato-05",
        "crop": "tomato",
        "disease": "tomato yellow leaf curl virus",
        "pathogen": "TYLCV (Begomovirus)",
        "query": "How do you manage Tomato Yellow Leaf Curl Virus and control its whitefly vector?",
        "ground_truth_answer": (
            "Because there is no chemical cure for Tomato Yellow Leaf Curl Virus, management focuses on vector control "
            "of Bemisia tabaci whiteflies using Thiamethoxam 25% WG, Acetamiprid 20% SP, or Spirotetramat 150 OD. "
            "Organic methods include yellow sticky card traps, insecticidal soap, silver reflective mulch, and "
            "Encarsia formosa parasitoids."
        ),
        "expected_active_ingredients": ["Thiamethoxam", "Acetamiprid", "Spirotetramat"],
        "expected_organic_remedies": ["Yellow sticky cards", "Insecticidal soap", "Silver reflective mulch", "Encarsia formosa"],
    },
    # 7. Tomato - Two-Spotted Spider Mite
    {
        "id": "eval-tomato-06",
        "crop": "tomato",
        "disease": "two-spotted spider mite",
        "pathogen": "Tetranychus urticae",
        "query": "What miticides and organic biological controls eliminate two-spotted spider mites on tomatoes?",
        "ground_truth_answer": (
            "Two-spotted spider mites (Tetranychus urticae) on tomatoes are controlled with miticides like "
            "Abamectin 1.8% EC, Spiromesifen 22.9% SC, and Bifenazate 43.2% SC. Organic controls include "
            "insecticidal soap, cold-pressed Neem oil 0.5%, and releasing predatory mites (Phytoseiulus persimilis)."
        ),
        "expected_active_ingredients": ["Abamectin", "Spiromesifen", "Bifenazate"],
        "expected_organic_remedies": ["Insecticidal soap", "Neem oil", "Phytoseiulus persimilis"],
    },
    # 8. Potato - Early Blight
    {
        "id": "eval-potato-01",
        "crop": "potato",
        "disease": "early blight",
        "pathogen": "Alternaria solani",
        "query": "What fungicides and bio-treatments effectively manage potato early blight caused by Alternaria solani?",
        "ground_truth_answer": (
            "Potato early blight is managed chemically with Azoxystrobin 23% SC or Chlorothalonil 75% WP. "
            "Organic remedies include Copper oxychloride 50% WP, Bacillus subtilis, and Potassium phosphite, "
            "rotating FRAC group 11 strobilurins with FRAC group M5 chlorothalonil to prevent resistance."
        ),
        "expected_active_ingredients": ["Azoxystrobin", "Chlorothalonil"],
        "expected_organic_remedies": ["Copper oxychloride", "Bacillus subtilis", "Potassium phosphite"],
    },
    # 9. Potato - Late Blight
    {
        "id": "eval-potato-02",
        "crop": "potato",
        "disease": "late blight",
        "pathogen": "Phytophthora infestans",
        "query": "How can potato growers control late blight and what active ingredients provide effective protection?",
        "ground_truth_answer": (
            "Late blight on potatoes is treated chemically with Metalaxyl-M + Mancozeb (Ridomil Gold), "
            "Mandipropamid 23.4% SC, or Cymoxanil 60% WG. Organic controls include Bordeaux mixture 1% (Copper sulfate), "
            "Copper hydroxide, and Trichoderma harzianum, alongside destroying volunteer potatoes and cull piles."
        ),
        "expected_active_ingredients": ["Metalaxyl-M", "Mancozeb", "Mandipropamid", "Cymoxanil"],
        "expected_organic_remedies": ["Copper sulfate", "Bordeaux mixture", "Copper hydroxide", "Trichoderma harzianum"],
    },
    # 10. Potato - Preventative / Tuber Health
    {
        "id": "eval-potato-03",
        "crop": "potato",
        "disease": "healthy",
        "pathogen": "None",
        "query": "What preventive cultural and organic practices protect potato tubers from diseases and greening?",
        "ground_truth_answer": (
            "Preventive potato tuber health relies on planting certified disease-free seed tubers, bio-fertilizer "
            "inoculation, and proper soil hilling to prevent tuber exposure to sunlight greening and down-washing blight spores."
        ),
        "expected_active_ingredients": ["None"],
        "expected_organic_remedies": ["Certified disease-free seed tubers", "Bio-fertilizer", "Proper hilling"],
    },
    # 11. Apple - Apple Scab
    {
        "id": "eval-apple-01",
        "crop": "apple",
        "disease": "apple scab",
        "pathogen": "Venturia inaequalis",
        "query": "What chemical and organic fungicides treat apple scab caused by Venturia inaequalis?",
        "ground_truth_answer": (
            "Apple scab (Venturia inaequalis) is treated chemically using Captan 50% WP or Difenoconazole 25% EC. "
            "Organic remedies include Sulfur 80% WDG, Potassium bicarbonate, and Bacillus subtilis. "
            "Avoid sulfur applications during bloom to protect honeybees and pollinators."
        ),
        "expected_active_ingredients": ["Captan", "Difenoconazole"],
        "expected_organic_remedies": ["Sulfur", "Potassium bicarbonate", "Bacillus subtilis"],
    },
    # 12. Apple - Black Rot
    {
        "id": "eval-apple-02",
        "crop": "apple",
        "disease": "black rot",
        "pathogen": "Botryosphaeria obtusa",
        "query": "How is apple black rot managed and what active ingredients are used?",
        "ground_truth_answer": (
            "Apple black rot (Botryosphaeria obtusa) is treated chemically with Thiophanate-methyl 70% WP or "
            "Captan 50% WP. Organic treatments include Copper octanoate, Bordeaux mixture 1%, and Trichoderma harzianum, "
            "paired with pruning mummified fruit and dead wood in winter."
        ),
        "expected_active_ingredients": ["Thiophanate-methyl", "Captan"],
        "expected_organic_remedies": ["Copper octanoate", "Bordeaux mixture", "Trichoderma harzianum"],
    },
    # 13. Apple - Cedar Apple Rust
    {
        "id": "eval-apple-03",
        "crop": "apple",
        "disease": "cedar apple rust",
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "query": "What treatments and fungicides are recommended for cedar apple rust on apple trees?",
        "ground_truth_answer": (
            "Cedar apple rust is treated with Myclobutanil 20% WP or Mancozeb 75% WP. "
            "Organic controls include liquid copper fungicide and elemental sulfur sprays, while removing nearby "
            "eastern red cedar alternate hosts within 1-2 miles."
        ),
        "expected_active_ingredients": ["Myclobutanil", "Mancozeb"],
        "expected_organic_remedies": ["Liquid copper fungicide", "Elemental sulfur"],
    },
    # 14. Corn - Northern Corn Leaf Blight
    {
        "id": "eval-corn-01",
        "crop": "corn",
        "disease": "northern corn leaf blight",
        "pathogen": "Exserohilum turcicum",
        "query": "What fungicides and biological treatments manage Northern Corn Leaf Blight?",
        "ground_truth_answer": (
            "Northern Corn Leaf Blight (Exserohilum turcicum) is treated with Propiconazole 25% EC or "
            "Azoxystrobin 23% SC. Organic/biological options include Trichoderma viride seed treatments and "
            "Bacillus subtilis foliar sprays, alongside avoiding continuous corn monoculture."
        ),
        "expected_active_ingredients": ["Propiconazole", "Azoxystrobin"],
        "expected_organic_remedies": ["Trichoderma viride", "Bacillus subtilis"],
    },
    # 15. Corn - Gray Leaf Spot
    {
        "id": "eval-corn-02",
        "crop": "corn",
        "disease": "gray leaf spot",
        "pathogen": "Cercospora zeae-maydis",
        "query": "How do you control gray leaf spot in corn fields and what active ingredients work?",
        "ground_truth_answer": (
            "Gray leaf spot (Cercospora zeae-maydis) in corn is controlled using Azoxystrobin + Propiconazole (Quilt Xcel) "
            "or Pyraclostrobin 250 EC. Organic management includes Bacillus amyloliquefaciens foliar sprays and rotating "
            "crops out of corn for 1-2 years."
        ),
        "expected_active_ingredients": ["Azoxystrobin", "Propiconazole", "Pyraclostrobin"],
        "expected_organic_remedies": ["Bacillus amyloliquefaciens", "Bio-fungicide", "Crop rotation"],
    },
    # 16. Corn - Common Rust
    {
        "id": "eval-corn-03",
        "crop": "corn",
        "disease": "common rust",
        "pathogen": "Puccinia sorghi",
        "query": "What fungicides and organic remedies are effective against common rust in corn?",
        "ground_truth_answer": (
            "Common rust (Puccinia sorghi) in corn is treated with Mancozeb 75% WP or Pyraclostrobin 20% WG. "
            "Organic treatment includes liquid copper octanoate and compost extract bio-sprays applied at the "
            "first sign of pustules on the upper canopy."
        ),
        "expected_active_ingredients": ["Mancozeb", "Pyraclostrobin"],
        "expected_organic_remedies": ["Copper octanoate", "Compost extract"],
    },
    # 17. Grape - Black Rot
    {
        "id": "eval-grape-01",
        "crop": "grape",
        "disease": "black rot",
        "pathogen": "Guignardia bidwellii",
        "query": "What fungicides control grape black rot and what organic sprays are recommended?",
        "ground_truth_answer": (
            "Grape black rot (Guignardia bidwellii) is controlled using Mancozeb 75% WP or Myclobutanil 20% WP. "
            "Organic remedies include Copper hydroxide 53.8% WG and lime sulfur dormant sprays, removing "
            "mummified berries and infected canes during winter pruning."
        ),
        "expected_active_ingredients": ["Mancozeb", "Myclobutanil"],
        "expected_organic_remedies": ["Copper hydroxide", "Lime sulfur"],
    },
    # 18. Grape - Esca (Black Measles)
    {
        "id": "eval-grape-02",
        "crop": "grape",
        "disease": "esca (black measles)",
        "pathogen": "Fomitiporia mediterranea / Phaeomoniella",
        "query": "How is grape esca or black measles disease treated and prevented in vineyards?",
        "ground_truth_answer": (
            "Grape esca (black measles) is managed with Thiophanate-methyl wound paste or Fosetyl-Aluminium 80% WP. "
            "Organic prevention involves pruning wound sealants containing Trichoderma atroviride, pruning strictly "
            "in dry weather, and disinfecting shears between vines with 70% isopropyl alcohol."
        ),
        "expected_active_ingredients": ["Thiophanate-methyl", "Fosetyl-Aluminium"],
        "expected_organic_remedies": ["Trichoderma atroviride", "Pruning wound sealants"],
    },
    # 19. Orange - Citrus Greening (Huanglongbing)
    {
        "id": "eval-orange-01",
        "crop": "orange",
        "disease": "citrus greening (huanglongbing)",
        "pathogen": "Candidatus Liberibacter asiaticus",
        "query": "What vector control and foliar treatments manage citrus greening (Huanglongbing) in oranges?",
        "ground_truth_answer": (
            "Citrus greening (Huanglongbing) is managed by controlling the Asian citrus psyllid vector using "
            "systemic insecticides like Imidacloprid 17.8% SL or Thiamethoxam 25% WG, combined with foliar Zn/Mn/Fe nutrition. "
            "Organic controls include 1% horticultural mineral oil, kaolin clay barriers, and Tamarixia radiata parasitoid wasps."
        ),
        "expected_active_ingredients": ["Imidacloprid", "Thiamethoxam"],
        "expected_organic_remedies": ["Horticultural mineral oil", "Kaolin clay", "Tamarixia radiata"],
    },
    # 20. Pepper - Bacterial Spot
    {
        "id": "eval-pepper-01",
        "crop": "bell pepper",
        "disease": "bacterial spot",
        "pathogen": "Xanthomonas campestris",
        "query": "What chemical tank mixes and bio-fungicides control bacterial spot on bell peppers?",
        "ground_truth_answer": (
            "Bacterial spot on bell peppers is controlled with a synergistic tank mix of Fixed Copper + Mancozeb 75% WP. "
            "Organic control uses Copper hydroxide 53.8% WG combined with Bacillus amyloliquefaciens, "
            "avoiding field work when foliage is wet and disinfecting seedling trays."
        ),
        "expected_active_ingredients": ["Fixed Copper", "Mancozeb"],
        "expected_organic_remedies": ["Copper hydroxide", "Bacillus amyloliquefaciens"],
    },
]

# ---------------------------------------------------------------------------
# Metric Calculation Engines
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "no", "nor", "not", "only",
    "own", "same", "too", "very", "s", "t", "can", "will", "don", "should", "now",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased alpha-numeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _extract_content_tokens(text: str) -> set[str]:
    """Extract content words excluding common stopwords."""
    tokens = _tokenize(text)
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _normalize_phrase(phrase: str) -> list[str]:
    """Extract keywords and sub-terms from an ingredient or remedy phrase."""
    # Remove percentage dosages like '53.8%', '75% WP', '1%' etc.
    cleaned = re.sub(r"\d+(\.\d+)?%", "", phrase)
    cleaned = re.sub(r"\b(wp|ec|sc|wg|wdg|sl|od|sp)\b", "", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"[/+()]|\bor\b|\band\b", cleaned)
    terms = []
    for part in parts:
        part_clean = part.strip()
        if part_clean and part_clean.lower() not in ("none", "null"):
            terms.append(part_clean.lower())
    if not terms and phrase.strip() and phrase.strip().lower() not in ("none", "null"):
        terms.append(phrase.strip().lower())
    return terms


def _is_item_in_text(item: str, text: str) -> bool:
    """Check if an expected active ingredient or remedy is present in text."""
    if not item or item.strip().lower() in ("none", "null"):
        return True

    text_lower = text.lower()
    item_lower = item.lower()

    # Exact full match or substring match
    if item_lower in text_lower:
        return True

    # Check normalized sub-terms (e.g. Captan in 'Captan 50% WP')
    sub_terms = _normalize_phrase(item)
    for term in sub_terms:
        if term in text_lower:
            return True
        # Content token coverage for longer phrases like 'Bacillus amyloliquefaciens strain D747'
        term_tokens = _extract_content_tokens(term)
        if term_tokens:
            text_tokens = set(_tokenize(text))
            if term_tokens.issubset(text_tokens) or (
                len(term_tokens.intersection(text_tokens)) / len(term_tokens) >= 0.60
            ):
                return True

    return False


def _get_chunk_text(chunk: Any) -> str:
    """Extract raw text from either a string, dict, or RetrievedChunk object."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("metadata", {}).get("text") or "")
    if hasattr(chunk, "text"):
        return str(chunk.text)
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        return str(chunk.metadata.get("text", ""))
    return str(chunk)


def compute_context_recall(
    expected_active_ingredients: Sequence[str],
    expected_organic_remedies: Sequence[str],
    retrieved_chunks: Sequence[Any],
    disease: str = "",
) -> tuple[float, list[str], list[str]]:
    """Compute Context Recall (0.0 - 1.0).

    Checks whether ground-truth active ingredients and organic remedies
    are present in the concatenated retrieved chunks.

    Returns:
        (recall_score, matched_items, missing_items)
    """
    chunks_text = " ".join(_get_chunk_text(c) for c in retrieved_chunks)

    expected_items: list[str] = []
    for item in list(expected_active_ingredients) + list(expected_organic_remedies):
        if item and item.strip().lower() not in ("none", "null"):
            expected_items.append(item.strip())

    if not expected_items:
        # If no specific chemicals/remedies required (e.g. purely preventative/healthy),
        # verify if disease/crop topic is retrieved in context.
        if disease and disease.lower() != "healthy":
            disease_found = disease.lower() in chunks_text.lower()
            return (1.0 if disease_found else 0.0, [disease] if disease_found else [], [] if disease_found else [disease])
        return 1.0, [], []

    if not retrieved_chunks or not chunks_text.strip():
        return 0.0, [], expected_items

    matched: list[str] = []
    missing: list[str] = []

    for item in expected_items:
        if _is_item_in_text(item, chunks_text):
            matched.append(item)
        else:
            missing.append(item)

    score = len(matched) / len(expected_items) if expected_items else 1.0
    return max(0.0, min(1.0, float(score))), matched, missing


def compute_context_precision(
    expected_active_ingredients: Sequence[str],
    expected_organic_remedies: Sequence[str],
    disease: str,
    retrieved_chunks: Sequence[Any],
) -> float:
    """Compute Context Precision (0.0 - 1.0) using Mean Average Precision @ K (MAP@K).

    Evaluates the rank-weighted quality of retrieved chunks. Chunks containing
    expected ingredients, remedies, or disease-specific pathogen facts are marked relevant.
    """
    if not retrieved_chunks:
        return 0.0

    target_items: list[str] = [
        item for item in list(expected_active_ingredients) + list(expected_organic_remedies)
        if item and item.strip().lower() not in ("none", "null")
    ]
    if disease and disease.strip().lower() not in ("none", "healthy"):
        target_items.append(disease.strip())

    relevance_flags: list[bool] = []
    for chunk in retrieved_chunks:
        c_text = _get_chunk_text(chunk)
        is_rel = False
        if not target_items:
            # Preventative / healthy scenario: non-empty chunks are considered relevant
            is_rel = len(c_text.strip()) > 0
        else:
            for item in target_items:
                if _is_item_in_text(item, c_text):
                    is_rel = True
                    break
        relevance_flags.append(is_rel)

    num_relevant = sum(1 for r in relevance_flags if r)
    if num_relevant == 0:
        return 0.0

    # Calculate Average Precision at K
    cumulative_relevant = 0
    precision_sum = 0.0
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            cumulative_relevant += 1
            precision_at_k = cumulative_relevant / rank
            precision_sum += precision_at_k

    ap_score = precision_sum / num_relevant
    return max(0.0, min(1.0, float(ap_score)))


def compute_faithfulness(
    answer: str,
    retrieved_chunks: Sequence[Any],
) -> tuple[float, int, int]:
    """Compute Faithfulness Score (0.0 - 1.0).

    Evaluates whether factual statements / claims made in the generated answer
    are grounded in the retrieved context chunks.

    Returns:
        (faithfulness_score, supported_claims, total_claims)
    """
    if not answer or not answer.strip():
        return 0.0, 0, 0

    chunks_text = " ".join(_get_chunk_text(c) for c in retrieved_chunks)
    if not retrieved_chunks or not chunks_text.strip():
        return 0.0, 0, 1

    # Split answer into sentence-level claims
    raw_sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    claims: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        # Filter out trivial boilerplates and very short conversational tokens
        if len(s_clean) > 15 and not s_clean.lower().startswith(("hello", "hi there", "sure", "here is", "based on")):
            claims.append(s_clean)

    if not claims:
        # If single sentence or short answer
        claims = [answer.strip()]

    context_tokens = set(_tokenize(chunks_text))
    supported_count = 0

    for claim in claims:
        claim_content = _extract_content_tokens(claim)
        if not claim_content:
            supported_count += 1
            continue

        overlap = claim_content.intersection(context_tokens)
        overlap_ratio = len(overlap) / len(claim_content)

        # A claim is supported if >= 50% of its content terms are present in context
        # or if an exact sub-phrase match is present
        if overlap_ratio >= 0.50 or claim.lower() in chunks_text.lower():
            supported_count += 1

    score = supported_count / len(claims) if claims else 1.0
    return max(0.0, min(1.0, float(score))), supported_count, len(claims)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cos_sim = dot / (norm_a * norm_b)
    # Scale from [-1.0, 1.0] to [0.0, 1.0]
    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))


def compute_answer_relevance(
    query: str,
    answer: str,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> float:
    """Compute Answer Relevance (0.0 - 1.0).

    Computes semantic similarity between query and generated answer.
    Uses sentence embedding cosine similarity when embed_fn is available,
    falling back to lexical/token alignment.
    """
    if not query.strip() or not answer.strip():
        return 0.0

    # 1. Try embedding similarity if embed_fn is provided or importable
    if embed_fn is not None:
        try:
            q_vec = embed_fn(query)
            a_vec = embed_fn(answer)
            return _cosine_similarity(q_vec, a_vec)
        except Exception as exc:
            logger.debug("Embedding similarity failed, falling back to lexical: %s", exc)

    # Try embedding_service if in backend environment
    try:
        from app.services.embedding_service import embed_query

        q_vec = embed_query(query)
        a_vec = embed_query(answer)
        return _cosine_similarity(q_vec, a_vec)
    except Exception:
        pass

    # 2. Robust Token & Semantic Alignment Fallback
    q_tokens = _extract_content_tokens(query)
    a_tokens = _extract_content_tokens(answer)

    if not q_tokens or not a_tokens:
        return 0.5

    overlap = q_tokens.intersection(a_tokens)
    query_coverage = len(overlap) / len(q_tokens)
    jaccard = len(overlap) / len(q_tokens.union(a_tokens))

    # Bonus for answering length & substantive structure
    answer_length_factor = min(1.0, len(a_tokens) / 10.0)

    score = 0.60 * query_coverage + 0.25 * jaccard + 0.15 * answer_length_factor
    return max(0.0, min(1.0, float(score)))


def compute_harmonic_composite(
    faithfulness: float | None,
    context_recall: float,
    context_precision: float,
    answer_relevance: float | None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute Harmonic Composite RAG Score (0.0 - 1.0).

    Blends available metric dimensions into an overall RAG scorecard index.
    In retrieval-only mode (when faithfulness & answer_relevance are None),
    computes the composite of Context Recall and Context Precision.
    """
    if faithfulness is not None and answer_relevance is not None:
        # Full end-to-end RAG mode
        default_weights = {
            "faithfulness": 0.30,
            "context_recall": 0.25,
            "context_precision": 0.25,
            "answer_relevance": 0.20,
        }
        w = weights or default_weights
        weighted_arithmetic = (
            w["faithfulness"] * faithfulness
            + w["context_recall"] * context_recall
            + w["context_precision"] * context_precision
            + w["answer_relevance"] * answer_relevance
        )
        # Apply slight penalty if any core dimension is 0.0 (harmonic sensitivity)
        scores = [faithfulness, context_recall, context_precision, answer_relevance]
        eps = 1e-4
        harmonic_mean = len(scores) / sum(1.0 / (s + eps) for s in scores)
        composite = 0.70 * weighted_arithmetic + 0.30 * harmonic_mean
        return max(0.0, min(1.0, float(composite)))
    else:
        # Retrieval-only mode (--no-llm)
        w_recall = 0.50
        w_precision = 0.50
        weighted_arithmetic = w_recall * context_recall + w_precision * context_precision
        eps = 1e-4
        harmonic_mean = 2.0 / ((1.0 / (context_recall + eps)) + (1.0 / (context_precision + eps)))
        composite = 0.70 * weighted_arithmetic + 0.30 * harmonic_mean
        return max(0.0, min(1.0, float(composite)))


# ---------------------------------------------------------------------------
# Data Models for Results & Reporting
# ---------------------------------------------------------------------------

@dataclass
class ItemEvalResult:
    id: str
    crop: str
    disease: str
    query: str
    ground_truth_answer: str
    retrieved_chunks_count: int
    context_recall: float
    context_precision: float
    faithfulness: float | None
    answer_relevance: float | None
    composite_score: float
    latency_sec: float
    matched_ingredients: list[str] = field(default_factory=list)
    missing_ingredients: list[str] = field(default_factory=list)
    generated_answer: str = ""
    retrieved_chunk_snippets: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    timestamp: str
    total_queries: int
    crops_evaluated: list[str]
    retrieval_mode: str
    llm_enabled: bool
    mean_context_recall: float
    mean_context_precision: float
    mean_faithfulness: float | None
    mean_answer_relevance: float | None
    mean_composite_score: float
    mean_latency_sec: float
    quality_gate_passed: bool
    item_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval & Pipeline Execution Helpers
# ---------------------------------------------------------------------------

def _load_vector_store() -> Any:
    """Load default FAISS vector store if available."""
    try:
        from app.core.exceptions import VectorStoreNotFoundError
        from app.services.faiss_vector_store import (
            DEFAULT_INDEX_PATH,
            DEFAULT_METADATA_PATH,
            FAISSVectorStore,
        )

        store = FAISSVectorStore(
            index_path=DEFAULT_INDEX_PATH,
            metadata_path=DEFAULT_METADATA_PATH,
        )
        store.load()
        return store
    except Exception as exc:
        logger.warning("Could not load FAISS vector store (%s). Using fallback retriever.", exc)
        return None


def execute_retrieval(
    query: str,
    top_k: int = 5,
    crop: str | None = None,
    vector_store: Any = None,
    hybrid: bool = True,
    rerank_flag: bool = True,
) -> list[Any]:
    """Execute hybrid retrieval with cross-encoder reranking."""
    if vector_store is not None:
        try:
            from app.services.retrieval_service import retrieve

            chunks = retrieve(
                query=query,
                vector_store=vector_store,
                top_k=top_k,
                collection=crop,
                rerank=rerank_flag,
            )
            if chunks:
                return chunks
        except Exception as exc:
            # Module 10 gap-closure: this was previously `logger.debug`,
            # which let a real bug (retrieve()'s actual keyword is `rerank`,
            # not `rerank_candidates` -- a TypeError on every single call)
            # go completely unnoticed for this script's entire history,
            # silently degrading every retrieval to the raw-vector fallback
            # below (no hybrid BM25, no collection filter, no reranking).
            # Elevated to `warning` so a real retrieval failure is never
            # invisible again -- fail loud, not silently degraded.
            logger.warning("Live retrieve() call failed: %s. Falling back to direct search.", exc)
            if hasattr(vector_store, "search"):
                try:
                    from app.services.embedding_service import embed_query

                    q_vec = embed_query(query)
                    return vector_store.search(q_vec, top_k)
                except Exception:
                    pass

    # Direct document corpus fallback for standalone evaluation without active server
    docs_dir = _PROJECT_ROOT / "data" / "plant_disease_docs"
    fallback_chunks: list[str] = []
    if docs_dir.exists():
        target_crop_dir = docs_dir / crop if crop else docs_dir
        if target_crop_dir.exists():
            for p in target_crop_dir.glob("**/*.md"):
                try:
                    content = p.read_text(encoding="utf-8")
                    fallback_chunks.append(content[:1000])
                except Exception:
                    pass

        # Also check treatment dosage matrix
        matrix_path = docs_dir / "treatment_dosage_matrix.csv"
        if matrix_path.exists():
            try:
                matrix_text = matrix_path.read_text(encoding="utf-8")
                fallback_chunks.append(matrix_text)
            except Exception:
                pass

    return fallback_chunks[:top_k]


def execute_answer_generation(
    query: str,
    retrieved_chunks: list[Any],
    chat_service: Any = None,
) -> str:
    """Generate answer from retrieved context chunks using LLM or structured prompt."""
    if chat_service is not None:
        try:
            resp = chat_service.handle_query(query)
            return resp.answer
        except Exception as exc:
            logger.debug("chat_service.handle_query failed: %s", exc)

    # Try LLM client directly
    try:
        from app.services.llm_provider import build_llm_client
        from app.services.prompt_builder import build_prompt

        client = build_llm_client()
        context_chunks = retrieved_chunks
        prompt = build_prompt(query, context_chunks)
        return client.generate(prompt)
    except Exception as exc:
        logger.debug("LLM generation unavailable (%s). Constructing context synthesis.", exc)

    # Fallback synthesizer: combine informative sections from chunks
    chunks_text = "\n\n".join(_get_chunk_text(c) for c in retrieved_chunks)
    if chunks_text:
        return f"Based on the plant pathology documents:\n{chunks_text[:500]}..."
    return "No sufficient context available in document repository."


# ---------------------------------------------------------------------------
# Core Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    no_llm: bool = False,
    output_path: Path | str | None = None,
    top_k: int = 5,
    hybrid: bool = True,
    rerank_flag: bool = True,
) -> BenchmarkReport:
    """Run quantitative evaluation across the evaluation dataset."""
    eval_data = dataset or GOLDEN_DATASET
    if limit is not None and limit > 0:
        eval_data = eval_data[:limit]

    vector_store = _load_vector_store()
    chat_service = None
    if not no_llm:
        try:
            from app.services.llm_provider import build_llm_client
            from app.services.rag_service import ChatService

            llm_client = build_llm_client()
            if vector_store is not None:
                chat_service = ChatService(vector_store=vector_store, llm_client=llm_client)
        except Exception:
            chat_service = None

    item_results: list[ItemEvalResult] = []
    crops_seen: set[str] = set()

    for entry in eval_data:
        item_id = entry.get("id", f"eval-{len(item_results)+1:02d}")
        crop = entry.get("crop", "general")
        disease = entry.get("disease", "")
        query = entry.get("query", "")
        gt_answer = entry.get("ground_truth_answer", "")
        exp_active = entry.get("expected_active_ingredients", [])
        exp_organic = entry.get("expected_organic_remedies", [])

        crops_seen.add(crop)

        start_time = time.perf_counter()

        # 1. Retrieve Chunks
        retrieved_chunks = execute_retrieval(
            query=query,
            top_k=top_k,
            crop=crop,
            vector_store=vector_store,
            hybrid=hybrid,
            rerank_flag=rerank_flag,
        )

        # 2. Compute Retrieval Metrics
        c_recall, matched, missing = compute_context_recall(
            expected_active_ingredients=exp_active,
            expected_organic_remedies=exp_organic,
            retrieved_chunks=retrieved_chunks,
            disease=disease,
        )

        c_precision = compute_context_precision(
            expected_active_ingredients=exp_active,
            expected_organic_remedies=exp_organic,
            disease=disease,
            retrieved_chunks=retrieved_chunks,
        )

        # 3. Answer Generation & Evaluation
        generated_answer = ""
        faithfulness: float | None = None
        relevance: float | None = None

        if not no_llm:
            generated_answer = execute_answer_generation(
                query=query,
                retrieved_chunks=retrieved_chunks,
                chat_service=chat_service,
            )
            f_score, _, _ = compute_faithfulness(generated_answer, retrieved_chunks)
            faithfulness = f_score
            relevance = compute_answer_relevance(query, generated_answer)

        latency = time.perf_counter() - start_time

        composite = compute_harmonic_composite(
            faithfulness=faithfulness,
            context_recall=c_recall,
            context_precision=c_precision,
            answer_relevance=relevance,
        )

        snippets = [_get_chunk_text(c)[:150].replace("\n", " ") + "..." for c in retrieved_chunks[:3]]

        res = ItemEvalResult(
            id=item_id,
            crop=crop,
            disease=disease,
            query=query,
            ground_truth_answer=gt_answer,
            retrieved_chunks_count=len(retrieved_chunks),
            context_recall=c_recall,
            context_precision=c_precision,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            composite_score=composite,
            latency_sec=latency,
            matched_ingredients=matched,
            missing_ingredients=missing,
            generated_answer=generated_answer,
            retrieved_chunk_snippets=snippets,
        )
        item_results.append(res)

    # Compute Summary Aggregates
    n = len(item_results)
    mean_recall = sum(r.context_recall for r in item_results) / n if n else 0.0
    mean_precision = sum(r.context_precision for r in item_results) / n if n else 0.0
    mean_latency = sum(r.latency_sec for r in item_results) / n if n else 0.0
    mean_composite = sum(r.composite_score for r in item_results) / n if n else 0.0

    mean_faithfulness: float | None = None
    mean_relevance: float | None = None
    if not no_llm:
        faith_vals = [r.faithfulness for r in item_results if r.faithfulness is not None]
        rel_vals = [r.answer_relevance for r in item_results if r.answer_relevance is not None]
        mean_faithfulness = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0
        mean_relevance = sum(rel_vals) / len(rel_vals) if rel_vals else 0.0

    # Quality Gate Threshold Check
    gate_passed = mean_composite >= 0.70 and mean_recall >= 0.70

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_queries=n,
        crops_evaluated=sorted(crops_seen),
        retrieval_mode="Hybrid RRF + Cross-Encoder" if hybrid and rerank_flag else "Standard Semantic",
        llm_enabled=not no_llm,
        mean_context_recall=round(mean_recall, 4),
        mean_context_precision=round(mean_precision, 4),
        mean_faithfulness=round(mean_faithfulness, 4) if mean_faithfulness is not None else None,
        mean_answer_relevance=round(mean_relevance, 4) if mean_relevance is not None else None,
        mean_composite_score=round(mean_composite, 4),
        mean_latency_sec=round(mean_latency, 4),
        quality_gate_passed=gate_passed,
        item_results=[asdict(r) for r in item_results],
    )

    # Save detailed JSON report
    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_REPORT_PATH
    save_eval_report(report, out_file)

    return report


def save_eval_report(report: BenchmarkReport, output_path: Path | str) -> Path:
    """Serialize and write detailed evaluation report to disk."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    report_dict = asdict(report)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Evaluation report saved to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Executive ASCII Scorecard Formatting
# ---------------------------------------------------------------------------

def format_ascii_scorecard(report: BenchmarkReport) -> str:
    """Format an executive ASCII benchmark scorecard table."""
    lines: list[str] = []
    w = 88

    lines.append("=" * w)
    lines.append(f"{'AGROSENSE-RAG QUANTITATIVE BENCHMARK SCORECARD':^{w}}")
    lines.append(f"{'Hybrid RRF Retrieval & Answer Generation Evaluation':^{w}}")
    lines.append("=" * w)

    # Configuration Header
    lines.append(f" Timestamp:        {report.timestamp}")
    lines.append(f" Retrieval Engine: {report.retrieval_mode}")
    lines.append(f" LLM Generation:   {'Enabled (Full E2E RAG)' if report.llm_enabled else 'Disabled (Retrieval-Only Fast CI)'}")
    lines.append(f" Crops Evaluated:  {', '.join(report.crops_evaluated)}")
    lines.append(f" Total Test Cases: {report.total_queries}")
    lines.append("-" * w)

    # Scorecard Table
    if report.llm_enabled:
        headers = f"{'ID':<13}{'Crop':<12}{'Recall':>9}{'Precision':>11}{'Faithful':>10}{'Relevance':>11}{'Composite':>11}{'Latency':>9}"
    else:
        headers = f"{'ID':<15}{'Crop':<15}{'Disease Target':<25}{'Recall':>9}{'Precision':>11}{'Composite':>11}"

    lines.append(headers)
    lines.append("-" * w)

    for item in report.item_results:
        item_id = item["id"]
        crop = item["crop"]
        c_rec = f"{item['context_recall']:.3f}"
        c_prec = f"{item['context_precision']:.3f}"
        comp = f"{item['composite_score']:.3f}"

        if report.llm_enabled:
            faith = f"{item['faithfulness']:.3f}" if item["faithfulness"] is not None else "N/A"
            rel = f"{item['answer_relevance']:.3f}" if item["answer_relevance"] is not None else "N/A"
            lat = f"{item['latency_sec']:.2f}s"
            row = f"{item_id:<13}{crop:<12}{c_rec:>9}{c_prec:>11}{faith:>10}{rel:>11}{comp:>11}{lat:>9}"
        else:
            disease = item["disease"][:23]
            row = f"{item_id:<15}{crop:<15}{disease:<25}{c_rec:>9}{c_prec:>11}{comp:>11}"

        lines.append(row)

    lines.append("=" * w)
    lines.append(f"{'EXECUTIVE AGGREGATE SUMMARY':^{w}}")
    lines.append("=" * w)

    lines.append(f" * Mean Context Recall:     {report.mean_context_recall:.4f}  [Target >= 0.80]")
    lines.append(f" * Mean Context Precision:  {report.mean_context_precision:.4f}  [Target >= 0.70]")
    if report.llm_enabled:
        f_val = f"{report.mean_faithfulness:.4f}" if report.mean_faithfulness is not None else "N/A"
        r_val = f"{report.mean_answer_relevance:.4f}" if report.mean_answer_relevance is not None else "N/A"
        lines.append(f" * Mean Faithfulness:       {f_val}  [Target >= 0.80]")
        lines.append(f" * Mean Answer Relevance:   {r_val}  [Target >= 0.75]")
    lines.append(f" * Harmonic Composite RAG:  {report.mean_composite_score:.4f}  [Target >= 0.75]")
    lines.append(f" * Mean Latency:            {report.mean_latency_sec:.4f}s")

    status_str = "PASSED (Quality Gate Met)" if report.quality_gate_passed else "ATTENTION (Below Target Threshold)"
    lines.append(f" * Benchmark Status:        {status_str}")
    lines.append("=" * w)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quantitative evaluation benchmark for AgroSense-RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N test cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Path to output JSON evaluation report.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run retrieval-only evaluation without LLM generation for fast CI runs.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to custom JSON dataset file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve per query.",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable BM25 lexical fusion (use dense semantic vector search only).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Cross-Encoder neural re-ranking stage.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    dataset: list[dict[str, Any]] | None = None
    if args.dataset is not None:
        p = Path(args.dataset)
        if not p.exists():
            print(f"Error: Dataset file not found at {p}", file=sys.stderr)
            return 1
        with p.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    report = run_evaluation(
        dataset=dataset,
        limit=args.limit,
        no_llm=args.no_llm,
        output_path=args.output,
        top_k=args.top_k,
        hybrid=not args.no_hybrid,
        rerank_flag=not args.no_rerank,
    )

    # Print executive ASCII scorecard table
    print("\n" + format_ascii_scorecard(report) + "\n")
    print(f"Detailed JSON report written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
