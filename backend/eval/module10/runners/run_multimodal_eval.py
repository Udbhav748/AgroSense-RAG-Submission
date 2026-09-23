#!/usr/bin/env python
"""Module 10 multimodal (LeafSense vision) evaluation runner.

Usage (from backend/):
    python eval/module10/runners/run_multimodal_eval.py

IMPORTANT SCOPE NOTE: no real, labeled plant-disease photo corpus exists
in this repository or was fetched externally. The images in
eval/module10/assets/ are SYNTHETIC (generated with PIL: simple shapes/
colors, not real photographs) and are used ONLY to test pipeline
ROBUSTNESS on degenerate/edge inputs (a plain gray non-plant image, a
heavily blurred image, a uniform "healthy-looking" green blob, a
spotted "diseased-looking" blob) — NOT to validate LeafSense's
diagnostic ACCURACY, which would require real, correctly-labeled leaf
photos this evaluation does not have. Every result below is labeled
accordingly; do not read a prediction here as evidence LeafSense
correctly identifies real diseases.

This IS a live run: the LeafSense service (POST /predict/insightai,
default http://127.0.0.1:8001) must actually be reachable — confirmed via
app.services.vision_client.is_leafsense_online() before running. If it
is not reachable, results are marked BLOCKED, not fabricated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.exceptions import VisionServiceError  # noqa: E402
from app.services.vision_client import diagnose_image, is_leafsense_online  # noqa: E402
from eval.module10 import config  # noqa: E402

ASSETS_DIR = config.MODULE10_DIR / "assets"

CASES = [
    {
        "id": "mm_001",
        "file": "synthetic_leaf_with_spots.jpg",
        "category": "synthetic_diseased_like",
        "description": "Synthetic green ellipse with brown dot 'spots' and vein lines — simulates gross diseased-leaf structure, not a real disease.",
    },
    {
        "id": "mm_002",
        "file": "synthetic_leaf_low_quality.jpg",
        "category": "low_quality_image",
        "description": "Same synthetic image, downsampled and heavily Gaussian-blurred, saved at quality=15 — simulates a poor-quality photo.",
    },
    {
        "id": "mm_003",
        "file": "synthetic_healthy_leaf.jpg",
        "category": "healthy_leaf_like",
        "description": "Synthetic uniform green ellipse, no spots — simulates a healthy-leaf-like image.",
    },
    {
        "id": "mm_004",
        "file": "synthetic_no_plant.jpg",
        "category": "no_useful_visual_evidence",
        "description": "Plain uniform gray image — no plant structure at all.",
    },
]


def run_case(case: dict) -> dict:
    path = ASSETS_DIR / case["file"]
    image_bytes = path.read_bytes()
    start = time.perf_counter()
    try:
        prediction = diagnose_image(image_bytes, case["file"], "image/jpeg", engine="hybrid")
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "case_id": case["id"],
            "category": case["category"],
            "description": case["description"],
            "success": True,
            "raw_class": prediction.raw_class,
            "crop": prediction.crop,
            "disease": prediction.disease,
            "confidence": prediction.confidence,
            "low_confidence": prediction.low_confidence,
            "engine": prediction.engine,
            "latency_ms": round(latency_ms, 2),
        }
    except VisionServiceError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "case_id": case["id"],
            "category": case["category"],
            "description": case["description"],
            "success": False,
            "error": str(exc),
            "latency_ms": round(latency_ms, 2),
        }


def main() -> None:
    online = is_leafsense_online(force_refresh=True)
    if not online:
        report = {
            "metadata": config.run_metadata(sample_count=0),
            "status": "BLOCKED",
            "reason": "LeafSense service not reachable at settings.vision_service_url "
            "(default http://127.0.0.1:8001). Start the LeafSense service and re-run "
            "this exact command: python eval/module10/runners/run_multimodal_eval.py",
            "missing_dependency": "LeafSense vision service (external process, POST /predict/insightai)",
        }
        path = config.save_report(report, name="multimodal_eval")
        print(f"BLOCKED — LeafSense not reachable. Saved: {path}")
        return

    results = [run_case(case) for case in CASES]
    report = {
        "metadata": config.run_metadata(sample_count=len(CASES)),
        "status": "MEASURED (synthetic images — pipeline robustness only, NOT diagnostic accuracy)",
        "leafsense_online": True,
        "per_case": results,
    }
    path = config.save_report(report, name="multimodal_eval")
    print(f"Saved: {path}")
    print()
    for r in results:
        if r["success"]:
            print(f"  {r['case_id']:<10s} {r['category']:<28s} crop={r['crop']:<12s} disease={r['disease']:<20s} conf={r['confidence']:.3f} low_conf={r['low_confidence']} engine={r['engine']} ({r['latency_ms']}ms)")
        else:
            print(f"  {r['case_id']:<10s} {r['category']:<28s} FAILED: {r['error']} ({r['latency_ms']}ms)")


if __name__ == "__main__":
    main()
