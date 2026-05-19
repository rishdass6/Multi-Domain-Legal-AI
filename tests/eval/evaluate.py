"""
Evaluation script — computes Exact Match and F1 against the golden dataset.
Saves results to data/processed/eval_results.csv

Run with: python -m tests.eval.evaluate
"""

import csv
import os
import re
import logging
from collections import Counter
from tests.eval.golden_dataset import GOLDEN_DATASET
from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)  # Suppress model load noise during eval

OUTPUT_PATH = "data/processed/eval_results.csv"


# ── Scoring Functions ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_match(prediction: str, keywords: list) -> bool:
    """True if any keyword appears in the normalized prediction."""
    pred_norm = normalize(prediction)
    return any(normalize(kw) in pred_norm for kw in keywords)


def token_f1(prediction: str, keywords: list) -> float:
    """
    Compute token-level F1 between prediction and the best-matching keyword.
    This is the standard SQuAD F1 metric adapted for keyword lists.
    """
    pred_tokens = normalize(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for kw in keywords:
        kw_tokens = normalize(kw).split()
        common = Counter(pred_tokens) & Counter(kw_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall    = num_common / len(kw_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)

    return round(best_f1, 4)


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 65)
    print("LEGAL QA EVALUATION — GOLDEN DATASET")
    print("=" * 65)

    print("\nLoading pipeline...")
    pipeline = RAGPipeline()
    pipeline.load_or_build_index()
    print("Pipeline ready.\n")

    results = []
    exact_matches = 0
    total_f1 = 0.0
    fallbacks = 0

    for i, item in enumerate(GOLDEN_DATASET, 1):
        qid      = item["id"]
        question = item["question"]
        keywords = item["keywords"]

        result = pipeline.answer(question, domain="legal", top_k=5)

        prediction   = result["answer"]
        confidence   = result["confidence"]
        is_fallback  = result.get("is_fallback", False)

        em = keyword_match(prediction, keywords) and not is_fallback
        f1 = token_f1(prediction, keywords) if not is_fallback else 0.0

        if em:
            exact_matches += 1
        if is_fallback:
            fallbacks += 1
        total_f1 += f1

        status = "PASS" if em else ("FALLBACK" if is_fallback else "MISS")

        print(f"[{i:02d}/{len(GOLDEN_DATASET)}] {qid:<15} {status:<10} "
              f"conf={confidence:.4f}  f1={f1:.4f}")
        print(f"         Q: {question}")
        print(f"         A: {prediction[:120]}")
        print()

        results.append({
            "id":           qid,
            "question":     question,
            "prediction":   prediction,
            "keywords":     "|".join(keywords),
            "exact_match":  int(em),
            "f1":           f1,
            "confidence":   confidence,
            "is_fallback":  int(is_fallback),
            "status":       status,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(GOLDEN_DATASET)
    em_pct  = 100 * exact_matches / n
    f1_avg  = total_f1 / n
    fb_pct  = 100 * fallbacks / n

    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total questions:    {n}")
    print(f"  Exact Match (EM):   {exact_matches}/{n}  ({em_pct:.1f}%)")
    print(f"  Average F1:         {f1_avg:.4f}")
    print(f"  Fallbacks:          {fallbacks}/{n}  ({fb_pct:.1f}%)")

    # Category breakdown
    from collections import defaultdict
    cat_em = defaultdict(list)
    for r in results:
        cat = r["id"].split("_")[0]
        cat_em[cat].append(r["exact_match"])

    print("\n  By category:")
    for cat, scores in sorted(cat_em.items()):
        passed = sum(scores)
        print(f"    {cat:<12} {passed}/{len(scores)} passed")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Results saved to {OUTPUT_PATH}")
    print("  Screenshot the summary above for your portfolio.")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    evaluate()