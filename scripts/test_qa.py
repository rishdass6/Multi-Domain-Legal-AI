import json
import sys
from app.retriever import Retriever
from app.qa_model import QAModel

# ── Test Questions ────────────────────────────────────────────────────────────
# Covering: contract law, tort law, IP, employment, general legal definitions.
# Mix of definitional, procedural, and element-based questions.

TEST_QUESTIONS = [
    {
        "question": "What losses must the indemnifying party cover?",
        "category": "Indemnification",
        "expected_keywords": ["claims", "damages", "losses", "costs", "expenses", "liabilities"]
    },
    {
        "question": "Who shall indemnify and hold harmless the other party?",
        "category": "Indemnification",
        "expected_keywords": ["indemnif", "shall", "party", "licensee", "licensor", "company"]
    },
    {
        "question": "What is the maximum aggregate liability under this agreement?",
        "category": "Limitation of Liability",
        "expected_keywords": ["aggregate", "exceed", "fees", "paid", "months", "total"]
    },
    {
        "question": "Are indirect or consequential damages excluded?",
        "category": "Limitation of Liability",
        "expected_keywords": ["indirect", "consequential", "incidental", "excluded", "liable"]
    },
    {
        "question": "What written notice is required before termination?",
        "category": "Termination",
        "expected_keywords": ["days", "notice", "written", "prior", "30", "60", "90"]
    },
    {
        "question": "Under what circumstances may a party terminate for cause?",
        "category": "Termination",
        "expected_keywords": ["breach", "default", "cure", "material", "terminat"]
    },
    {
        "question": "What constitutes a force majeure event under this agreement?",
        "category": "Force Majeure",
        "expected_keywords": ["war", "flood", "fire", "act of god", "beyond", "control", "strike"]
    },
    {
        "question": "What information is deemed confidential under this agreement?",
        "category": "Confidentiality",
        "expected_keywords": ["confidential", "proprietary", "information", "disclose"]
    },
    {
        "question": "How long does the confidentiality obligation survive termination?",
        "category": "Confidentiality",
        "expected_keywords": ["year", "years", "period", "survive", "termination"]
    },
    {
        "question": "Who shall own all intellectual property developed under this agreement?",
        "category": "Intellectual Property",
        "expected_keywords": ["own", "sole", "exclusive", "assign", "company", "licensee"]
    },
    {
        "question": "What rights are licensed to the licensee under this agreement?",
        "category": "Intellectual Property",
        "expected_keywords": ["non-exclusive", "royalty", "license", "right", "use"]
    },
    {
        "question": "The laws of which state shall govern this agreement?",
        "category": "Governing Law",
        "expected_keywords": ["state", "new york", "california", "delaware", "laws"]
    },
    {
        "question": "What business activities is the party prohibited from engaging in?",
        "category": "Non-Compete",
        "expected_keywords": ["compet", "business", "prohibit", "engage", "restrict", "solicit"]
    },
    {
        "question": "When are payments due under this agreement?",
        "category": "Payment",
        "expected_keywords": ["days", "invoice", "due", "payment", "net", "receipt"]
    },
    {
        "question": "How shall disputes between the parties be resolved?",
        "category": "Dispute Resolution",
        "expected_keywords": ["arbitrat", "mediat", "court", "jurisdiction", "resolv", "proceeding"]
    },
]


# ── Test Runner ───────────────────────────────────────────────────────────────

def keyword_check(answer: str, keywords: list) -> bool:
    """Check if any expected keyword appears in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def run_tests():
    print("=" * 70)
    print("PHASE 4: END-TO-END QA TEST")
    print("=" * 70)

    # Load components
    print("\nLoading Retriever...")
    retriever = Retriever()

    print("Loading QA Model...")
    qa_model = QAModel()

    print(f"\nRunning {len(TEST_QUESTIONS)} test questions...\n")

    results = []
    passed = 0
    fallbacks = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        category = test["category"]
        expected_keywords = test["expected_keywords"]

        print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {category}")
        print(f"       Q: {question}")

        # Step 1: Retrieve
        chunks = retriever.search(question, top_k=3, domain_filter="legal")

        # Step 2: Answer
        result = qa_model.answer(question, chunks)

        answer = result["answer"]
        confidence = result["confidence"]
        level = result["confidence_meta"]["level"]
        is_fallback = result["is_fallback"]
        top_source = chunks[0]["source"] if chunks else "none"

        # Evaluate
        keyword_hit = keyword_check(answer, expected_keywords) and not is_fallback
        if keyword_hit:
            passed += 1
            status = "✓ PASS"
        elif is_fallback:
            fallbacks += 1
            status = "⚠ FALLBACK"
        else:
            status = "✗ MISS"

        print(f"       A: {answer[:200]}")
        print(f"       Confidence: {confidence:.4f} ({level}) | Source: {top_source}")
        print(f"       {status}\n")

        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "confidence": confidence,
            "level": level,
            "is_fallback": is_fallback,
            "keyword_hit": keyword_hit,
            "status": status,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(TEST_QUESTIONS)
    misses = total - passed - fallbacks

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total questions:  {total}")
    print(f"  ✓ Passed:         {passed} ({100*passed//total}%)")
    print(f"  ⚠ Fallbacks:      {fallbacks} ({100*fallbacks//total}%)")
    print(f"  ✗ Misses:         {misses} ({100*misses//total}%)")

    # Category breakdown
    print("\n  By category:")
    from collections import defaultdict
    cat_results = defaultdict(list)
    for r in results:
        cat_results[r["category"]].append(r["status"])
    for cat, statuses in sorted(cat_results.items()):
        cat_pass = sum(1 for s in statuses if "PASS" in s)
        print(f"    {cat:<20} {cat_pass}/{len(statuses)} passed")

    # Confidence distribution
    confidences = [r["confidence"] for r in results if not r["is_fallback"]]
    if confidences:
        print(f"\n  Confidence (answered questions):")
        print(f"    Min:  {min(confidences):.4f}")
        print(f"    Max:  {max(confidences):.4f}")
        print(f"    Mean: {sum(confidences)/len(confidences):.4f}")

    # Save results to JSON for portfolio evidence
    output_path = "data/processed/qa_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {output_path}")
    print("  (Screenshot or include this file in your portfolio README)")

    print("\n✓ Phase 4 complete. QAModel ready for Phase 5 (RAG Pipeline).")


if __name__ == "__main__":
    run_tests()