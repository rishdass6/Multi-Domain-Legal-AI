"""
Phase 5 smoke test — runs the full RAGPipeline end to end.
Run with: python -m scripts.test_pipeline
"""

from app.rag_pipeline import RAGPipeline

QUESTIONS = [
    ("What losses must the indemnifying party cover?",         "auto"),
    ("How many days notice is required to terminate?",         "auto"),
    ("What constitutes a force majeure event?",                "legal"),
    ("What information is deemed confidential?",               "auto"),
    ("The laws of which state shall govern this agreement?",   "auto"),
]

def main():
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    pipeline.load_or_build_index()

    print(f"\nPipeline ready: {pipeline.is_ready()}")
    print(f"Index size:     {pipeline.get_index_size()} vectors")
    print(f"Stats:          {pipeline.get_stats()}")
    print("\n" + "=" * 65)

    for question, domain in QUESTIONS:
        result = pipeline.answer(question, domain=domain, top_k=5)
        print(f"Q:      {question}")
        print(f"Domain: {result['domain_detected']}")
        print(f"A:      {result['answer']}")
        print(f"Conf:   {result['confidence']:.4f}")
        print(f"Sources retrieved: {len(result['sources'])}")
        print("-" * 65)

if __name__ == "__main__":
    main()