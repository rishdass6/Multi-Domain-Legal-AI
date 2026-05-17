"""
Smoke test for the Retriever class.
Run after build_index.py to confirm end-to-end retrieval works.
"""

from app.retriever import Retriever

TEST_QUERIES = [
    ("What are the elements of negligence?", "legal"),
    ("What is indemnification in a contract?", "legal"),
    ("Define breach of contract", "legal"),
    ("What is intellectual property licensing?", "legal"),
    ("Explain termination clauses", "legal"),
]


def test_retriever():
    print("Loading retriever...")
    retriever = Retriever()

    print(f"\nRetriever stats: {retriever.get_stats()}\n")
    print("=" * 60)

    for query, domain in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print(f"Domain filter: {domain}")
        results = retriever.search(query, top_k=3, domain_filter=domain)
        print(f"Results returned: {len(results)}")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] score={r['score']:.4f} | source={r['source']}")
            print(f"       {r['text'][:200]}...")

    print("\n✓ Retriever test complete. Ready for Phase 4.")


if __name__ == "__main__":
    test_retriever()