import pytest
from app.retriever import Retriever

@pytest.fixture(scope="module")
def retriever():
    return Retriever()

def test_retriever_loads(retriever):
    assert retriever.is_ready()

def test_retriever_returns_results(retriever):
    chunks = retriever.search("indemnification clause", top_k = 5)
    assert len(chunks) > 0

def test_retriever_respects_top_k(retriever):
    for k in [1, 3, 5]:
        chunk = retriever.search("termincation notice", top_k=k)
        assert len(chunk) > 0


def test_retriever_results_have_required_fields(retriever):
    chunks = retriever.search("force majeure events", top_k = 3)
    for r in chunks:
        assert "text" in r
        assert "domain" in r
        assert "score" in r
        assert "domain" in r
        assert "chunk_id" in r

def test_retriever_scores_are_valid(retriever):
    results = retriever.search("governing law", top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0

def test_retriever_scores_descending(retriever):
    results = retriever.search("confidentiality obligation", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_retriever_domain_filter(retriever):
    results = retriever.search("liability damages", top_k=5, domain_filter="legal")
    for r in results:
        assert r["domain"] == "legal"

def test_retriever_empty_query_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("", top_k=5)

def test_retriever_stats(retriever):
    stats = retriever.get_stats()
    assert stats["total_documents"] > 0
    assert stats["index_dimension"] == 384
    assert "model_name" in stats