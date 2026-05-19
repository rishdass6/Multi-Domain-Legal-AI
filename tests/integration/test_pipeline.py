import pytest
from app.rag_pipeline import RAGPipeline

QUESTIONS = [
    "What losses must the indemnifying party cover?",
    "Who is responsible for indemnifying against third party claims?",
    "What is the maximum liability cap under the agreement?",
    "Are consequential damages excluded under this contract?",
    "Under what conditions can a party terminate this agreement?",
    "How many days notice is required to terminate the contract?",
    "What constitutes a force majeure event under this agreement?",
    "What information is deemed confidential under this agreement?",
    "How long does the confidentiality obligation survive termination?",
    "Who shall own all intellectual property developed under this agreement?",
    "What license rights are granted to the licensee?",
    "The laws of which state shall govern this agreement?",
    "What business activities is the party prohibited from engaging in?",
    "When are payments due under this agreement?",
    "How shall disputes between the parties be resolved?",
    "What warranties does the licensor provide?",
    "Can the agreement be assigned to a third party?",
    "What happens upon a change of control of either party?",
    "What is the initial term of this agreement?",
    "What are the audit rights under this agreement?",
]

@pytest.fixture(scope="module")
def pipeline():
    p = RAGPipeline()
    p.load_or_build_index()
    return p

def test_pipeline_is_ready(pipeline):
    assert pipeline.is_ready()

def test_pipeline_index_size(pipeline):
    assert pipeline.get_index_size() > 1000

def test_pipeline_stats_structure(pipeline):
    stats = pipeline.get_stats()
    assert stats.total_documents > 0
    assert stats.index_dimension == 384
    assert stats.model_name != ""

def test_all_questions_return_response(pipeline):
    for q in QUESTIONS:
        result = pipeline.answer(q, domain="auto", top_k=5)
        assert result is not None, f"None returned for: {q}"

def test_all_responses_have_required_keys(pipeline):
    required = {"question", "answer", "confidence", "domain_detected", "sources"}
    for q in QUESTIONS:
        result = pipeline.answer(q)
        assert required.issubset(result.keys()), f"Missing keys for: {q}"

def test_all_answers_are_non_empty_strings(pipeline):
    for q in QUESTIONS:
        result = pipeline.answer(q)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

def test_all_confidence_scores_in_range(pipeline):
    for q in QUESTIONS:
        result = pipeline.answer(q)
        assert 0.0 <= result["confidence"] <= 1.0, \
            f"Confidence out of range for: {q}"

def test_domain_detected_is_valid(pipeline):
    for q in QUESTIONS:
        result = pipeline.answer(q)
        assert result["domain_detected"] in ("legal", "medical"), \
            f"Invalid domain for: {q}"

def test_sources_is_list(pipeline):
    for q in QUESTIONS:
        result = pipeline.answer(q)
        assert isinstance(result["sources"], list)

def test_domain_override_respected(pipeline):
    result = pipeline.answer("What is the governing law?", domain="legal")
    assert result["domain_detected"] == "legal"

def test_top_k_affects_sources(pipeline):
    result3 = pipeline.answer("indemnification clause", top_k=3)
    result5 = pipeline.answer("indemnification clause", top_k=5)
    assert len(result3["sources"]) <= len(result5["sources"])