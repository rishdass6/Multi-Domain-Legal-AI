import pytest
from app.qa_model import QAModel, assemble_context, interpret_confidence, clean_answer

KNOWN_CONTEXT = """
10.1 Governing Law. This Agreement shall be governed by and construed
in accordance with the laws of the State of Delaware, without regard
to its conflict of law provisions.
"""

KNOWN_QUESTION = "Which state law governs this agreement?"
KNOWN_ANSWER_KEYWORD = "Delaware"

@pytest.fixture(scope = "module")
def qa_model():
    return QAModel()

def test_qa_model_loads(qa_model):
    assert qa_model.is_ready()

def test_qa_model_finds_known_answer(qa_model):
    test_context = {
        "text": KNOWN_CONTEXT,
        "source": "test", 
        "domain": "legal",
        "score": 0.9,
        "chunk_id": "test-001"
    }

    answer = qa_model.answer(KNOWN_QUESTION, [test_context])
    assert "answer" in answer
    assert "confidence" in answer
    assert "is_fallback" in answer

def test_qa_model_answer_contains_delaware(qa_model):
    text_context = {
        "text": KNOWN_CONTEXT,
        "source": "test",
        "domain": "legal",
        "score": 0.9,
        "chunk": "test-001"
    }
    answer = qa_model.answer(KNOWN_QUESTION, [text_context])
    if not answer["is_fallback"]:
        assert KNOWN_ANSWER_KEYWORD in answer["answer"]

def test_qa_model_empty_chunks_returns_fallback(qa_model):
    answer = qa_model.answer("What is the governing law?", [])
    assert answer["is_fallback"] is True

def test_qa_model_empty_question_raises(qa_model):
    with pytest.raises(ValueError):
        qa_model.answer("", [{"text": "some text", "source": "test",
                               "domain": "legal", "score": 0.5, "chunk_id": "x"}])
        
def test_confidence_interpretation_high():
    result = interpret_confidence(0.9)
    assert result["level"] == "high"
    assert result["flag"] is False

def test_confidence_interpretation_medium():
    result = interpret_confidence(0.15)
    assert result["level"] == "medium"
    assert result["flag"] is False

def test_confidence_interpretation_low():
    result = interpret_confidence(0.01)
    assert result["level"] == "low"
    assert result["flag"] is True

def test_clean_answer_strips_whitespace():
    assert clean_answer("  Delaware   ") == "Delaware"

def test_clean_answer_removes_source_label():
    assert clean_answer("[Source: cuad]\nDelaware") == "Delaware"

def test_assemble_context_respects_limit():
    chunks = [{"text": "x" * 500, "source": "test", "domain": "legal",
               "score": 0.9, "chunk_id": str(i)} for i in range(10)]
    context = assemble_context(chunks, max_chars=600)
    assert len(context) <= 700  # small buffer for separators