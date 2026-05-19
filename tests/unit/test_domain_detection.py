import pytest
from app.rag_pipeline import detect_domain

def test_legal_keywords_detected():
    assert detect_domain("What does the indemnification cause require?") == "legal"
    assert detect_domain("When can a party terminate this contract?") == "legal"
    assert detect_domain("What is the governing law of this agreement") == "legal"
    assert detect_domain("Who owns the intellectual property?") == "legal"

def test_medical_keywords_detected():
    assert detect_domain("What is the patient diagnosis?") == "medical"
    assert detect_domain("What drug dosage is prescribed?") == "medical"

def test_defaults_to_legal():
    # Ambiguous or unrecognized questions should default to legal
    assert detect_domain("What happened?") == "legal"
    assert detect_domain("Tell me more") == "legal"

def test_legal_wins_on_tie():
    # legal should win when neither or both match
    assert detect_domain("the contract patient") == "legal"
