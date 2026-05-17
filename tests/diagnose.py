"""
RAG Pipeline Diagnostic Script
Run with: python -m scripts.diagnose
"""
 
from app.retriever import Retriever
from app.qa_model import assemble_context
from transformers import pipeline
 
TEST_CASES = [
    "What does the indemnification clause require the party to cover?",
    "How many days notice is required to terminate the contract?",
    "What events are covered under the force majeure clause?",
]
 
def main():
    print("Loading retriever...")
    retriever = Retriever()
 
    print("Loading QA model...")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
 
    for q in TEST_CASES:
        print("=" * 70)
        print(f"QUESTION: {q}")
 
        chunks = retriever.search(q, top_k=5, domain_filter="legal")
        print(f"CHUNKS RETRIEVED: {len(chunks)}")
        for i, c in enumerate(chunks):
            score = c["score"]
            preview = c["text"][:120]
            print(f"  [{i+1}] score={score:.4f} | {preview}")
 
        context = assemble_context(chunks)
        print(f"CONTEXT LENGTH: {len(context)} chars")
        print(f"CONTEXT PREVIEW:")
        print(context[:800])
 
        result = qa(question=q, context=context)
        raw_answer = result["answer"]
        raw_score = result["score"]
        print(f"RAW ANSWER: {raw_answer}")
        print(f"RAW SCORE:  {raw_score:.4f}")
        print()
 
if __name__ == "__main__":
    main()
 
