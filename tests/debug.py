# paste this into a quick debug script
from app.retriever import Retriever

r = Retriever()
test_questions = [
    "What events are covered under the force majeure clause?",
    "Who is responsible for indemnifying against third party claims?",
    "Which state law governs this agreement?",
]

for q in test_questions:
    print(f"\nQ: {q}")
    chunks = r.search(q, top_k=5)
    for i, c in enumerate(chunks):
        print(f"  [{i}] score={c['score']:.4f} | {c['text'][:120]}")