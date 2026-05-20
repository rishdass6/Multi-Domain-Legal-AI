# Multi-Domain Legal QA System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for question-answering over legal documents. The system ingests contract clauses from the CUAD dataset and legal definitions from the Cornell LII Wex glossary, builds a FAISS vector index over ~12,600 semantic chunks, and serves answers through a FastAPI endpoint backed by a cross-encoder reranker and an extractive QA head — returning the answer span, a calibrated confidence score, and the source passages it drew from.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   Data Ingestion                    │
                        │  ┌──────────────────┐   ┌──────────────────────┐   │
                        │  │  CUAD Contracts   │   │  Cornell LII Wex     │   │
                        │  │  (HuggingFace)    │   │  (BeautifulSoup4)    │   │
                        │  └────────┬─────────┘   └──────────┬───────────┘   │
                        └──────────┼──────────────────────────┼───────────────┘
                                   │                          │
                        ┌──────────▼──────────────────────────▼───────────────┐
                        │           Preprocessing & Chunking                   │
                        │  Domain-specific cleaning → 512-token chunks         │
                        │  (100-token overlap, recursive character splitter)   │
                        └────────────────────────┬────────────────────────────┘
                                                 │  12,603 chunks
                        ┌────────────────────────▼────────────────────────────┐
                        │               Index Building (Phase 3)               │
                        │  all-MiniLM-L6-v2 embeddings → FAISS IndexFlatIP    │
                        └────────────────────────┬────────────────────────────┘
                                                 │
              ┌──────────────────────────────────┼──────────────────────────────────┐
              │                       Query-Time Pipeline                           │
              │                                  │                                  │
              │  User Question                   │                                  │
              │      │                           │                                  │
              │      ▼                           │                                  │
              │  Domain Detection            FAISS Index                            │
              │  (keyword heuristic)             │                                  │
              │      │                           │                                  │
              │      └──────► Retriever ◄────────┘                                 │
              │               (top-k × 10 candidates)                              │
              │                    │                                                │
              │                    ▼                                                │
              │            Cross-Encoder Reranker                                  │
              │          (ms-marco-MiniLM-L-6-v2)                                  │
              │                    │                                                │
              │                    ▼                                                │
              │          Source Deduplication                                       │
              │          (max 2 chunks/document)                                    │
              │                    │                                                │
              │                    ▼                                                │
              │         Extractive QA Model                                         │
              │       (roberta-base-squad2)                                         │
              │          + Confidence Gating                                        │
              │                    │                                                │
              │                    ▼                                                │
              │         FastAPI /ask endpoint                                       │
              └────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- Docker and Docker Compose installed
- ~5 GB free disk space (models + index)

### Run with Docker

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Domain_Legal.git
cd Multi-Domain_Legal

docker-compose up --build
```

The first start downloads model weights (~500 MB) and builds the FAISS index from the pre-processed chunks. This takes 3-5 minutes. Subsequent starts load the cached index in under 10 seconds.

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "index_size": 12603,
  "retriever_loaded": true,
  "qa_model_loaded": true
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "The laws of which state shall govern this agreement?",
    "domain": "auto",
    "top_k": 5
  }'
```

```json
{
  "question": "The laws of which state shall govern this agreement?",
  "answer": "New York",
  "confidence": 0.613,
  "domain_detected": "legal",
  "sources": [
    {
      "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York...",
      "source": "cuad",
      "domain": "legal",
      "score": 0.8241,
      "chunk_id": "a3f91c..."
    }
  ],
  "latency_ms": 312.4
}
```

### Index statistics

```bash
curl http://localhost:8000/stats
```

```json
{
  "total_documents": 12603,
  "legal_documents": 12603,
  "medical_documents": 0,
  "index_dimension": 384,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

## Sample Questions & Answers

### Contract Domain (CUAD)

| Question | Answer | Confidence |
|----------|--------|------------|
| The laws of which state shall govern this agreement? | New York | 0.613 |
| How shall disputes between the parties be resolved? | by binding arbitration | 0.138 |
| Who shall indemnify and hold harmless the other party? | Each party | 0.204 |
| What license rights are granted to the licensee? | non-exclusive license to use the logos, trademarks and service marks | 0.089 |
| Can either party assign this agreement without consent? | This Agreement may not be assigned by either party without the prior written consent | 0.105 |
| How long does the non-compete restriction last? | three (3) years | 0.310 |
| What is the initial term of this agreement? | one (1) year | 0.054 |
| What notice must be given to trigger indemnification? | written | 0.113 |

### Legal Definitions Domain (Cornell LII Wex)

| Question | Answer |
|----------|--------|
| What is indemnification? | A contractual obligation where one party agrees to compensate another for specified losses, damages, or liabilities arising from a transaction or relationship. |
| What does force majeure mean? | A clause that excuses a party from liability if an extraordinary event beyond its control prevents performance of contractual obligations. |
| What is the parol evidence rule? | A rule preventing parties to a written contract from presenting extrinsic evidence that contradicts, varies, or adds to its terms. |
| What constitutes a breach of contract? | A failure without legal excuse to perform any promise forming part of a contract, including partial or incomplete performance. |

> **Note:** Low-confidence answers trigger a safe fallback: *"I was unable to find a confident answer in the available legal documents. Consider consulting a licensed attorney."* Thresholds: ≥0.25 = direct answer, 0.05-0.25 = hedged answer, <0.05 = fallback.

---

## Phase 7 Evaluation Metrics

Evaluated on a **30-question golden dataset** spanning 13 contract law categories (Confidentiality, Governing Law, Termination, Liability, Indemnification, Intellectual Property, Force Majeure, Payment, Dispute Resolution, Assignment, Warranties, Non-Compete, General).

| Metric | Score |
|--------|-------|
| Exact Match (EM) | 33.3% (10/30) |
| Average Token F1 | 0.157 |
| Fallback Rate | 43.3% (13/30) |
| Wrong Answer Rate | 23.3% (7/30) |

### By Category

| Category | Result |
|----------|--------|
| Governing Law | 2/2 PASS |
| Indemnification | 2/3 PASS |
| IP Rights | 1/3 PASS |
| Force Majeure | 1/2 PASS |
| Dispute Resolution | 1/2 PASS |
| Termination | 1/4 PASS |
| Assignment | 1/2 PASS |
| Non-Compete | 1/2 PASS |
| Confidentiality | 0/3 — fallback on all |
| Liability | 0/2 — wrong answers |
| Payment | 0/2 — fallback on all |
| Warranties | 0/2 — fallback on all |
| General | 0/1 — fallback |

**Key observation:** The model answers confidently and correctly on clause-type questions (governing law, indemnification, force majeure) but struggles with numeric specifics (payment terms, notice periods, liability caps) and highly context-dependent clauses (confidentiality scope, warranty disclaimers).

---

## Project Structure

```
Multi-Domain_Legal/
├── app/                       # Core application
│   ├── main.py               # FastAPI server + endpoints
│   ├── rag_pipeline.py       # Pipeline orchestrator
│   ├── retriever.py          # FAISS retrieval + cross-encoder reranking
│   ├── qa_model.py           # Extractive QA + confidence gating
│   ├── chunker.py            # Token-based document chunker
│   ├── preprocessing.py      # Domain-specific text cleaning
│   └── models.py             # Pydantic request/response schemas
├── scripts/                   # Offline pipeline scripts
│   ├── download_cuad.py      # Phase 1: download CUAD dataset
│   ├── scrape_lii.py         # Phase 1: scrape LII Wex glossary
│   ├── build_dataset.py      # Phase 2: preprocess + chunk
│   └── build_index.py        # Phase 3: embed + build FAISS index
├── tests/
│   ├── unit/                 # Unit tests (retriever, QA, domain detection)
│   ├── integration/          # End-to-end pipeline tests
│   └── eval/
│       ├── evaluate.py       # Golden dataset evaluation runner
│       └── golden_dataset.py # 30-question benchmark
├── data/
│   ├── raw/                  # Phase 1 output (CUAD JSON, LII JSON)
│   ├── processed/            # Phase 2 output (chunks.jsonl, eval_results.csv)
│   └── index/                # Phase 3 output (index.faiss, metadata.pkl)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## API Reference

### `POST /ask`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | 5–1000 characters |
| `domain` | string | `"auto"` | `"legal"`, `"medical"`, or `"auto"` |
| `top_k` | int | `5` | Number of passages to retrieve (1–20) |

### `GET /health`
Returns liveness status, index size, and model load state.

### `GET /stats`
Returns document counts by domain and index dimensionality.

### `POST /index/rebuild`
Triggers async re-ingestion, re-chunking, and re-indexing.

---

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| API Server | FastAPI 0.136 + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector Index | FAISS IndexFlatIP (cosine via L2-norm) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Extractive QA | `deepset/roberta-base-squad2` |
| Tokenization | `tiktoken` cl100k_base |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Data Validation | Pydantic v2 |
| Containerization | Docker + Docker Compose |
