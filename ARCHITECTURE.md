# Architecture Deep-Dive

A technical reference for the retrieval strategy, chunking decisions, model choices, and design trade-offs in the Multi-Domain Legal QA system.

---

## System Overview

The system is a two-stage RAG pipeline: an offline indexing pipeline that processes raw documents into a searchable vector store, and an online serving pipeline that answers questions by retrieving relevant passages and running extractive QA over them.

```
Offline (scripts/)          Online (app/)
─────────────────           ──────────────
download_cuad.py            retriever.py
scrape_lii.py          ──►  qa_model.py
build_dataset.py            rag_pipeline.py
build_index.py              main.py
```

---

## Data Sources

### CUAD (Contract Understanding Atticus Dataset)
Downloaded from HuggingFace via the `datasets` library. CUAD contains 510 commercial legal agreements (NDAs, employment contracts, IP licenses, SaaS agreements, etc.) manually annotated by lawyers for 41 clause types. We use the raw contract text — not the annotations — treating it as an unlabeled corpus for open-domain retrieval.

**Why CUAD:** Freely available, professionally authored contracts, broad coverage of clause types that appear in the evaluation benchmark, and pre-cleaned enough that minimal text munging is required.

### Cornell LII Wex Glossary
Scraped from the Cornell Law School Legal Information Institute using BeautifulSoup4. The Wex glossary contains ~400 legal term definitions written for a general audience — useful for definitional questions ("What is indemnification?", "What does force majeure mean?").

**Why LII:** Complements CUAD's contract-centric corpus with authoritative definitions. A question about what a clause *means* is different from a question about what a specific contract *says*, and having both data sources lets the retriever answer both question types from the same index.

---

## Preprocessing

### Text Cleaning (`app/preprocessing.py`)

Each source gets its own cleaner:

**CUAD cleaner:**
- Strips PDF-extraction artifacts: page numbers (`Page X of Y`), section headers that are purely numeric, signature block boilerplate (`IN WITNESS WHEREOF`, `EXHIBIT`), repeated underscores used as signature lines
- Normalizes Unicode to ASCII where safe (curly quotes → straight quotes, em-dashes → hyphens)
- Collapses runs of whitespace and blank lines
- Minimum token threshold: discards paragraphs shorter than 20 tokens (typically headers, exhibit labels)

**LII cleaner:**
- Strips HTML navigation artifacts left by BeautifulSoup
- Removes "See also" / "Further reading" sections at end of entries
- Lighter touch than CUAD — definitions are already well-structured prose

The cleaners are intentionally conservative. Over-cleaning legal text removes signal: section numbering, defined terms in ALL CAPS, and cross-references ("as defined in Section 4.2") are all meaningful in context.

---

## Chunking Strategy

### Why token-based, not character-based

Legal contracts contain long sentences with embedded subclauses. Character-based splitters cut at fixed byte offsets, which means they frequently split mid-clause or mid-defined-term. Token-based splitting respects the natural token boundaries that the downstream transformer models were trained on, and the chunk size (512 tokens) aligns with the typical context window of the embedding model.

### Parameters

```python
CHUNK_SIZE = 512        # tokens (tiktoken cl100k_base encoder)
CHUNK_OVERLAP = 100     # tokens
MIN_CHUNK_TOKENS = 20   # discard chunks smaller than this
```

**Why 512 tokens:** Matches the sequence length that `all-MiniLM-L6-v2` was trained on. Going larger would require truncation at embedding time, losing the tail of each chunk. Going smaller increases the number of chunks without proportionally increasing coverage, and short chunks often lack enough context for the QA model to extract a confident answer.

**Why 100-token overlap:** A 20% overlap ensures that clause boundaries — which rarely align with our chunk boundaries — appear in at least one complete chunk. Without overlap, a sentence that straddles two chunks would be split and might not retrieve correctly for either half. The cost is a ~10% increase in total chunks (from ~11,400 to ~12,600).

### Splitter hierarchy

```python
separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
```

The `RecursiveCharacterTextSplitter` (LangChain) tries each separator in order, splitting only when the current chunk exceeds the size limit. This means the splitter prefers paragraph breaks, then sentence endings, then clause breaks — preserving semantic coherence as long as possible. The final `""` fallback ensures no chunk ever exceeds the limit even for run-on prose.

### Output

After chunking, each chunk carries metadata:
- `chunk_id` — UUID for deduplication
- `source` — `"cuad"` or `"lii"`
- `domain` — `"legal"` (medical domain is stubbed for future expansion)
- `title` — contract filename or glossary term
- `url` — original document URL (for LII entries)
- `token_count` — actual token count of this chunk

Final corpus: **12,603 chunks**, ~5.6M tokens, avg 444 tokens/chunk.

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
**Dimensionality:** 384
**Normalization:** L2-normalized before indexing (converts inner product to cosine similarity)
**Batch size at index time:** 64

### Why all-MiniLM-L6-v2

The choice was driven by four constraints:

1. **CPU inference.** The system targets deployment without a GPU. MiniLM-L6 (6 transformer layers, 384-dim hidden states) runs a 512-token chunk through the encoder in ~15 ms on a modern CPU. The larger `all-mpnet-base-v2` (768-dim, 12 layers) takes ~80 ms per chunk and would make index building prohibitively slow.

2. **MTEB benchmark quality.** On the BEIR retrieval benchmark, `all-MiniLM-L6-v2` scores within ~3 NDCG@10 points of `all-mpnet-base-v2` while being 5× faster. That trade-off is acceptable given that the cross-encoder reranker compensates for lower bi-encoder precision.

3. **Memory footprint.** 384-dim vectors for 12,603 chunks require 19 MB in the FAISS index. 768-dim would double that and increase RAM pressure at query time.

4. **Pretrained on legal-adjacent data.** The model was fine-tuned on a combination of SNLI, MNLI, and MS MARCO, which includes formal language. It generalizes reasonably to legal prose without domain-specific fine-tuning.

**Known limitation:** The model has no legal domain fine-tuning. Terms like "material breach" or "indemnification" are encoded using general-purpose token embeddings rather than representations calibrated on legal semantics. A domain-adapted model (e.g., `legal-xlm-roberta-base` or a MiniLM fine-tuned on CUAD QA pairs) would improve retrieval precision on legal jargon.

---

## Vector Index

**Library:** FAISS (Facebook AI Similarity Search), CPU build  
**Index type:** `IndexFlatIP` (exact inner product search, no approximation)  
**Stored on disk:** `data/index/index.faiss` (19 MB), `data/index/metadata.pkl` (26 MB)

### Why IndexFlatIP over approximate indexes (HNSW, IVF)

At 12,603 vectors of 384 dimensions, an exact brute-force search completes in ~2 ms on CPU. Approximate indexes (HNSW, IVF) are faster asymptotically but have nontrivial build-time, tuning overhead (nlist, ef_construction), and recall < 100%. At this corpus size there is no justification for approximation: exact search is fast enough and eliminates recall loss.

If the corpus grows to ~1M+ chunks, the right migration path is `IndexIVFFlat` (inverted file with flat quantizer) with `nlist ≈ sqrt(N)`, which gives ~10× query speedup with ~99% recall at reasonable `nprobe` settings.

### Search procedure

```
query → embed (MiniLM) → L2-normalize → FAISS.search(top_k × 10) → cross-encoder rerank → dedup → top_k results
```

The `× 10` over-fetch multiplier is the key design choice. FAISS returns by cosine similarity, which is good at approximate semantic matching but cannot account for the interaction between a full query and a passage. The cross-encoder can, but requires explicit candidate pairs — it cannot search a corpus directly. The multiplier creates a candidate pool large enough that the cross-encoder's re-ordering is meaningful.

---

## Cross-Encoder Reranker

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Input:** `[CLS] question [SEP] passage [SEP]`  
**Output:** Scalar relevance score  
**Loading:** Lazy (instantiated on first use, cached thereafter)

### Why a cross-encoder

The bi-encoder (MiniLM) encodes query and document independently and compares their embeddings. This is fast but loses the fine-grained word-level interaction between query terms and document terms. A cross-encoder encodes both together with full self-attention across the concatenated input, capturing whether specific words in the query match specific words in the passage.

For legal QA, this interaction is critical. The question "What interest rate applies to late payments?" and a passage about "interest in the licensee's intellectual property" have high embedding similarity but low relevance — the cross-encoder distinguishes them; the bi-encoder does not.

The MS MARCO training data is question-passage pairs from web search, which is imperfect for legal text. A cross-encoder fine-tuned on legal question–clause pairs would perform better, but the off-the-shelf model provides a meaningful improvement over bi-encoder-only retrieval.

**Latency cost:** ~80 ms for 50 candidates (10× top_k=5). This is acceptable for an interactive API but would be a bottleneck at high throughput — at that scale, batching or a dedicated reranking service would be necessary.

---

## Source Deduplication

After reranking, results are filtered to a maximum of **2 chunks per source document** (keyed on `title`). Without this cap, a single verbose contract can dominate all top-k slots — especially for generic questions like "What is the governing law?" where the governing law clause appears in similar form in dozens of contracts.

The deduplication forces the context window to draw from diverse documents, which:
1. Reduces the risk that the QA model sees the same boilerplate phrasing repeated and over-indexes on it
2. Provides more representative source attribution in the API response

**Trade-off:** If the relevant information spans more than 2 chunks of a single document (e.g., a multi-part indemnification clause), the cap discards potentially useful context. This is a known limitation for complex multi-part questions.

---

## Extractive QA Model

**Model:** `deepset/roberta-base-squad2`  
**Architecture:** RoBERTa-base fine-tuned on SQuAD 2.0  
**Input:** `question [SEP] context` (up to 4,000 characters of concatenated chunks)  
**Output:** Start/end token span + confidence score

### Why extractive over generative

For a legal QA system, extractive models have two important properties:

1. **Grounded answers.** Every answer is literally a substring of the source document. The model cannot paraphrase, hallucinate, or mix information from multiple sources into a synthetic answer.

2. **Abstain on SQuAD 2.0.** The model was fine-tuned on SQuAD 2.0, which includes unanswerable questions. When the passage does not contain the answer, the model assigns high probability to the null span (`[CLS]` token), producing a low confidence score that triggers the fallback rather than a confabulated answer.

**Known limitation:** Extractive models fail on questions whose answers require synthesis across multiple passages (e.g., "What are all the conditions under which this agreement can be terminated?"). A generative model with retrieved context could combine information across chunks; the extractive model can only pick one span from one position in the assembled context.

### Context assembly

```python
MAX_CONTEXT_CHARS = 4000
separator = "\n\n---\n\n"
```

Chunks are concatenated in retrieval order (post-rerank) up to the character limit. The `---` separator gives the model a visual signal that the context switches documents, which may help it avoid conflating passages from different contracts.

4,000 characters ≈ 700-800 tokens, which fits comfortably within RoBERTa-base's 512-token maximum with the question prepended. The limit discards the tail of the retrieved context; in practice, the reranker ensures the most relevant passage is first, so the discarded tail is rarely the answer location.

### Confidence gating

```python
HIGH_CONFIDENCE   = 0.25   # answer returned directly
MEDIUM_CONFIDENCE = 0.05   # answer returned with hedging
LOW_CONFIDENCE    = 0.05   # fallback message
```

These thresholds were set conservatively for the legal domain. A wrong legal answer — especially one that sounds authoritative — can cause real harm. The fallback message explicitly directs users to consult a licensed attorney, which is the correct behavior when the system is uncertain.

The 43% fallback rate on the evaluation benchmark reflects both the threshold conservatism and a genuine difficulty: many questions in the golden dataset ask about contract-specific numeric values (payment terms, notice periods, liability caps) that appear in varied formats across contracts, and the current single-contract retrieval doesn't guarantee retrieving the exact contract whose values match the expected answer.

---

## Domain Detection

```python
LEGAL_KEYWORDS = ["contract", "agreement", "clause", "indemnif", ...]  # 21 terms
MEDICAL_KEYWORDS = ["patient", "diagnosis", "treatment", ...]          # 16 terms
```

Domain is detected by counting keyword hits in the question. The medical domain is a stub — the current corpus has no medical documents, but the architecture is designed to accept additional source directories. Adding a medical domain requires: (1) dropping documents into `data/raw/medical/`, (2) adding a medical preprocessor, (3) re-running the build pipeline.

The keyword approach is deliberately simple. At one domain it cannot make a wrong call; at two or more it's a tiebreaker that directs the FAISS search to a filtered subindex. A proper classifier (fine-tuned BERT) would be worth building if the system expands to 3+ domains with significant lexical overlap.

---

## API Design

```
POST /ask          → main QA endpoint
GET  /health       → liveness probe
GET  /stats        → index metadata
POST /index/rebuild → async re-indexing
```

The `/health` endpoint is used as the Docker health check (every 30s). It validates that both the FAISS index and QA model are loaded — a response of `"status": "ok"` means the system can serve requests, not just that the container is running.

The `/index/rebuild` endpoint runs `build_dataset.py` and `build_index.py` as subprocesses, allowing the index to be refreshed without restarting the container. This is useful for adding new documents without a full redeploy.

---

## Performance Characteristics

| Operation | Latency (CPU) |
|-----------|---------------|
| Embed query (MiniLM) | ~5 ms |
| FAISS search (12K vectors, 384-dim) | ~2 ms |
| Cross-encoder rerank (50 candidates) | ~80 ms |
| RoBERTa extractive QA | ~60 ms |
| **Total end-to-end** | **~150-350 ms** |

Index build times (one-time, offline):
- Embed 12,603 chunks (batch_size=64): ~8 minutes on CPU
- FAISS index construction: ~1 second

---

## Known Limitations & Future Work

| Limitation | Root Cause | Mitigation |
|------------|------------|------------|
| 43% fallback rate | Conservative confidence thresholds + sparse numeric answers in corpus | Fine-tune extractive model on CUAD QA pairs; add numeric extraction post-processor |
| Wrong answers on high-confidence retrieval | Bi-encoder finds right document, extractive model picks wrong span | Replace extractive head with a prompted LLM reader |
| No multi-hop reasoning | Extractive model reads a single assembled context | Add query decomposition; retrieve per sub-question |
| Medical domain stub | No medical corpus | Ingest clinical guidelines / medical literature |
| Keyword domain detection | Fragile for ambiguous queries | Train a proper domain classifier |
| Index rebuild is blocking | Subprocess call on main process | Move to background task queue (Celery, ARQ) |
