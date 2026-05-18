import os
import logging
import subprocess
from typing import Optional
from app.retriever import Retriever
from app.qa_model import QAModel
from app.models import AnswerResponse, SourceDocument, IndexStats

logger = logging.getLogger(__name__)

INDEX_PATH = "data/index/index.faiss"
METADATA_PATH = "data/index/metadata.pkl"
CHUNKS_PATH = "data/processed/chunks.jsonl"

LEGAL_KEYWORDS = [
    "contract", "agreement", "clause", "indemnif", "liability", "terminat",
    "breach", "damages", "govern", "jurisdiction", "confidential", "licens",
    "intellectual property", "non-compete", "arbitrat", "force majeure",
    "warrant", "obligat", "party", "parties", "payment", "dispute",
    "negligence", "tort", "statute", "counsel", "attorney", "legal",
]

MEDICAL_KEYWORDS = [
    "patient", "diagnosis", "treatment", "clinical", "drug", "medication",
    "hospital", "physician", "symptom", "disease", "medical", "health",
    "dosage", "prescription", "surgery", "therapy",
]

def detect_domain(question: str) -> str:
    q = question.lower()
    legal_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in q)
    medical_hits = sum(1 for kw in MEDICAL_KEYWORDS if kw in q)

    if medical_hits > legal_hits:
        return "medical"
    return "legal"

def format_sources(chunks: list) -> list:
    sources = []
    seen = set()
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)

        sources.append(SourceDocument(
            text=chunk.get("text", "")[:300],
            source=chunk.get("source", "Unknown"),
            domain =chunk.get("source", "Unknown"),
            score = round(chunk.get("score", 0.0), 4),
            chunk_id = cid
        ))
        return sources
    
class RAGPipeline:
    def __init__(self):
        self.retriever: Optional[Retriever] = None
        self.qa_model: Optional[QAModel] = None
        self._ready = False

    def load_or_build_index(self):
        index_exists = (
        os.path.exists(INDEX_PATH) and
        os.path.exists(METADATA_PATH)
        )

        if not index_exists:
            logger.info("Index not found — building from scratch...")
            self._build_index()

        logger.info("Loading retriever...")
        self.retriever = Retriever(
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
        )
        logger.info(f"Retriever loaded: {self.retriever is not None}")

        logger.info("Loading QA model...")
        self.qa_model = QAModel()
        logger.info(f"QA model loaded: {self.qa_model is not None}")

        self._ready = True
        logger.info(f"Pipeline ready flag set: {self._ready}")
        logger.info(f"is_ready() returns: {self.is_ready()}")

    def _build_index(self):
        chunks_exists = os.path.exists(CHUNKS_PATH)
        if not chunks_exists:
            logger.info("Chunks being built...")
            subprocess.run(["python", "-m", "scripts.build_dataset"], check=True)

        logger.info("Building Faiss index...")
        subprocess.run(["python", "-m", "scripts.build_index"])

    def rebuild_index(self):
        logger.info("Rebuilding index...")
        subprocess.run(["python", "-m", "scripts.build_dataset"], check=True)
        subprocess.run(["python", "-m", "scripts.build_index"], check=True)

        logger.info("Rebuilding Retriever")
        self.retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)

        logger.info("Index rebuilt and retriever reloaded.")

    def is_ready(self) -> bool:
        return self._ready and self.retriever is not None and self.qa_model is not None
    
    def get_index_size(self) -> int:
        if self.retriever:
            return self.retriever.index.ntotal
        return 0
    

    def get_stats(self) -> IndexStats:
        if not self.retriever:
            return IndexStats(
                total_documents=0,
                legal_documents=0,
                medical_documents=0,
                index_dimension=0,
                model_name="not loaded",
            )
        stats = self.retriever.get_stats()
        return IndexStats(**stats)
    
    def answer(self, question: str, domain: str = "auto", top_k: int = 5) -> dict:
        if not self.is_ready():
            raise ValueError("Pipeline not ready. Run load_or_build_index()")
        
        if domain == "auto":
            domain_detected = detect_domain(question)
        else:
            domain_detected = domain

        chunks = self.retriever.search(
            question,
            top_k,
            domain_detected
        )

        qa_result = self.qa_model.answer(
            question,
            chunks
        )

        sources = format_sources(qa_result.get("sources", []))

        return {
            "question": question,
            "answer": qa_result["answer"],
            "confidence": qa_result["confidence"],
            "domain_detected": domain_detected,
            "sources": sources
        }
    
