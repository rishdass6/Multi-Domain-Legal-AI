import logging
import re
from typing import Dict, Any, List, Optional
from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "deepset/roberta-base-squad2"

MAX_CONTENT_CHARS = 4000

CONFIDENCE_LOW = 0.05
CONFIDENCE_MEDIUM = 0.25

CHUNK_SEPARATOR = "\n\n---\n\n"

FALLBACK_MESSAGE = (
    "I was unable to find a confident answer to your question in the available "
    "legal documents. Consider rephrasing your question or consulting a licensed "
    "attorney for authoritative legal advice."
)

def assemble_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTENT_CHARS) -> str:
    parts = []
    total_chars = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "").strip()
        block = chunk_text  # No source label prefix — saves tokens for actual content
        block_chars = len(block)

        if total_chars + block_chars + len(CHUNK_SEPARATOR) > max_chars:
            remaining = max_chars - total_chars - len(CHUNK_SEPARATOR) - 10
            if remaining > 200:
                parts.append(chunk_text[:remaining])
            break

        parts.append(block)
        total_chars += block_chars + len(CHUNK_SEPARATOR)

    return CHUNK_SEPARATOR.join(parts)

def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^\[Source:.*?\]\s*", "", answer)
    answer = re.sub(r"^[,;:\-–—]+\s*", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()

def interpret_confidence(score: float) -> Dict[str, Any]:
    if score >= CONFIDENCE_MEDIUM:
        return {
            "level": "high",
            "flag": False,
            "message": "Answer found with high confidence."
        }
    elif score >= CONFIDENCE_LOW:
        return {
            "level": "medium",
            "flag": False,
            "message": "Answer found with moderate confidence. Verify with primary legal sources."
        }
    else:
        return {
            "level": "low",
            "flag": True,
            "message": "Confidence too low to return a reliable answer."
        }
    
class QAModel:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._load()

    def _load(self):
        logger.info(f"Loading QA model: {self.model_name}")
        self._pipeline = pipeline(
            "question-answering",
            model = self.model_name,
            tokenizer= self.model_name,
            device = self.device
        )
        logger.info("QA model loaded.")

    def is_ready(self) -> bool:
        return self._pipeline is not None
    
    def answer(self, question: str, chunks: List[Dict[str, Any]], top_k_answers: int = 1) -> Dict[str, Any]:
        if not self._pipeline:
            raise RuntimeError("QA model not loaded.")
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not chunks:
            return self._build_fallback(
                question=question, context="", chunks=[],
                reason="No relevant documents retrieved."
            )

        context = assemble_context(chunks)
        if not context.strip():
            return self._build_fallback(
                question=question, context="", chunks=chunks,
                reason="Retrieved chunks contained no usable text."
            )

        try:
            raw = self._pipeline(
                question=question,
                context=context,
                top_k=1,
                handle_impossible_answer=True,
            )
            if isinstance(raw, list):
                raw = raw[0]
        except Exception as e:
            logger.warning(f"Pipeline error: {e}")
            return self._build_fallback(
                question=question, context=context, chunks=chunks,
                reason=f"Inference failed: {e}",
            )

        score = float(raw.get("score", 0.0))
        answer_text = (raw.get("answer") or "").strip()
        confidence_meta = interpret_confidence(score)

        if not answer_text:
            return self._build_fallback(
                question=question, context=context, chunks=chunks,
                reason="Model declined to answer (no supporting span in context).",
                raw_score=score,
            )

        if confidence_meta["level"] == "low":
            return self._build_fallback(
                question=question, context=context, chunks=chunks,
                reason=confidence_meta["message"],
                raw_score=score,
            )

        cleaned = clean_answer(answer_text)
        if not cleaned:
            return self._build_fallback(
                question=question, context=context, chunks=chunks,
                reason="Cleaned answer span is empty.",
                raw_score=score,
            )

        source_chunk = next(
            (c for c in chunks if cleaned and cleaned in c.get("text", "")),
            chunks[0],
        )

        return {
            "question": question,
            "answer": cleaned,
            "confidence": round(score, 4),
            "confidence_meta": confidence_meta,
            "context_used": source_chunk.get("text", ""),
            "is_fallback": confidence_meta["flag"],
            "sources": chunks,
        }
    
    def _build_fallback(
        self,
        question: str,
        context: str,
        chunks: List[Dict[str, Any]],
        reason: str,
        raw_score: float = 0.0,
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Fallback triggered: {reason}")
        if level is None:
            level = interpret_confidence(raw_score)["level"]
        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "confidence": round(raw_score, 4),
            "confidence_meta": {
                "level": level,
                "flag": True,
                "message": reason,
            },
            "context_used": context,
            "is_fallback": True,
            "sources": chunks,
        }