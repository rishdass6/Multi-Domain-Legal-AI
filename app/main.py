import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import QuestionRequest, AnswerResponse, HealthResponse, IndexStats
from app.rag_pipeline import RAGPipeline

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Server. Loading RAG pipeline")
    pipeline = RAGPipeline()
    pipeline.load_or_build_index()
    logger.info("RAG Pipeline loaded. Server is live.")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title = "Legal QA API",
    version = "1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {latency}ms")
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if (pipeline and pipeline.is_ready()) else "unavailable",
        model_loaded=pipeline is not None and pipeline.is_ready(),
        index_size=pipeline.get_index_size() if pipeline else 0,
    )
 
 
@app.get("/stats", response_model=IndexStats)
async def stats():
    if not pipeline or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return pipeline.get_stats()
 
 
@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if not pipeline or not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")
 
    start = time.time()
    try:
        result = pipeline.answer(
            question=request.question,
            domain=request.domain or "auto",
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
    latency_ms = round((time.time() - start) * 1000, 2)
 
    logger.info(
        f"ASK | domain={result['domain_detected']} | "
        f"conf={result['confidence']:.4f} | {latency_ms}ms | "
        f"q={request.question[:60]}"
    )
 
    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        confidence=result["confidence"],
        domain_detected=result["domain_detected"],
        sources=result["sources"],
        latency_ms=latency_ms,
    )
 
 
@app.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    background_tasks.add_task(pipeline.rebuild_index)
    return {"message": "Index rebuild started in background."}
 
 
# ── Run directly ──────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
