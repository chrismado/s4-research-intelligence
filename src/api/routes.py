"""
FastAPI routes for the S4 Research Intelligence API.

Provides endpoints for querying the research corpus, ingesting documents,
and inspecting the vector store state. Designed for both programmatic
access and integration with a frontend.
"""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from config.settings import settings
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.loader import load_document, load_from_manifest
from src.ingestion.vectorstore import VectorStore
from src.models.documents import SourceType
from src.models.queries import ResearchQuery, ResearchResponse
from src.retrieval.pipeline import ResearchPipeline

router = APIRouter()

# Singleton instances (initialized on first request)
_vector_store: VectorStore | None = None
_pipeline: ResearchPipeline | None = None


def _get_pipeline() -> ResearchPipeline:
    global _vector_store, _pipeline
    if _pipeline is None:
        _vector_store = VectorStore()
        _pipeline = ResearchPipeline(vector_store=_vector_store)
    return _pipeline


def _get_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# --- Research endpoints ---


@router.post("/research", response_model=ResearchResponse)
async def research(query: ResearchQuery) -> ResearchResponse:
    """
    Submit a research question to the S4 corpus.

    Returns a structured response with cited sources, contradiction
    detection, timeline extraction, and confidence scoring.
    Uses async LLM calls for non-blocking performance.
    """
    pipeline = _get_pipeline()

    try:
        response = await pipeline.async_query(query)
        logger.info(
            f"Query answered | confidence={response.confidence:.2f} "
            f"| sources={len(response.sources)} | contradictions={len(response.contradictions)}"
        )
        return response
    except Exception as e:
        logger.error(f"Research query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/research/quick")
async def quick_research(question: str = Form(...)) -> ResearchResponse:
    """Simplified endpoint — just a question string, no filters."""
    query = ResearchQuery(question=question)
    return await research(query)


@router.post("/research/conversation", response_model=ResearchResponse)
async def conversation_research(query: ResearchQuery) -> ResearchResponse:
    """
    Multi-turn research query with conversation memory.

    Follow-up questions automatically include context from prior turns.
    """
    pipeline = _get_pipeline()

    try:
        response = pipeline.query_with_memory(query)
        logger.info(
            f"Conversation query answered | confidence={response.confidence:.2f} "
            f"| turns={pipeline.memory.turn_count}"
        )
        return response
    except Exception as e:
        logger.error(f"Conversation query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Ingestion endpoints ---


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),  # noqa: B008
    source_type: SourceType = Form(SourceType.PRODUCTION_NOTE),  # noqa: B008
    title: str = Form(""),
    author: str = Form(""),
):
    """Upload and ingest a single document."""
    store = _get_store()

    # Save uploaded file with size check
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(content) // (1024*1024)}MB). "
                f"Max: {settings.max_upload_size_mb}MB"
            ),
        )

    upload_dir = settings.raw_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / file.filename
    filepath.write_bytes(content)

    # Load, chunk, embed
    metadata_override = {"source_type": source_type.value}
    if title:
        metadata_override["title"] = title
    if author:
        metadata_override["author"] = author

    doc = load_document(filepath, metadata_override=metadata_override)
    chunks = chunk_document(doc)
    added = store.add_chunks(chunks)

    return {
        "status": "ingested",
        "filename": file.filename,
        "chunks": added,
        "total_in_store": store.count,
    }


@router.post("/ingest/manifest")
async def ingest_manifest(manifest_path: str = Form(...)):
    """Ingest documents from a JSON manifest file."""
    store = _get_store()
    path = Path(manifest_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {path}")

    docs = load_from_manifest(path)
    chunks = chunk_documents(docs)
    added = store.add_chunks(chunks)

    return {
        "status": "ingested",
        "documents": len(docs),
        "chunks": added,
        "total_in_store": store.count,
    }


# --- Store info ---


@router.get("/store/stats")
async def store_stats():
    """Get vector store statistics."""
    store = _get_store()
    return {
        "collection": settings.chroma_collection,
        "total_chunks": store.count,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }


# --- Evaluation endpoints ---

_eval_results: dict[str, dict] = {}
_eval_status: dict[str, str] = {}

_VALID_EVAL_MODULES = {
    "golden_set",
    "hallucination",
    "adversarial",
    "benchmarks",
    "regression",
    "quantization",
    "compare",
    "scaling",
}


@router.post("/eval/run")
async def run_eval(
    modules: list[str] = Form(default=["golden_set"]),  # noqa: B008
):
    """Launch an evaluation run (async, returns run_id)."""
    import threading
    import uuid

    from src.evaluation.suite import EvalSuite

    invalid = [m for m in modules if m not in _VALID_EVAL_MODULES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown eval module(s): {invalid}. " f"Valid: {sorted(_VALID_EVAL_MODULES)}"),
        )

    run_id = str(uuid.uuid4())[:8]
    _eval_status[run_id] = "running"

    def _run():
        suite = EvalSuite()
        results = {}
        for mod in modules:
            runner = getattr(suite, f"run_{mod}", None)
            if runner:
                result = runner()
                results[mod] = result.to_dict() if hasattr(result, "to_dict") else result
        _eval_results[run_id] = results
        _eval_status[run_id] = "complete"

    threading.Thread(target=_run, daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@router.get("/eval/status/{run_id}")
async def eval_status(run_id: str):
    """Check the status of an evaluation run."""
    status = _eval_status.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {"run_id": run_id, "status": status}


@router.get("/eval/report/{run_id}")
async def eval_report(run_id: str):
    """Get the results of a completed evaluation run."""
    if run_id not in _eval_results:
        status = _eval_status.get(run_id)
        if status == "running":
            raise HTTPException(status_code=202, detail="Evaluation still running")
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _eval_results[run_id]


@router.get("/eval/history")
async def eval_history():
    """List all evaluation runs."""
    return [{"run_id": rid, "status": _eval_status[rid]} for rid in _eval_status]
