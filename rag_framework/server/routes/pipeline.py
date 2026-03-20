"""
Pipeline API routes — ingest and query.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from rag_framework.server import state

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def _get_or_build_pipeline():
    pipeline = state.get_pipeline()
    if pipeline is None:
        from rag_framework.pipeline.orchestrator import RAGPipeline
        config = state.get_config()
        pipeline = RAGPipeline(config)
        pipeline.validate()
        state.set_pipeline(pipeline)
    return pipeline


@router.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    summaries = []
    errors = []

    try:
        pipeline = _get_or_build_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pipeline setup failed: {e}")

    for upload in files:
        suffix = os.path.splitext(upload.filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            summary = pipeline.ingest(tmp_path)
            summary["filename"] = upload.filename
            summaries.append(summary)
            state.add_ingestion_summary(summary)
        except Exception as e:
            errors.append({"filename": upload.filename, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    return {
        "ok": len(errors) == 0,
        "summaries": summaries,
        "errors": errors,
    }


@router.post("/query")
def query(body: QueryRequest):
    try:
        pipeline = _get_or_build_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pipeline setup failed: {e}")

    try:
        results = pipeline.query(body.query, top_k=body.top_k)
        state.set_query_results(results)
        return {
            "ok": True,
            "results": [
                {
                    "text": r.chunk.text,
                    "score": round(r.score, 4),
                    "metadata": r.chunk.metadata,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
def reset():
    try:
        pipeline = _get_or_build_pipeline()
        pipeline.reset_store()
        state.set_pipeline(None)
        state.clear_ingestion_summaries()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status():
    summaries = state.get_ingestion_summaries()
    pipeline = state.get_pipeline()
    config = state.get_config()
    return {
        "pipeline_ready": pipeline is not None,
        "documents_ingested": len(summaries),
        "total_chunks": sum(s.get("chunk_count", 0) for s in summaries),
        "config_name": config.name,
        "parser": config.parser.implementation.value,
        "chunker": config.chunker.implementation.value,
        "embedder": config.embedder.implementation.value,
        "vector_store": config.vector_store.implementation.value,
        "retriever": config.retriever.implementation.value,
    }
