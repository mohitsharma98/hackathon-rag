"""
Config API routes — CRUD, validation, YAML import/export.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError

from rag_framework.config.models import PipelineConfig
from rag_framework.core.exceptions import PipelineSetupError
from rag_framework.server import state

router = APIRouter(prefix="/api/config", tags=["config"])


class SaveConfigRequest(BaseModel):
    config: dict


@router.get("/current")
def get_current_config():
    return state.get_config().model_dump()


@router.post("/save")
def save_config(body: SaveConfigRequest):
    try:
        config = PipelineConfig.model_validate(body.config)
        state.set_config(config)
        return {"ok": True}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@router.post("/validate")
def validate_config(body: SaveConfigRequest):
    """Save config then run health checks on every module."""
    try:
        config = PipelineConfig.model_validate(body.config)
        state.set_config(config)
    except ValidationError as e:
        return {"ok": False, "errors": [str(e)]}

    from rag_framework.pipeline.orchestrator import RAGPipeline
    results: list[dict] = []
    try:
        pipeline = RAGPipeline(config)
    except Exception as e:
        return {"ok": False, "errors": [str(e)], "modules": []}

    modules = {
        "parser": pipeline.parser,
        "chunker": pipeline.chunker,
        "embedder": pipeline.embedder,
        "vector_store": pipeline.vector_store,
        "retriever": pipeline.retriever,
        "evaluator": pipeline.evaluator,
    }
    all_ok = True
    for name, module in modules.items():
        try:
            module.health_check()
            results.append({"module": name, "ok": True, "message": "healthy"})
        except Exception as e:
            all_ok = False
            results.append({"module": name, "ok": False, "message": str(e)})

    if all_ok:
        state.set_pipeline(pipeline)

    return {"ok": all_ok, "modules": results}


@router.get("/export")
def export_config():
    yaml_str = state.get_config().to_yaml()
    name = state.get_config().name or "pipeline"
    return Response(
        content=yaml_str,
        media_type="text/yaml",
        headers={"Content-Disposition": f'attachment; filename="{name}.yaml"'},
    )


@router.post("/import")
async def import_config(file: UploadFile = File(...)):
    try:
        content = await file.read()
        config = PipelineConfig.from_yaml(content.decode("utf-8"))
        state.set_config(config)
        return {"ok": True, "config": config.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
