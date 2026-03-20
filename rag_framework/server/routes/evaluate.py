"""
Evaluation API routes — run and compare evaluation reports.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_framework.config.models import PipelineConfig
from rag_framework.modules.evaluation.evaluator import RAGEvaluator
from rag_framework.pipeline.orchestrator import RAGPipeline
from rag_framework.server import state

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


class EvalItem(BaseModel):
    query: str
    expected: list[str]


class CompareRequest(BaseModel):
    eval_set: list[EvalItem]
    config_b: dict  # raw dict for Config B, Config A is always the active session config


@router.post("/compare")
def compare(body: CompareRequest):
    if not body.eval_set:
        raise HTTPException(status_code=400, detail="eval_set is empty")

    queries = [item.query for item in body.eval_set]
    expected = [item.expected for item in body.eval_set]

    # Config A = active session config
    config_a = state.get_config()

    # Config B = from request body
    try:
        config_b = PipelineConfig.model_validate(body.config_b)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Config B invalid: {e}")

    results = {}
    for label, config in [("a", config_a), ("b", config_b)]:
        try:
            pipeline = RAGPipeline(config)
            pipeline.validate()
            retrieved = [pipeline.query(q) for q in queries]
            evaluator = RAGEvaluator(config.evaluator)
            report = evaluator.evaluate(
                queries=queries,
                expected=expected,
                retrieved=retrieved,
            )
            results[label] = report.metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Config {label.upper()} eval failed: {e}")

    # Compute deltas
    all_keys = sorted(set(results["a"]) | set(results["b"]))
    comparison = []
    for key in all_keys:
        a_val = results["a"].get(key, 0.0)
        b_val = results["b"].get(key, 0.0)
        delta = b_val - a_val
        comparison.append({
            "metric": key,
            "a": round(a_val, 4),
            "b": round(b_val, 4),
            "delta": round(delta, 4),
            "better": "b" if delta > 0.001 else ("a" if delta < -0.001 else "tie"),
        })

    return {
        "ok": True,
        "config_a_name": config_a.name,
        "config_b_name": config_b.name,
        "comparison": comparison,
    }
