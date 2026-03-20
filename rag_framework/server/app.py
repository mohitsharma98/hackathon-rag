"""
FastAPI application — serves the dark dashboard UI and REST API.

Run with:
    uvicorn rag_framework.server.app:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_framework.server.routes import config, pipeline, evaluate

BASE_DIR = Path(__file__).parent

app = FastAPI(title="RAG Framework", docs_url="/api/docs")

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# API routers
app.include_router(config.router)
app.include_router(pipeline.router)
app.include_router(evaluate.router)


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def page_configure(request: Request):
    return templates.TemplateResponse("configure.html", {"request": request})


@app.get("/evaluate", response_class=HTMLResponse)
async def page_evaluate(request: Request):
    return templates.TemplateResponse("evaluate.html", {"request": request})
