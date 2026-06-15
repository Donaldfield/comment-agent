"""FastAPI application — E-Commerce Review Analysis System v2."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.api.routes import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting E-Commerce Review Analysis System v2")
    logger.info("LLM: DeepSeek (%s)", settings.deepseek_model)
    logger.info("Milvus: %s:%d", settings.milvus_host, settings.milvus_port)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="E-Commerce Review Analysis API",
    description="基于 LangGraph 多智能体的电商评价智能分析系统",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the HTML dashboard."""
    html_path = Path(__file__).parent / "dashboard" / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("""<html><body style="font-family:sans-serif;padding:40px">
        <h1>E-Commerce Review Analysis Dashboard</h1>
        <p>Dashboard coming in Phase 4. Use <a href="/docs">/docs</a> for API reference.</p>
        </body></html>""", status_code=200)
