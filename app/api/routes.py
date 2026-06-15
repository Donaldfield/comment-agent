"""FastAPI route handlers for the review analysis system."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.api.schemas import (
    TaskStatus, AnalyzeResult, RAGQuery, ReviewStats, HealthResponse,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory task store (simple dict for lightweight deployment)
_tasks: dict = {}


def _make_task(product_id: str, platform: str = "taobao") -> str:
    """Create a new task and return its ID."""
    task_id = str(uuid.uuid4())[:12]
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "current_step": "",
        "retry_count": 0,
        "errors": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    return task_id


# ── Import ──

@router.post("/import")
async def import_reviews(
    file: UploadFile = File(...),
    platform: str = Form(default="taobao"),
    product_id: str = Form(default=""),
    column_map: Optional[str] = Form(default=None),
):
    """Upload a CSV/Excel file and import reviews."""
    import json
    import shutil

    settings = get_settings()
    import_dir = Path(settings.import_dir)
    import_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = import_dir / f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cm = None
    if column_map:
        try:
            cm = json.loads(column_map)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid column_map JSON")

    # Run collector agent synchronously (lightweight, no LLM needed)
    from app.agents.collector import collector_node

    state = {
        "task_id": _make_task(product_id, platform),
        "product_id": product_id,
        "platform": platform,
        "status": "running",
        "current_step": "collector",
        "retry_count": 0,
        "max_retries": 2,
        "errors": [],
        "source_file": str(file_path),
        "raw_reviews": [],
        "cleaned_reviews": [],
        "sentiment_results": [],
        "absa_results": [],
        "keyword_results": [],
        "pain_points": [],
        "anomaly_events": [],
        "alerts": [],
        "report_path": "",
        "historical_issues": [],
        "improvement_suggestions": [],
    }

    result = await collector_node(state)

    task_id = state["task_id"]
    _tasks[task_id].update({
        "status": result.get("status", "error"),
        "current_step": "collector",
        "errors": result.get("errors", []),
        "updated_at": datetime.now().isoformat(),
    })

    if result.get("status") == "error":
        raise HTTPException(400, detail=result.get("errors", [{"message": "Import failed"}]))

    raw_count = len(result.get("raw_reviews", []))
    return {
        "task_id": task_id,
        "status": "ok",
        "reviews_imported": raw_count,
        "file": str(file_path),
    }


# ── Analyze ──

@router.post("/analyze/{product_id}")
async def start_analysis(
    product_id: str,
    platform: str = Query(default="taobao"),
    source_file: str = Query(default=""),
):
    """Start a full 5-agent analysis pipeline for a product."""
    from app.agents.graph import get_graph

    task_id = _make_task(product_id, platform)

    # Build initial state
    state = {
        "task_id": task_id,
        "product_id": product_id,
        "platform": platform,
        "status": "running",
        "current_step": "collector",
        "retry_count": 0,
        "max_retries": 2,
        "errors": [],
        "source_file": source_file,
        "raw_reviews": [],
        "cleaned_reviews": [],
        "sentiment_results": [],
        "absa_results": [],
        "keyword_results": [],
        "pain_points": [],
        "anomaly_events": [],
        "alerts": [],
        "report_path": "",
        "historical_issues": [],
        "improvement_suggestions": [],
    }

    # If reviews already in DB, skip collector and start from cleaner
    from src.memory.repository import ReviewRepository
    repo = ReviewRepository(get_settings().db_path)
    existing = repo.get_reviews(product_id=product_id, valid_only=True)
    repo.close()

    if existing:
        state["raw_reviews"] = [r.model_dump() for r in existing]
        state["current_step"] = "cleaner"
        state["source_file"] = ""  # Already in DB

    graph = get_graph()

    try:
        final_state = await graph.ainvoke(state)

        # Save analysis results to DB for later retrieval
        repo = ReviewRepository(get_settings().db_path)
        repo.save_analysis_result(product_id, "pain_points",
                                   {"items": final_state.get("pain_points", [])})
        repo.save_analysis_result(product_id, "keywords",
                                   {"items": final_state.get("keyword_results", [])})
        repo.save_analysis_result(product_id, "alerts",
                                   {"items": final_state.get("alerts", [])})
        repo.close()

        _tasks[task_id].update({
            "status": final_state.get("status", "error"),
            "current_step": final_state.get("current_step", ""),
            "errors": final_state.get("errors", []),
            "updated_at": datetime.now().isoformat(),
        })
        return {
            "task_id": task_id,
            "status": final_state.get("status", "done"),
            "report_path": final_state.get("report_path", ""),
            "alerts_count": len(final_state.get("alerts", [])),
        }
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        _tasks[task_id].update({
            "status": "error",
            "errors": [{"step": "pipeline", "message": str(e)}],
            "updated_at": datetime.now().isoformat(),
        })
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
        }


# ── Status ──

@router.get("/analyze/{task_id}/status", response_model=TaskStatus)
async def get_analysis_status(task_id: str):
    """Get the current status of an analysis task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return TaskStatus(**task)


# ── Results ──

@router.get("/analysis/{product_id}", response_model=AnalyzeResult)
async def get_analysis_results(
    product_id: str,
    days: int = Query(default=30),
):
    """Get full analysis results for a product."""
    from src.memory.repository import ReviewRepository
    from app.config import get_settings

    settings = get_settings()
    repo = ReviewRepository(settings.db_path)

    reviews = repo.get_reviews(product_id=product_id, valid_only=True)
    if not reviews:
        repo.close()
        return AnalyzeResult(product_id=product_id)

    pos = sum(1 for r in reviews if r.rating >= 4)
    neu = sum(1 for r in reviews if r.rating == 3)
    neg = sum(1 for r in reviews if r.rating <= 2)
    total = len(reviews)

    # Load stored analysis results from DB
    import json
    pain_points = _load_analysis_data(repo, product_id, "pain_points")
    keywords = _load_analysis_data(repo, product_id, "keywords")
    alerts = _load_analysis_data(repo, product_id, "alerts")

    daily_stats = repo.get_daily_stats(product_id=product_id, days=days)
    repo.close()

    score = round((pos * 5 + neu * 3 + neg * 1) / max(total, 1), 1)

    return AnalyzeResult(
        product_id=product_id,
        total_reviews=total,
        positive_count=pos,
        neutral_count=neu,
        negative_count=neg,
        positive_pct=f"{pos/max(total,1):.1%}",
        neutral_pct=f"{neu/max(total,1):.1%}",
        negative_pct=f"{neg/max(total,1):.1%}",
        sentiment_score=score,
        pain_points=pain_points,
        keywords=keywords,
        alerts=alerts,
        daily_stats=daily_stats,
    )


# ── ABSA (Aspect-Based Sentiment) ──

@router.get("/analysis/{product_id}/absa")
async def get_absa_results(product_id: str):
    """Get per-aspect sentiment distribution for a product.

    Computed from keyword matching against reviews — no LLM needed.
    """
    from src.memory.repository import ReviewRepository
    from app.config import get_settings

    settings = get_settings()
    repo = ReviewRepository(settings.db_path)
    reviews = repo.get_reviews(product_id=product_id, valid_only=True)
    repo.close()

    aspects = {
        "quality": {"cn": "质量", "keywords": settings.painpoint_categories.get("quality", [])},
        "logistics": {"cn": "物流", "keywords": settings.painpoint_categories.get("logistics", [])},
        "packaging": {"cn": "包装", "keywords": settings.painpoint_categories.get("logistics", [])[:3]},
        "size": {"cn": "尺码", "keywords": settings.painpoint_categories.get("size", [])},
        "color": {"cn": "色差", "keywords": settings.painpoint_categories.get("color_diff", [])},
        "service": {"cn": "客服", "keywords": settings.painpoint_categories.get("service", [])},
        "value": {"cn": "性价比", "keywords": settings.painpoint_categories.get("value_for_money", [])},
    }

    radar_data = []
    for aspect_key, aspect_info in aspects.items():
        keywords = aspect_info.get("keywords", [])
        positive = 0
        negative = 0
        mentioned = 0

        for review in reviews:
            matched = any(kw in review.content for kw in keywords)
            if matched:
                mentioned += 1
                if review.rating >= 4:
                    positive += 1
                elif review.rating <= 2:
                    negative += 1

        total = len(reviews) if reviews else 1
        radar_data.append({
            "aspect": aspect_key,
            "aspect_cn": aspect_info["cn"],
            "positive_pct": round(positive / max(mentioned, 1) * 100, 1) if mentioned else 0,
            "negative_pct": round(negative / max(mentioned, 1) * 100, 1) if mentioned else 0,
            "total_mentioned": mentioned,
            "mention_pct": round(mentioned / total * 100, 1),
        })

    return {
        "product_id": product_id,
        "radar_data": radar_data,
    }


# ── Reviews ──

@router.get("/reviews/{product_id}")
async def list_reviews(
    product_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    rating: Optional[int] = Query(default=None),
):
    """List reviews for a product with pagination and filtering."""
    from src.memory.repository import ReviewRepository

    repo = ReviewRepository(get_settings().db_path)
    all_reviews = repo.get_reviews(product_id=product_id, valid_only=True)
    repo.close()

    if rating is not None:
        all_reviews = [r for r in all_reviews if r.rating == rating]

    start = (page - 1) * page_size
    end = start + page_size
    page_reviews = all_reviews[start:end]

    return {
        "product_id": product_id,
        "total": len(all_reviews),
        "page": page,
        "page_size": page_size,
        "reviews": [r.model_dump() for r in page_reviews],
    }


@router.get("/reviews/{product_id}/stats", response_model=ReviewStats)
async def get_review_stats(product_id: str):
    """Get review statistics for a product."""
    from src.memory.repository import ReviewRepository
    from collections import Counter

    repo = ReviewRepository(get_settings().db_path)
    reviews = repo.get_reviews(product_id=product_id, valid_only=True)
    repo.close()

    if not reviews:
        return ReviewStats(product_id=product_id)

    valid = [r for r in reviews if r.is_valid]
    rating_dist = Counter(r.rating for r in valid)

    return ReviewStats(
        product_id=product_id,
        total_reviews=len(reviews),
        valid_reviews=len(valid),
        positive_count=sum(1 for r in valid if r.rating >= 4),
        neutral_count=sum(1 for r in valid if r.rating == 3),
        negative_count=sum(1 for r in valid if r.rating <= 2),
        rating_distribution={str(k): v for k, v in sorted(rating_dist.items())},
    )


# ── Reports ──

@router.get("/reports/{product_id}/excel")
async def download_excel(product_id: str):
    """Download the Excel report for a product."""
    settings = get_settings()
    report_dir = Path(settings.report_dir)
    # Find the latest report for this product
    files = sorted(report_dir.glob(f"{product_id}_analysis_*.xlsx"), reverse=True)
    if not files:
        raise HTTPException(404, f"No report found for {product_id}")
    return FileResponse(
        str(files[0]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=files[0].name,
    )


@router.get("/reports/{product_id}/json")
async def download_json(product_id: str):
    """Download the JSON report for a product."""
    settings = get_settings()
    report_dir = Path(settings.report_dir)
    files = sorted(report_dir.glob(f"{product_id}_analysis_*.json"), reverse=True)
    if not files:
        raise HTTPException(404, f"No JSON report found for {product_id}")
    return FileResponse(
        str(files[0]),
        media_type="application/json",
        filename=files[0].name,
    )


def _load_analysis_data(repo, product_id: str, result_type: str) -> list[dict]:
    """Load stored analysis results from DB."""
    import sqlite3, json
    try:
        conn = sqlite3.connect(repo._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT result_data FROM analysis_results WHERE product_id=? AND result_type=? ORDER BY id DESC LIMIT 1",
            (product_id, result_type)
        ).fetchone()
        conn.close()
        if row:
            data = json.loads(row["result_data"])
            return data.get("items", [])
    except Exception:
        pass
    return []


# ── RAG ──

@router.post("/rag/query")
async def rag_query(body: RAGQuery):
    """Query for similar reviews and issues using keyword matching + TF-IDF.

    Works without Milvus — uses local SQLite + jieba tokenization.
    """
    from src.memory.repository import ReviewRepository
    from app.config import get_settings
    import jieba
    from collections import Counter

    settings = get_settings()
    repo = ReviewRepository(settings.db_path)

    reviews = repo.get_reviews(product_id=body.product_id or "", valid_only=True)
    if not reviews:
        repo.close()
        return {"query": body.query, "similar_reviews": [], "similar_pain_points": [], "historical_issues": []}

    # Tokenize query and each review
    query_tokens = set(jieba.cut(body.query))

    scored = []
    for review in reviews:
        review_tokens = set(jieba.cut(review.content))
        overlap = len(query_tokens & review_tokens)
        if overlap > 0:
            scored.append({
                "id": review.id,
                "text": review.content,
                "rating": review.rating,
                "score": overlap,
                "created_at": review.created_at.isoformat(),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Load stored pain points from DB
    pain_points = _load_analysis_data(repo, body.product_id, "pain_points")

    # Find relevant pain points by keyword overlap
    relevant_pp = []
    for pp in pain_points:
        pp_tokens = set(jieba.cut(pp.get("description", "") + pp.get("category", "")))
        if query_tokens & pp_tokens:
            relevant_pp.append(pp)

    repo.close()

    return {
        "query": body.query,
        "similar_reviews": scored[:body.top_k],
        "similar_pain_points": relevant_pp[:5],
        "historical_issues": [],
    }


# ── Health ──

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check with LLM and Milvus status."""
    llm_ok = False
    try:
        from app.llm.client import get_llm_client
        get_llm_client()
        llm_ok = True
    except Exception:
        pass

    milvus_ok = False
    try:
        from pymilvus import connections
        settings = get_settings()
        connections.connect(host=settings.milvus_host, port=settings.milvus_port, timeout=2)
        milvus_ok = True
        connections.disconnect("default")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="2.0.0",
        llm_available=llm_ok,
        milvus_available=milvus_ok,
    )
