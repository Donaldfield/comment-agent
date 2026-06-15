"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    """Request body for importing review data."""
    platform: str = Field(default="taobao", description="Platform: taobao, jd, pinduoduo, douyin")
    product_id: str = Field(default="", description="Product/SKU identifier")
    column_map: Optional[dict] = Field(default=None, description="Optional column name mapping")


class AnalyzeRequest(BaseModel):
    """Request body for starting analysis."""
    product_id: str = Field(..., description="Product/SKU identifier")
    platform: str = Field(default="taobao")


class TaskStatus(BaseModel):
    """Response for task status queries."""
    task_id: str
    status: str  # pending|running|done|error
    current_step: str
    retry_count: int = 0
    errors: list[dict] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class AnalyzeResult(BaseModel):
    """Response for analysis results."""
    product_id: str
    total_reviews: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    positive_pct: str = "0%"
    neutral_pct: str = "0%"
    negative_pct: str = "0%"
    sentiment_score: float = 0.0
    pain_points: list[dict] = Field(default_factory=list)
    alerts: list[dict] = Field(default_factory=list)
    keywords: list[dict] = Field(default_factory=list)
    daily_stats: list[dict] = Field(default_factory=list)
    absa_results: list[dict] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class RAGQuery(BaseModel):
    """Request body for RAG similarity search."""
    query: str = Field(..., description="Natural language query about product issues")
    product_id: str = Field(default="", description="Optional product filter")
    top_k: int = Field(default=10, ge=1, le=50)


class RAGResult(BaseModel):
    """Response for RAG query."""
    query: str
    similar_reviews: list[dict] = Field(default_factory=list)
    similar_pain_points: list[dict] = Field(default_factory=list)
    historical_issues: list[dict] = Field(default_factory=list)
    suggestion: str = ""


class ReviewStats(BaseModel):
    """Review statistics for a product."""
    product_id: str
    total_reviews: int = 0
    valid_reviews: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    rating_distribution: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    llm_available: bool = False
    milvus_available: bool = False
