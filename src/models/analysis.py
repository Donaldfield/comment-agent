"""Pydantic v2 models for analysis results."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """Overall sentiment classification for a single review."""
    review_id: str = ""
    sentiment: str = "neutral"  # positive, neutral, negative
    confidence: float = 0.5
    keywords: list[str] = Field(default_factory=list)


class KeywordResult(BaseModel):
    """Aggregate keyword extraction result."""
    keyword: str = ""
    frequency: int = 0
    associated_sentiment: str = ""


class PainPoint(BaseModel):
    """A mined pain point from review analysis."""
    category: str = "other"
    description: str = ""
    frequency: int = 0
    severity: str = "medium"  # low, medium, high
    is_high_frequency: bool = False
    example_review_ids: list[str] = Field(default_factory=list)
    sentiment_trend: str = "stable"
    representative_keywords: list[str] = Field(default_factory=list)


class AnomalyEvent(BaseModel):
    """A detected anomaly in review patterns."""
    event_type: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str = ""
    affected_review_ids: list[str] = Field(default_factory=list)
    severity: str = "medium"


class AnalysisResultBundle(BaseModel):
    """Complete analysis result for a product."""
    product_id: str = ""
    analysis_time: datetime = Field(default_factory=datetime.now)
    total_reviews: int = 0
    valid_reviews: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    sentiment_results: list[SentimentResult] = Field(default_factory=list)
    keywords: list[KeywordResult] = Field(default_factory=list)
    pain_points: list[PainPoint] = Field(default_factory=list)
    anomaly_events: list[AnomalyEvent] = Field(default_factory=list)
    alerts: list = Field(default_factory=list)
    daily_stats: list[dict] = Field(default_factory=list)
    cluster_summaries: list[dict] = Field(default_factory=list)
    absa_results: list = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
