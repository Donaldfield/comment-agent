"""Pydantic v2 models for alerts and rule evaluation."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Alert(BaseModel):
    """An alert triggered by the rule engine."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: str = "info"  # info, warning, critical
    rule_name: str = ""
    title: str = ""
    detail: str = ""
    affected_review_ids: list[str] = Field(default_factory=list)
    remediation_plan: str = ""
    triggered_at: datetime = Field(default_factory=datetime.now)


class RuleContext(BaseModel):
    """Data bundle passed to every alert rule for evaluation."""
    product_id: str = ""
    time_window_hours: int = 24
    reviews: list = Field(default_factory=list)
    sentiment_results: list = Field(default_factory=list)
    pain_points: list = Field(default_factory=list)
    historical_baseline: dict = Field(default_factory=dict)
    anomaly_events: list = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
