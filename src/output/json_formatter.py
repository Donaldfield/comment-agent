"""Structured JSON output formatter.

Serializes analysis results as pretty-printed JSON for
programmatic consumption or piping to other tools.
"""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from src.models.analysis import AnalysisResultBundle


class EnhancedEncoder(json.JSONEncoder):
    """JSON encoder that handles Pydantic models and datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def to_json(bundle: AnalysisResultBundle, indent: int = 2) -> str:
    """Serialize an AnalysisResultBundle to pretty-printed JSON.

    Args:
        bundle: Complete analysis result bundle.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    return json.dumps(
        _bundle_to_dict(bundle),
        ensure_ascii=False,
        indent=indent,
        cls=EnhancedEncoder,
    )


def to_file(bundle: AnalysisResultBundle, output_path: str) -> None:
    """Write analysis results as JSON file.

    Args:
        bundle: Complete analysis result bundle.
        output_path: Path to .json output file.
    """
    json_str = to_json(bundle)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)


def _bundle_to_dict(bundle: AnalysisResultBundle) -> dict:
    """Convert bundle to serializable dict, handling dataclasses."""
    return {
        "product_id": bundle.product_id,
        "analysis_time": bundle.analysis_time.isoformat(),
        "summary": {
            "total_reviews": bundle.total_reviews,
            "valid_reviews": bundle.valid_reviews,
            "positive_count": bundle.positive_count,
            "neutral_count": bundle.neutral_count,
            "negative_count": bundle.negative_count,
            "positive_pct": _pct(bundle.positive_count, bundle.valid_reviews),
            "neutral_pct": _pct(bundle.neutral_count, bundle.valid_reviews),
            "negative_pct": _pct(bundle.negative_count, bundle.valid_reviews),
            "sentiment_score": _sentiment_score(
                bundle.positive_count, bundle.neutral_count, bundle.negative_count
            ),
        },
        "pain_points": [
            {
                "category": pp.category,
                "description": pp.description,
                "frequency": pp.frequency,
                "severity": pp.severity,
                "is_high_frequency": pp.is_high_frequency,
                "sentiment_trend": pp.sentiment_trend,
                "example_review_ids": pp.example_review_ids,
            }
            for pp in bundle.pain_points
        ],
        "keywords": [
            {"keyword": k.keyword, "frequency": k.frequency, "sentiment": k.associated_sentiment}
            for k in bundle.keywords[:20]
        ],
        "anomaly_events": [
            {
                "type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "description": e.description,
                "severity": e.severity,
            }
            for e in bundle.anomaly_events
        ],
        "alerts": [
            {
                "id": a.id[:8],
                "level": a.level,
                "rule_name": a.rule_name,
                "title": a.title,
                "detail": a.detail,
                "remediation_plan": a.remediation_plan,
            }
            for a in bundle.alerts
        ],
        "daily_stats": bundle.daily_stats,
        "cluster_summaries": bundle.cluster_summaries,
    }


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total:.1%}"


def _sentiment_score(positive: int, neutral: int, negative: int) -> float:
    total = positive + neutral + negative
    if total == 0:
        return 3.0
    return round((positive * 5 + neutral * 3 + negative * 1) / total, 1)
