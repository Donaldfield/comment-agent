"""ABSA aggregator — compute per-aspect sentiment distributions."""

import logging
from collections import defaultdict

from app.absa.extractor import ASPECTS

logger = logging.getLogger(__name__)


def aggregate_absa(absa_results: list[dict]) -> dict:
    """Aggregate ABSA results across all reviews.

    Args:
        absa_results: List of per-review ABSA results from extract_absa().

    Returns:
        Dict with per-aspect sentiment counts and radar chart data.
    """
    if not absa_results:
        return {"aspect_summary": {}, "radar_data": [], "top_negative_aspects": []}

    # Per-aspect sentiment counts
    aspect_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "neutral": 0, "negative": 0, "not_mentioned": 0}
    )

    for result in absa_results:
        aspects = result.get("aspects", {})
        for aspect in ASPECTS:
            sentiment = aspects.get(aspect, {}).get("sentiment", "not_mentioned")
            if sentiment in aspect_summary[aspect]:
                aspect_summary[aspect][sentiment] += 1

    # Radar chart data: positive ratio per aspect (0-100)
    radar_data = []
    for aspect in ASPECTS:
        summary = aspect_summary[aspect]
        total_mentioned = summary["positive"] + summary["neutral"] + summary["negative"]
        if total_mentioned > 0:
            positive_ratio = round(summary["positive"] / total_mentioned * 100, 1)
            negative_ratio = round(summary["negative"] / total_mentioned * 100, 1)
        else:
            positive_ratio = 0
            negative_ratio = 0

        radar_data.append({
            "aspect": aspect,
            "aspect_cn": _aspect_cn(aspect),
            "positive_pct": positive_ratio,
            "negative_pct": negative_ratio,
            "total_mentioned": total_mentioned,
        })

    # Top negative aspects (sorted by negative count descending)
    top_negative = sorted(
        [(aspect, summary["negative"])
         for aspect, summary in aspect_summary.items()
         if summary["negative"] > 0],
        key=lambda x: x[1], reverse=True,
    )

    top_negative_aspects = [
        {"aspect": aspect, "aspect_cn": _aspect_cn(aspect), "negative_count": count}
        for aspect, count in top_negative
    ]

    logger.info("ABSA aggregation: %d reviews, top negative=%s",
                len(absa_results),
                [a["aspect"] for a in top_negative_aspects[:3]])

    return {
        "aspect_summary": dict(aspect_summary),
        "radar_data": radar_data,
        "top_negative_aspects": top_negative_aspects,
    }


def _aspect_cn(aspect: str) -> str:
    """Get Chinese label for an aspect."""
    labels = {
        "quality": "质量",
        "logistics": "物流",
        "packaging": "包装",
        "size": "尺码",
        "color": "色差",
        "service": "客服",
        "value": "性价比",
    }
    return labels.get(aspect, aspect)
