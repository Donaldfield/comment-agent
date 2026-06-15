"""Time-series anomaly detection for review sentiment and volume.

Detects two anomaly types:
1. Sentiment shift: Negative ratio spikes beyond historical baseline
2. Volume spike: Review volume suddenly exceeds baseline
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.models.review import ReviewRecord
from src.models.analysis import AnomalyEvent, SentimentResult

logger = logging.getLogger(__name__)


def detect_sentiment_shift(
    sentiment_results: list[SentimentResult],
    reviews: list[ReviewRecord],
    granularity: str = "daily",
    stddev_multiplier: float = 2.0,
    historical_baseline: Optional[dict] = None,
) -> list[AnomalyEvent]:
    """Detect time periods where negative sentiment ratio is abnormally high.

    Groups reviews by time bucket, computes negative ratio per bucket,
    and flags buckets where the ratio exceeds historical mean + N*stddev.

    Args:
        sentiment_results: Sentiment classifications.
        reviews: Review records with timestamps.
        granularity: "hourly" or "daily" time buckets.
        stddev_multiplier: Number of standard deviations for anomaly threshold.
        historical_baseline: Optional pre-computed baseline stats.

    Returns:
        List of AnomalyEvent for periods with abnormal negative sentiment.
    """
    if not sentiment_results or not reviews:
        return []

    # Build review lookup by ID
    review_map = {r.id: r for r in reviews}
    sentiment_map = {s.review_id: s.sentiment for s in sentiment_results}

    # Group by time bucket
    bucket_key = "%Y-%m-%d" if granularity == "daily" else "%Y-%m-%dT%H"
    buckets: dict[str, dict] = defaultdict(lambda: {"total": 0, "negative": 0, "review_ids": []})

    for sr in sentiment_results:
        review = review_map.get(sr.review_id)
        if not review:
            continue
        key = review.created_at.strftime(bucket_key)
        buckets[key]["total"] += 1
        buckets[key]["review_ids"].append(sr.review_id)
        if sr.sentiment == "negative":
            buckets[key]["negative"] += 1

    if len(buckets) < 2:
        return []

    # Compute statistics across all buckets
    sorted_keys = sorted(buckets.keys())
    neg_ratios = [
        buckets[k]["negative"] / max(buckets[k]["total"], 1)
        for k in sorted_keys
    ]

    avg_ratio = np.mean(neg_ratios)
    std_ratio = np.std(neg_ratios) if len(neg_ratios) > 1 else 0.1

    # Use historical baseline if significantly different
    if historical_baseline and "avg_neg_ratio" in historical_baseline:
        avg_ratio = historical_baseline["avg_neg_ratio"]

    # Detect anomalies
    threshold = avg_ratio + stddev_multiplier * std_ratio
    anomalies: list[AnomalyEvent] = []

    for key in sorted_keys:
        bucket = buckets[key]
        ratio = bucket["negative"] / max(bucket["total"], 1)

        if ratio > threshold and bucket["total"] >= 3:
            severity = "high" if ratio > threshold * 1.5 else "medium"
            try:
                ts = datetime.strptime(key, bucket_key)
            except ValueError:
                ts = datetime.now()

            anomalies.append(AnomalyEvent(
                event_type="sentiment_shift",
                timestamp=ts,
                description=(
                    f"Negative sentiment ratio {ratio:.1%} exceeds threshold {threshold:.1%} "
                    f"(avg {avg_ratio:.1%}). {bucket['negative']}/{bucket['total']} negative."
                ),
                affected_review_ids=bucket["review_ids"],
                severity=severity,
            ))

    logger.info(
        "Detected %d sentiment shift anomalies (threshold=%.1f%%)",
        len(anomalies), threshold * 100,
    )
    return anomalies


def detect_volume_spike(
    reviews: list[ReviewRecord],
    granularity: str = "hourly",
    multiplier: float = 3.0,
    historical_baseline: Optional[dict] = None,
) -> list[AnomalyEvent]:
    """Detect time periods with abnormally high review volume.

    Args:
        reviews: Review records with timestamps.
        granularity: "hourly" or "daily" time buckets.
        multiplier: Volume must exceed baseline * multiplier to trigger.
        historical_baseline: Optional pre-computed avg daily volume.

    Returns:
        List of AnomalyEvent for periods with abnormal volume.
    """
    if not reviews:
        return []

    bucket_key = "%Y-%m-%dT%H" if granularity == "hourly" else "%Y-%m-%d"
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "review_ids": []})

    for review in reviews:
        key = review.created_at.strftime(bucket_key)
        buckets[key]["count"] += 1
        buckets[key]["review_ids"].append(review.id)

    if len(buckets) < 3:
        return []

    # Compute baseline
    counts = [b["count"] for b in buckets.values()]
    avg_count = np.mean(counts)

    if historical_baseline and "avg_daily_volume" in historical_baseline:
        avg_count = historical_baseline["avg_daily_volume"]

    threshold = avg_count * multiplier
    anomalies: list[AnomalyEvent] = []

    for key, bucket in buckets.items():
        if bucket["count"] >= threshold and bucket["count"] >= 5:
            severity = "high" if bucket["count"] > threshold * 2 else "medium"
            try:
                ts = datetime.strptime(key, bucket_key)
            except ValueError:
                ts = datetime.now()

            anomalies.append(AnomalyEvent(
                event_type="volume_spike",
                timestamp=ts,
                description=(
                    f"Review volume {bucket['count']} exceeds threshold {threshold:.0f} "
                    f"(avg {avg_count:.1f}, {multiplier}x baseline)"
                ),
                affected_review_ids=bucket["review_ids"],
                severity=severity,
            ))

    if anomalies:
        logger.info("Detected %d volume spike anomalies (threshold=%.0f)", len(anomalies), threshold)

    return anomalies
