"""Platform-specific field normalization.

Unifies rating schemes, timestamps, and platform tags across
Taobao, JD, Pinduoduo, and Douyin review formats.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.models.review import ReviewRecord

logger = logging.getLogger(__name__)

# Platform name normalization
PLATFORM_ALIASES: dict[str, str] = {
    "taobao": "taobao",
    "淘宝": "taobao",
    "tmall": "taobao",
    "天猫": "taobao",
    "jd": "jd",
    "京东": "jd",
    "pinduoduo": "pinduoduo",
    "拼多多": "pinduoduo",
    "pdd": "pinduoduo",
    "douyin": "douyin",
    "抖音": "douyin",
    "dy": "douyin",
}


def normalize(records: list[ReviewRecord]) -> list[ReviewRecord]:
    """Normalize all records to a unified schema.

    - Unifies platform names
    - Ensures UTC timestamps
    - Fills missing review_type from rating
    - Deduplicates by content hash (cross-platform)
    """
    normalized: list[ReviewRecord] = []
    seen_content: set[str] = set()

    for record in records:
        # Normalize platform name
        platform = _normalize_platform(record.platform)

        # Normalize timestamp to UTC if timezone-aware
        created_at = record.created_at
        if created_at.tzinfo is not None:
            created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)

        # Infer review_type if empty
        review_type = record.review_type
        if not review_type:
            review_type = _infer_review_type(record.rating)

        # Cross-platform dedup by content
        content_key = record.content.strip().lower()
        if content_key in seen_content:
            logger.debug("Cross-platform duplicate skipped: %s", record.id)
            continue
        seen_content.add(content_key)

        normalized.append(
            ReviewRecord(
                id=record.id,
                platform=platform,
                product_id=record.product_id,
                content=record.content.strip(),
                rating=record.rating,
                review_type=review_type,
                created_at=created_at,
                imported_at=record.imported_at,
                metadata=dict(record.metadata),
                is_valid=record.is_valid,
            )
        )

    logger.info("Normalized %d records (%d duplicates removed)", len(normalized), len(records) - len(normalized))
    return normalized


def _normalize_platform(name: str) -> str:
    """Map platform aliases to canonical names."""
    return PLATFORM_ALIASES.get(name.lower().strip(), name.lower().strip())


def _infer_review_type(rating: int) -> str:
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    return "negative"
