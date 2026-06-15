"""Function Calling tool — incremental review scraping.

Currently a stub. In production, this would integrate with platform-specific
scrapers (Taobao API, JD API, etc.) to fetch new reviews since a given date.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def scrape_incremental_reviews(product_id: str, platform: str, since_date: str) -> dict:
    """Fetch new reviews for a product since a given date.

    Args:
        product_id: Product/SKU identifier.
        platform: Platform name (taobao, jd, pinduoduo, douyin).
        since_date: ISO format date string to fetch reviews from.

    Returns:
        Dict with 'reviews' list and 'count' int.
    """
    logger.info("Scrape requested: product=%s platform=%s since=%s (STUB)",
                product_id, platform, since_date)

    # In production, this would call platform-specific APIs.
    # For now, returns empty — reviews come from CSV uploads.
    return {
        "count": 0,
        "reviews": [],
        "message": "Incremental scraping is not yet implemented. Use CSV upload via POST /api/v1/import.",
    }
