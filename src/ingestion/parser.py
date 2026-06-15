"""Multi-format review data parser.

Supports CSV and Excel files with configurable column mapping
for platform-specific column names (e.g., JD uses "score",
Pinduoduo uses "star_level").
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.review import ReviewRecord

logger = logging.getLogger(__name__)

# Default column mappings for each platform
PLATFORM_DEFAULTS: dict[str, dict[str, str]] = {
    "taobao": {
        "content": "content",
        "rating": "rating",
        "created_at": "created_at",
        "product_id": "product_id",
    },
    "jd": {
        "content": "content",
        "rating": "score",
        "created_at": "creation_time",
        "product_id": "sku_id",
    },
    "pinduoduo": {
        "content": "content",
        "rating": "star_level",
        "created_at": "review_time",
        "product_id": "goods_id",
    },
    "douyin": {
        "content": "content",
        "rating": "rating",
        "created_at": "create_time",
        "product_id": "product_id",
    },
}


def parse_file(
    filepath: str,
    platform: str,
    product_id: str = "",
    column_map: Optional[dict] = None,
) -> list[ReviewRecord]:
    """Parse a CSV or Excel file into ReviewRecord list.

    Args:
        filepath: Path to CSV (.csv) or Excel (.xlsx/.xls) file.
        platform: Platform name (taobao, jd, pinduoduo, douyin).
        product_id: Fallback product ID if not in file.
        column_map: Dict mapping standard fields to file column names.
                    If None, uses PLATFORM_DEFAULTS for the given platform.

    Returns:
        List of ReviewRecord (with raw content, not yet cleaned).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = _read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .xlsx, or .xls")

    if df.empty:
        logger.warning("Empty file: %s", filepath)
        return []

    # Resolve column mapping
    mapping = column_map or PLATFORM_DEFAULTS.get(platform, PLATFORM_DEFAULTS["taobao"])

    records = []
    for idx, row in df.iterrows():
        try:
            record = _row_to_record(row, mapping, platform, product_id)
            if record:
                records.append(record)
        except Exception as e:
            logger.warning("Skipping row %d: %s", idx, e)

    logger.info("Parsed %d records from %s (platform=%s)", len(records), filepath, platform)
    return records


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with encoding detection."""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            # Try reading with pandas csv reader
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def _row_to_record(
    row: pd.Series,
    mapping: dict,
    platform: str,
    fallback_product_id: str,
) -> Optional[ReviewRecord]:
    """Convert a single DataFrame row to a ReviewRecord."""
    content_col = mapping.get("content", "content")
    rating_col = mapping.get("rating", "rating")
    time_col = mapping.get("created_at", "created_at")
    product_col = mapping.get("product_id", "product_id")

    content = str(row.get(content_col, "")).strip()
    if not content or content in ("nan", "None", ""):
        return None

    # Parse rating
    rating = _parse_rating(row.get(rating_col, 3))

    # Parse timestamp
    created_at = _parse_timestamp(row.get(time_col))

    # Determine product ID
    pid = str(row.get(product_col, fallback_product_id) or fallback_product_id).strip()

    # Infer review type from rating
    review_type = _infer_review_type(rating)

    return ReviewRecord(
        content=content,
        rating=rating,
        created_at=created_at,
        platform=platform,
        product_id=pid,
        review_type=review_type,
        metadata={"source_file_column": str(row.to_dict()) if False else ""},
    )


def _parse_rating(value) -> int:
    """Normalize rating to 1-5 integer."""
    if value is None:
        return 3
    try:
        r = float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return 3

    # Handle 0-1 scale (e.g., 0.8 -> 4)
    if 0 <= r <= 1:
        r = r * 4 + 1
    # Handle 0-10 scale
    elif r > 5 and r <= 10:
        r = r / 2
    # Handle percentage
    elif r > 10 and r <= 100:
        r = r / 20

    r = max(1, min(5, round(r)))
    return int(r)


def _parse_timestamp(value) -> datetime:
    """Parse timestamp from various formats."""
    if value is None:
        return datetime.now()

    if isinstance(value, datetime):
        return value

    s = str(value).strip()

    # Common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # Try pandas Timestamp
    try:
        return pd.Timestamp(s).to_pydatetime()
    except (ValueError, TypeError):
        pass

    logger.debug("Could not parse timestamp: %s, using current time", s)
    return datetime.now()


def _infer_review_type(rating: int) -> str:
    """Infer review type from star rating."""
    if rating >= 5:
        return "positive"
    elif rating == 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    elif rating <= 2:
        return "negative"
    return "neutral"
