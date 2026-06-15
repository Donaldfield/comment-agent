"""SQLite-based review persistence.

Stores ReviewRecords and analysis results in a local SQLite database.
Zero setup required — the database file is created on first use.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.models.review import ReviewRecord
from src.models.analysis import SentimentResult, PainPoint, AnalysisResultBundle

logger = logging.getLogger(__name__)


class ReviewRepository:
    """SQLite-backed repository for reviews and analysis results."""

    def __init__(self, db_path: str = "data/reviews.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL DEFAULT 'unknown',
                product_id TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                review_type TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                is_valid INTEGER NOT NULL DEFAULT 1,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                analysis_time TEXT NOT NULL,
                result_type TEXT NOT NULL,
                result_data TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_product
                ON reviews(product_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_created
                ON reviews(created_at);
            CREATE INDEX IF NOT EXISTS idx_reviews_valid
                ON reviews(is_valid);
            CREATE INDEX IF NOT EXISTS idx_analysis_product
                ON analysis_results(product_id, result_type);
        """)
        self._conn.commit()

    # ── Write operations ──────────────────────────────────────────────

    def insert_reviews(self, records: list[ReviewRecord]) -> int:
        """Insert or replace review records. Returns count inserted."""
        rows = [
            (
                r.id, r.platform, r.product_id, r.content, r.rating,
                r.review_type, r.created_at.isoformat(), r.imported_at.isoformat(),
                int(r.is_valid), json.dumps(r.metadata, ensure_ascii=False)
            )
            for r in records
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO reviews
               (id, platform, product_id, content, rating, review_type,
                created_at, imported_at, is_valid, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        logger.info("Inserted/updated %d reviews", len(rows))
        return len(rows)

    def update_review_validity(self, review_id: str, is_valid: bool, cleaned_content: str = "") -> None:
        """Mark a review as valid/invalid and optionally update content."""
        if cleaned_content:
            self._conn.execute(
                "UPDATE reviews SET is_valid = ?, content = ? WHERE id = ?",
                (int(is_valid), cleaned_content, review_id),
            )
        else:
            self._conn.execute(
                "UPDATE reviews SET is_valid = ? WHERE id = ?",
                (int(is_valid), review_id),
            )
        self._conn.commit()

    def save_analysis_result(self, product_id: str, result_type: str, result_data: dict) -> None:
        """Persist an analysis result as JSON."""
        self._conn.execute(
            "INSERT INTO analysis_results (product_id, analysis_time, result_type, result_data) "
            "VALUES (?, ?, ?, ?)",
            (
                product_id,
                datetime.now().isoformat(),
                result_type,
                json.dumps(result_data, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    # ── Read operations ───────────────────────────────────────────────

    def get_reviews(
        self,
        product_id: str = "",
        valid_only: bool = True,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 0,
    ) -> list[ReviewRecord]:
        """Query reviews with optional filters."""
        query = "SELECT * FROM reviews WHERE 1=1"
        params: list = []

        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if valid_only:
            query += " AND is_valid = 1"
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from.isoformat())
        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to.isoformat())

        query += " ORDER BY created_at DESC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_review(r) for r in rows]

    def get_negative_reviews(
        self,
        product_id: str = "",
        date_from: Optional[datetime] = None,
    ) -> list[ReviewRecord]:
        """Get reviews classified as negative (rating <= 2 or review_type = 'negative')."""
        query = "SELECT * FROM reviews WHERE is_valid = 1 AND (rating <= 2 OR review_type = 'negative')"
        params: list = []

        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from.isoformat())

        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_review(r) for r in rows]

    def get_review_count(
        self,
        product_id: str = "",
        date_from: Optional[datetime] = None,
    ) -> int:
        """Count reviews matching filters."""
        query = "SELECT COUNT(*) as cnt FROM reviews WHERE is_valid = 1"
        params: list = []

        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from.isoformat())

        row = self._conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    def get_daily_stats(
        self,
        product_id: str = "",
        days: int = 30,
    ) -> list[dict]:
        """Get daily sentiment breakdown for the past N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = """
            SELECT
                DATE(created_at) as date,
                COUNT(*) as total,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as neutral,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
            FROM reviews
            WHERE is_valid = 1 AND created_at >= ?
        """
        params: list = [cutoff]
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        query += " GROUP BY DATE(created_at) ORDER BY date"

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_historical_baseline(
        self,
        product_id: str = "",
        window_days: int = 90,
        exclude_recent_days: int = 7,
    ) -> dict:
        """Get historical metric baselines for anomaly detection."""
        cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
        recent = (datetime.now() - timedelta(days=exclude_recent_days)).isoformat()

        query = """
            SELECT
                AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END) as avg_neg_ratio,
                AVG(rating) as avg_rating,
                COUNT(*) * 1.0 / ? as avg_daily_volume
            FROM reviews
            WHERE is_valid = 1 AND created_at >= ? AND created_at < ?
        """
        params: list = [window_days, cutoff, recent]
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)

        row = self._conn.execute(query, params).fetchone()
        if row and row["avg_neg_ratio"] is not None:
            return {
                "avg_neg_ratio": round(row["avg_neg_ratio"], 4),
                "avg_rating": round(row["avg_rating"], 2),
                "avg_daily_volume": round(row["avg_daily_volume"], 2),
            }
        return {"avg_neg_ratio": 0.15, "avg_rating": 4.0, "avg_daily_volume": 10.0}

    def get_unprocessed_reviews(self, product_id: str = "") -> list[ReviewRecord]:
        """Get reviews that haven't been cleaned yet (still have raw content)."""
        # Reviews with empty review_type or very short content are considered unprocessed
        query = """
            SELECT * FROM reviews
            WHERE (review_type = '' OR review_type IS NULL)
        """
        params: list = []
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_review(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


def _row_to_review(row: sqlite3.Row) -> ReviewRecord:
    """Convert a SQLite row to a ReviewRecord."""
    return ReviewRecord(
        id=row["id"],
        platform=row["platform"],
        product_id=row["product_id"],
        content=row["content"],
        rating=row["rating"],
        review_type=row["review_type"],
        created_at=datetime.fromisoformat(row["created_at"]),
        imported_at=datetime.fromisoformat(row["imported_at"]),
        is_valid=bool(row["is_valid"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
