"""Collection Agent — parse CSV/Excel, normalize, store to DB."""

import logging
from pathlib import Path

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


async def collector_node(state: AgentState) -> dict:
    """Parse and import review data from a file.

    Reads state["source_file"] and state["platform"].
    Skips if raw_reviews already populated (data already in DB).
    Outputs state["raw_reviews"] as list of dicts.
    """
    try:
        # Skip if reviews already loaded (e.g., from DB)
        if state.get("raw_reviews"):
            logger.info("Collector: %d reviews already loaded, skipping", len(state["raw_reviews"]))
            return {"current_step": "cleaner", "status": "running"}

        from src.ingestion.parser import parse_file
        from src.ingestion.normalizer import normalize
        from src.memory.repository import ReviewRepository

        source_file = state.get("source_file", "")
        platform = state.get("platform", "taobao")
        product_id = state.get("product_id", "")

        if not source_file or not Path(source_file).exists():
            return {
                "status": "error",
                "current_step": "collector",
                "errors": [{"step": "collector", "message": f"File not found: {source_file}"}],
            }

        # Parse
        records = parse_file(source_file, platform=platform, product_id=product_id)

        # Normalize
        records = normalize(records)

        if not records:
            return {
                "status": "error",
                "current_step": "collector",
                "errors": [{"step": "collector", "message": "No valid records parsed from file"}],
            }

        # Store to DB
        repo = ReviewRepository("data/reviews.db")
        repo.insert_reviews(records)
        repo.close()

        # Convert to dicts for state
        raw_reviews = [r.model_dump() for r in records]

        logger.info("Collector: imported %d reviews", len(raw_reviews))

        return {
            "raw_reviews": raw_reviews,
            "current_step": "cleaner",
            "status": "running",
        }

    except Exception as e:
        logger.error("Collector failed: %s", e)
        return {
            "status": "error",
            "current_step": "collector",
            "errors": [{"step": "collector", "message": str(e)}],
        }
