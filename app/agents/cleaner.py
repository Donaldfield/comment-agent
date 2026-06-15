"""Cleaning Agent — rule-based + LLM review filtering."""

import logging

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


async def cleaner_node(state: AgentState) -> dict:
    """Filter ad/spam/meaningless/duplicate reviews.

    Reads state["raw_reviews"].
    Outputs state["cleaned_reviews"] as list of dicts.
    """
    try:
        from src.ingestion.cleaner import clean_rule_based, clean_with_llm
        from src.models.review import ReviewRecord

        raw = state.get("raw_reviews", [])
        if not raw:
            return {
                "status": "error",
                "current_step": "cleaner",
                "errors": [{"step": "cleaner", "message": "No raw reviews to clean"}],
            }

        # Convert dicts back to ReviewRecord objects
        records = [ReviewRecord(**r) for r in raw]

        # Step 1: Rule-based cleaning (always runs, free)
        records = clean_rule_based(records)
        valid_rule = sum(1 for r in records if r.is_valid)
        logger.info("Cleaner: after rule-based: %d/%d valid", valid_rule, len(records))

        # Step 2: LLM deep cleaning (attempt, but fall back gracefully)
        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            ambiguous = [r for r in records if r.is_valid]
            if ambiguous:
                clean_with_llm(ambiguous, llm)
        except Exception as e:
            logger.warning("Cleaner: LLM cleaning skipped (%s), using rule-based only", e)

        # Persist to DB
        from src.memory.repository import ReviewRepository
        repo = ReviewRepository("data/reviews.db")
        for record in records:
            repo.update_review_validity(record.id, record.is_valid, record.content)
        repo.close()

        cleaned = [r.model_dump() for r in records]
        valid_count = sum(1 for r in records if r.is_valid)

        logger.info("Cleaner: %d/%d valid after full cleaning", valid_count, len(records))

        return {
            "cleaned_reviews": cleaned,
            "current_step": "analyzer",
            "status": "running",
        }

    except Exception as e:
        logger.error("Cleaner failed: %s", e)
        return {
            "status": "error",
            "current_step": "cleaner",
            "errors": [{"step": "cleaner", "message": str(e)}],
        }
