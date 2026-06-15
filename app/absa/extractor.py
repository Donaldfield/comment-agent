"""ABSA extractor — per-aspect sentiment via LLM.

Sends reviews in batches of 15 (smaller than general sentiment batch
because ABSA output is much larger — 7 aspects per review).
"""

import json
import logging

from typing import Optional

from src.models.review import ReviewRecord
from src.llm.prompts import PromptRegistry

logger = logging.getLogger(__name__)

BATCH_SIZE = 15  # Smaller than general sentiment (20) due to larger output

ASPECTS = ["quality", "logistics", "packaging", "size", "color", "service", "value"]


def extract_absa(
    reviews: list[ReviewRecord],
    llm,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """Extract aspect-based sentiment for a list of reviews.

    Args:
        reviews: Cleaned review records.
        llm: LLM client with generate() method.
        batch_size: Reviews per LLM call.

    Returns:
        List of ABSA result dicts, one per review.
    """
    if not reviews:
        return []

    results: list[dict] = []
    prompt_registry = PromptRegistry()

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i : i + batch_size]
        reviews_json = json.dumps(
            [{"id": r.id, "content": r.content} for r in batch],
            ensure_ascii=False,
        )

        system_prompt, user_prompt = prompt_registry.render(
            "absa_extract.jinja2", reviews_json=reviews_json
        )

        try:
            response = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_format="json_object",
            )
            parsed = json.loads(response)
            batch_results = parsed.get("results", [])

            for item in batch_results:
                results.append({
                    "review_id": item.get("review_id", ""),
                    "overall_sentiment": item.get("overall_sentiment", "neutral"),
                    "aspects": item.get("aspects", {}),
                })

        except Exception as e:
            logger.warning("ABSA batch %d failed: %s. Using empty results.", i, e)
            for review in batch:
                results.append({
                    "review_id": review.id,
                    "overall_sentiment": "neutral",
                    "aspects": {
                        aspect: {"sentiment": "not_mentioned", "confidence": 0, "evidence": ""}
                        for aspect in ASPECTS
                    },
                })

    logger.info("ABSA extracted for %d reviews", len(results))
    return results
