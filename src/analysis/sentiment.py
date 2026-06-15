"""Batch sentiment classification using LLM.

Sends 20 reviews per LLM call to minimize API round-trips.
Falls back to rule-based sentiment if LLM call fails.
"""

import json
import logging
from typing import Optional

from src.models.review import ReviewRecord
from src.models.analysis import SentimentResult
# LLM client provided by caller (app/llm/client.py)
from src.llm.prompts import PromptRegistry

logger = logging.getLogger(__name__)

BATCH_SIZE = 20


def analyze_batch(
    reviews: list[ReviewRecord],
    llm,
    batch_size: int = BATCH_SIZE,
) -> list[SentimentResult]:
    """Classify sentiment for a list of reviews using LLM batching.

    Args:
        reviews: Reviews to analyze.
        llm: LLM provider instance.
        batch_size: Reviews per API call (default 20).

    Returns:
        List of SentimentResult, one per input review.
    """
    if not reviews:
        return []

    results: list[SentimentResult] = []
    prompt_registry = PromptRegistry()

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i : i + batch_size]
        reviews_json = json.dumps(
            [
                {"id": r.id, "content": r.content, "rating": r.rating}
                for r in batch
            ],
            ensure_ascii=False,
        )

        system_prompt, user_prompt = prompt_registry.render(
            "sentiment_batch.jinja2", reviews_json=reviews_json
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
                results.append(SentimentResult(
                    review_id=item["review_id"],
                    sentiment=item.get("sentiment", "neutral"),
                    confidence=float(item.get("confidence", 0.5)),
                    keywords=item.get("keywords", []),
                ))

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("LLM sentiment batch %d failed: %s. Using rule-based fallback.", i, e)
            # Fallback: rule-based sentiment from rating
            for review in batch:
                results.append(_rule_based_sentiment(review))

    logger.info(
        "Sentiment analyzed: %d reviews (%d positive, %d neutral, %d negative)",
        len(results),
        sum(1 for r in results if r.sentiment == "positive"),
        sum(1 for r in results if r.sentiment == "neutral"),
        sum(1 for r in results if r.sentiment == "negative"),
    )
    return results


def _rule_based_sentiment(review: ReviewRecord) -> SentimentResult:
    """Fallback sentiment from star rating."""
    if review.rating >= 4:
        sentiment = "positive"
        confidence = 0.8
    elif review.rating == 3:
        sentiment = "neutral"
        confidence = 0.7
    else:
        sentiment = "negative"
        confidence = 0.8

    return SentimentResult(
        review_id=review.id,
        sentiment=sentiment,
        confidence=confidence,
        keywords=[],
    )
