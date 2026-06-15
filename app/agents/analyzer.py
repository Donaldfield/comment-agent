"""Analysis Agent — sentiment, keywords, pain points, anomaly detection."""

import logging

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


async def analyzer_node(state: AgentState) -> dict:
    """Run sentiment analysis + keyword extraction + pain point mining.

    Reads state["cleaned_reviews"].
    Outputs sentiment_results, keyword_results, pain_points, anomaly_events.
    """
    try:
        from src.models.review import ReviewRecord
        from src.models.analysis import SentimentResult

        cleaned = state.get("cleaned_reviews", [])
        if not cleaned:
            return {
                "status": "error",
                "current_step": "analyzer",
                "errors": [{"step": "analyzer", "message": "No cleaned reviews to analyze"}],
            }

        reviews = [ReviewRecord(**r) for r in cleaned if r.get("is_valid", True)]
        if not reviews:
            return {"current_step": "alerter", "status": "running",
                    "sentiment_results": [], "keyword_results": [], "pain_points": []}

        # ── Sentiment analysis ──
        sentiment_results = []
        try:
            from src.analysis.sentiment import analyze_batch
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            sentiment_results = analyze_batch(reviews, llm)
        except Exception as e:
            logger.warning("LLM sentiment failed (%s), using rule-based fallback", e)
            for r in reviews:
                sentiment = "positive" if r.rating >= 4 else ("negative" if r.rating <= 2 else "neutral")
                sentiment_results.append(SentimentResult(
                    review_id=r.id, sentiment=sentiment, confidence=0.8,
                ))

        # ── Keyword extraction ──
        from src.analysis.keywords import extract_keywords_tfidf
        keywords = extract_keywords_tfidf(reviews, top_n=30, sentiment_results=sentiment_results)

        # ── Pain point mining ──
        from app.config import get_settings
        ecommerce_rules = {"painpoint_categories": get_settings().painpoint_categories}

        negative_reviews = [
            r for r in reviews
            if r.rating <= 2 or any(
                s.review_id == r.id and s.sentiment == "negative"
                for s in sentiment_results
            )
        ]

        pain_points = _extract_pain_points_keyword(negative_reviews, ecommerce_rules)

        # ── Anomaly detection ──
        from src.analysis.anomaly import detect_sentiment_shift, detect_volume_spike
        anomaly_events = detect_sentiment_shift(sentiment_results, reviews)
        anomaly_events += detect_volume_spike(reviews)

        # ── Build output ──
        pos = sum(1 for s in sentiment_results if s.sentiment == "positive")
        neu = sum(1 for s in sentiment_results if s.sentiment == "neutral")
        neg = sum(1 for s in sentiment_results if s.sentiment == "negative")

        logger.info("Analyzer: %d pos, %d neu, %d neg, %d keywords, %d pain points, %d anomalies",
                    pos, neu, neg, len(keywords), len(pain_points), len(anomaly_events))

        return {
            "sentiment_results": [s.model_dump() for s in sentiment_results],
            "keyword_results": [k.model_dump() for k in keywords],
            "pain_points": [p.model_dump() for p in pain_points],
            "anomaly_events": [a.model_dump() for a in anomaly_events],
            "current_step": "alerter",
            "status": "running",
        }

    except Exception as e:
        logger.error("Analyzer failed: %s", e)
        return {
            "status": "error",
            "current_step": "analyzer",
            "errors": [{"step": "analyzer", "message": str(e)}],
        }


def _extract_pain_points_keyword(reviews, ecommerce_rules: dict) -> list:
    """Keyword-based pain point extraction (no FAISS/embedding needed)."""
    from src.models.analysis import PainPoint

    categories = ecommerce_rules.get("painpoint_categories", {})
    if not categories or not reviews:
        return []

    results = []
    for category, keywords in categories.items():
        matching = [r for r in reviews if any(kw in r.content for kw in keywords)]
        if matching:
            results.append(PainPoint(
                category=category,
                description=f"{category} related issues",
                frequency=len(matching),
                severity="high" if len(matching) >= 5 else "medium",
                is_high_frequency=len(matching) >= 15,
                example_review_ids=[r.id for r in matching[:5]],
                representative_keywords=keywords[:5],
            ))

    results.sort(key=lambda p: p.frequency, reverse=True)
    logger.info("Keyword-based pain points: %d categories found", len(results))
    return results
