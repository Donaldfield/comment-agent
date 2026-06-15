"""Alert Agent — evaluate alert rules and generate remediation plans."""

import logging

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


async def alerter_node(state: AgentState) -> dict:
    """Evaluate alert rules against analysis results.

    Reads sentiment_results, pain_points, anomaly_events.
    Outputs alerts list.
    """
    try:
        from src.models.alert import RuleContext
        from src.analysis.alert_rules import evaluate_rules
        from src.memory.repository import ReviewRepository
        from app.config import get_settings

        settings = get_settings()
        repo = ReviewRepository(settings.db_path)

        product_id = state.get("product_id", "")
        pain_points = state.get("pain_points", [])
        anomaly_events = state.get("anomaly_events", [])
        sentiment_results = state.get("sentiment_results", [])

        # Build rule context
        alert_config = {
            "rules": {
                "high_freq_negative": {
                    "enabled": True,
                    "threshold": settings.alert_threshold_high_freq,
                    "time_window_hours": settings.alert_time_window_hours,
                },
                "sentiment_shift": {
                    "enabled": True,
                    "stddev_multiplier": settings.alert_sentiment_stddev,
                },
                "new_pain_point": {"enabled": True},
                "volume_spike": {
                    "enabled": True,
                    "multiplier": settings.alert_volume_multiplier,
                },
            }
        }

        # Convert dicts back to model objects
        from src.models.analysis import PainPoint, AnomalyEvent, SentimentResult
        from src.models.review import ReviewRecord

        pp_objects = [PainPoint(**pp) for pp in pain_points]
        ae_objects = [AnomalyEvent(**ae) for ae in anomaly_events]
        sr_objects = [SentimentResult(**sr) for sr in sentiment_results]

        # Get reviews for context
        cleaned = state.get("cleaned_reviews", [])
        review_objects = [ReviewRecord(**r) for r in cleaned if r.get("is_valid", True)]

        historical_baseline = repo.get_historical_baseline(product_id=product_id)

        ctx = RuleContext(
            product_id=product_id,
            time_window_hours=settings.alert_time_window_hours,
            reviews=review_objects,
            sentiment_results=sr_objects,
            pain_points=pp_objects,
            historical_baseline=historical_baseline,
            anomaly_events=ae_objects,
            config=alert_config,
        )

        # Evaluate standard rules with LLM if available
        try:
            from app.llm.client import get_llm_client
            llm = get_llm_client()
            alerts = evaluate_rules(ctx, llm=llm)
        except Exception:
            alerts = evaluate_rules(ctx, llm=None)

        # ── Baseline alerts (always works, even on small data) ──
        alerts = list(alerts)  # ensure mutable
        from src.models.alert import Alert

        # Alert if any negative reviews exist
        neg_count = sum(1 for s in sr_objects if s.sentiment == "negative")
        if neg_count > 0:
            alerts.append(Alert(
                level="info",
                rule_name="basic_negative_detected",
                title=f"检测到 {neg_count} 条负面评价",
                detail=f"共 {len(sr_objects)} 条评价中有 {neg_count} 条负面 ({neg_count/max(len(sr_objects),1):.1%})，建议关注。",
            ))

        # Alert if pain points found
        if pain_points:
            top_pp = pain_points[0]
            alerts.append(Alert(
                level="warning",
                rule_name="pain_point_detected",
                title=f"发现痛点: {top_pp.get('category', 'unknown')}",
                detail=top_pp.get("description", ""),
            ))

        repo.close()

        logger.info("Alerter: %d alerts triggered", len(alerts))

        return {
            "alerts": [a.model_dump() for a in alerts],
            "current_step": "reporter",
            "status": "running",
        }

    except Exception as e:
        logger.error("Alerter failed: %s", e)
        return {
            "status": "error",
            "current_step": "alerter",
            "errors": [{"step": "alerter", "message": str(e)}],
        }
