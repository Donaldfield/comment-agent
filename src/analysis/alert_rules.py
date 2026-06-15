"""Alert rule engine.

Evaluates a pipeline of rules against analysis results and triggers
alerts when thresholds are crossed. Each rule is a standalone function
with signature (RuleContext) -> Optional[Alert].

Four built-in rules:
1. HighFrequencyNegative — Pain point frequency crosses threshold
2. SentimentShift — Negative ratio spikes vs historical baseline
3. NewPainPoint — Previously unseen pain point category appears
4. VolumeSpike — Review volume suddenly exceeds baseline
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

from src.models.alert import Alert, RuleContext
from src.models.analysis import PainPoint, AnomalyEvent
# LLM client provided by caller

logger = logging.getLogger(__name__)

# Type alias for rule functions
RuleFunc = Callable[[RuleContext], Optional[Alert]]


def evaluate_rules(ctx: RuleContext, llm = None) -> list[Alert]:
    """Evaluate all enabled rules and return triggered alerts.

    Args:
        ctx: RuleContext with all data needed for evaluation.
        llm: Optional LLM for generating remediation plans.

    Returns:
        List of triggered Alert objects.
    """
    rules_enabled = ctx.config.get("rules", {})
    alerts: list[Alert] = []

    rule_definitions: list[tuple[str, RuleFunc]] = [
        ("high_freq_negative", rule_high_freq_negative),
        ("sentiment_shift", rule_sentiment_shift),
        ("new_pain_point", rule_new_pain_point),
        ("volume_spike", rule_volume_spike),
    ]

    for rule_name, rule_func in rule_definitions:
        rule_config = rules_enabled.get(rule_name, {})
        if not rule_config.get("enabled", True):
            continue

        try:
            alert = rule_func(ctx)
            if alert:
                # Generate remediation plan if LLM available
                if llm and not alert.remediation_plan:
                    alert.remediation_plan = _generate_remediation(alert, ctx, llm)
                alerts.append(alert)
                logger.info("Alert triggered: %s (level=%s)", alert.title, alert.level)
        except Exception as e:
            logger.error("Rule %s evaluation failed: %s", rule_name, e)

    logger.info("Alert evaluation complete: %d alerts triggered", len(alerts))
    return alerts


def rule_high_freq_negative(ctx: RuleContext) -> Optional[Alert]:
    """Alert when a pain point exceeds the high-frequency threshold."""
    rule_config = ctx.config.get("high_freq_negative", {})
    threshold = rule_config.get("threshold", 15)

    high_freq_points = [
        pp for pp in ctx.pain_points
        if pp.frequency >= threshold
    ]

    if not high_freq_points:
        return None

    # Alert on the most frequent pain point
    top = max(high_freq_points, key=lambda p: p.frequency)

    return Alert(
        level="warning",
        rule_name="high_freq_negative",
        title=f"High-frequency issue: {top.category}",
        detail=(
            f"Pain point '{top.description}' mentioned in {top.frequency} reviews "
            f"(threshold: {threshold}). Category: {top.category}, Severity: {top.severity}."
        ),
        affected_review_ids=top.example_review_ids,
        triggered_at=datetime.now(),
    )


def rule_sentiment_shift(ctx: RuleContext) -> Optional[Alert]:
    """Alert when negative sentiment ratio has spiked anomalously."""
    # Check anomaly events for sentiment_shift type
    sentiment_anomalies = [
        e for e in ctx.anomaly_events
        if e.event_type == "sentiment_shift"
    ]

    if not sentiment_anomalies:
        return None

    # Alert on the most severe anomaly
    top = max(sentiment_anomalies, key=lambda e: 0 if e.severity == "high" else 1)

    return Alert(
        level="critical",
        rule_name="sentiment_shift",
        title="Sudden negative sentiment surge detected",
        detail=top.description,
        affected_review_ids=top.affected_review_ids,
        triggered_at=datetime.now(),
    )


def rule_new_pain_point(ctx: RuleContext) -> Optional[Alert]:
    """Alert when a pain point category appears that had zero historical mentions."""
    # Get historical pain point categories
    historical_categories = set(
        ctx.historical_baseline.get("pain_point_categories", [])
    )

    # Find new categories
    new_points = [
        pp for pp in ctx.pain_points
        if pp.category not in historical_categories
        and pp.frequency >= 3  # Minimum threshold to avoid noise
    ]

    if not new_points:
        return None

    # Alert on the most frequent new pain point
    top = max(new_points, key=lambda p: p.frequency)

    return Alert(
        level="info",
        rule_name="new_pain_point",
        title=f"New issue detected: {top.category}",
        detail=(
            f"Previously unseen pain point category '{top.category}' appeared: "
            f"'{top.description}'. Mentioned in {top.frequency} reviews."
        ),
        affected_review_ids=top.example_review_ids,
        triggered_at=datetime.now(),
    )


def rule_volume_spike(ctx: RuleContext) -> Optional[Alert]:
    """Alert when review volume suddenly exceeds baseline."""
    volume_anomalies = [
        e for e in ctx.anomaly_events
        if e.event_type == "volume_spike"
    ]

    if not volume_anomalies:
        return None

    top = max(volume_anomalies, key=lambda e: 0 if e.severity == "high" else 1)

    return Alert(
        level="warning",
        rule_name="volume_spike",
        title="Review volume spike detected",
        detail=top.description,
        affected_review_ids=top.affected_review_ids,
        triggered_at=datetime.now(),
    )


def _generate_remediation(alert: Alert, ctx: RuleContext, llm) -> str:
    """Generate a remediation plan using LLM."""
    from src.llm.prompts import PromptRegistry

    prompt_registry = PromptRegistry()

    # Collect relevant review excerpts
    review_excerpts = []
    for rid in alert.affected_review_ids[:10]:
        for review in ctx.reviews:
            if review.id == rid:
                review_excerpts.append(review.content[:300])
                break

    # Find related pain point
    pain_point_desc = "Unknown"
    for pp in ctx.pain_points:
        if any(rid in pp.example_review_ids for rid in alert.affected_review_ids):
            pain_point_desc = pp.description
            break

    system_prompt, user_prompt = prompt_registry.render(
        "alert_remediation.jinja2",
        alert_title=alert.title,
        alert_level=alert.level,
        rule_name=alert.rule_name,
        pain_point=pain_point_desc,
        affected_count=len(alert.affected_review_ids),
        reviews_context=json.dumps(review_excerpts, ensure_ascii=False),
        product_context=f"Product {ctx.product_id}, {ctx.time_window_hours}h window",
    )

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format="json_object",
        )
        result = json.loads(response)

        # Format remediation as readable text
        parts = [f"**Diagnosis**: {result.get('diagnosis', 'N/A')}"]
        parts.append("\n**Immediate Actions**:")
        for action in result.get("immediate_actions", []):
            parts.append(
                f"  - [{action.get('timeline', 'N/A')}] {action.get('action', '')} "
                f"({action.get('expected_effect', '')})"
            )
        parts.append("\n**Long-term Actions**:")
        for action in result.get("long_term_actions", []):
            parts.append(
                f"  - [{action.get('timeline', 'N/A')}] {action.get('action', '')} "
                f"({action.get('expected_effect', '')})"
            )
        parts.append(f"\n**Estimated Impact**: {result.get('estimated_impact', 'N/A')}")

        return "\n".join(parts)

    except Exception as e:
        logger.warning("Remediation generation failed: %s", e)
        return f"Unable to generate remediation plan: {e}"
