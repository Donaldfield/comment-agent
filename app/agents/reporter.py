"""Report Agent — generate Excel report + JSON output."""

import logging
from datetime import datetime
from pathlib import Path

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


async def reporter_node(state: AgentState) -> dict:
    """Generate Excel and JSON reports from analysis results.

    Reads all analysis data from state.
    Outputs report_path.
    """
    try:
        from src.models.analysis import AnalysisResultBundle
        from app.config import get_settings

        settings = get_settings()
        product_id = state.get("product_id", "all")

        # Assemble bundle
        sentiment = state.get("sentiment_results", [])
        keywords = state.get("keyword_results", [])
        pain_points = state.get("pain_points", [])
        anomaly_events = state.get("anomaly_events", [])
        alerts = state.get("alerts", [])
        cleaned = state.get("cleaned_reviews", [])

        pos_count = sum(1 for s in sentiment if s.get("sentiment") == "positive")
        neu_count = sum(1 for s in sentiment if s.get("sentiment") == "neutral")
        neg_count = sum(1 for s in sentiment if s.get("sentiment") == "negative")

        valid_reviews = [r for r in cleaned if r.get("is_valid", True)]

        # Daily stats from repository
        from src.memory.repository import ReviewRepository
        repo = ReviewRepository(settings.db_path)
        daily_stats = repo.get_daily_stats(product_id=product_id, days=30)
        repo.close()

        bundle = AnalysisResultBundle(
            product_id=product_id,
            analysis_time=datetime.now(),
            total_reviews=len(valid_reviews),
            valid_reviews=len(valid_reviews),
            positive_count=pos_count,
            neutral_count=neu_count,
            negative_count=neg_count,
            daily_stats=daily_stats,
            improvement_suggestions=state.get("improvement_suggestions", []),
        )

        # Generate Excel report
        report_dir = Path(settings.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = str(report_dir / f"{product_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        from src.output.excel_report import generate_report
        generate_report(bundle, report_path)

        # Generate JSON report
        json_path = report_path.replace(".xlsx", ".json")
        from src.output.json_formatter import to_file
        to_file(bundle, json_path)

        logger.info("Reporter: reports saved to %s", report_path)

        return {
            "report_path": report_path,
            "current_step": "reporter",
            "status": "done",
        }

    except Exception as e:
        logger.error("Reporter failed: %s", e)
        return {
            "status": "error",
            "current_step": "reporter",
            "errors": [{"step": "reporter", "message": str(e)}],
        }
