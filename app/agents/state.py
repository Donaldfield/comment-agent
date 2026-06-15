"""LangGraph AgentState — shared state across all 5 agent nodes."""

from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    # ── Identity ──
    task_id: str
    product_id: str
    platform: str

    # ── Pipeline control ──
    status: str           # pending|running|done|error
    current_step: str     # collector|cleaner|analyzer|alerter|reporter
    retry_count: int
    max_retries: int      # default 2

    # ── Errors (append-only via reducer) ──
    errors: Annotated[list[dict], operator.add]

    # ── Data payloads ──
    source_file: str
    raw_reviews: list[dict]
    cleaned_reviews: list[dict]
    sentiment_results: list[dict]
    absa_results: list[dict]
    keyword_results: list[dict]
    pain_points: list[dict]
    anomaly_events: list[dict]
    alerts: list[dict]
    report_path: str
    historical_issues: list[dict]
    improvement_suggestions: list[str]
