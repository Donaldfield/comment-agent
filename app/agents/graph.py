"""LangGraph StateGraph — 5-agent pipeline with basic retry.

Graph flow:
  START → collector → cleaner → analyzer → alerter → reporter → END
              │           │          │          │          │
              └───────────┴──────────┴──────────┴──────────┘
                                  │ (error)
                           error_handler
                                  │
                          retry<2 → 回原节点
                          retry≥2 → END(error)
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.collector import collector_node
from app.agents.cleaner import cleaner_node
from app.agents.analyzer import analyzer_node
from app.agents.alerter import alerter_node
from app.agents.reporter import reporter_node

logger = logging.getLogger(__name__)


# ── Routing functions ──

def route_result(state: AgentState) -> Literal["success", "error"]:
    """Route based on node execution result."""
    if state.get("status") == "error":
        return "error"
    return "success"


def route_retry(state: AgentState) -> str:
    """Decide whether to retry or give up."""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    failed_step = state.get("current_step", "")

    if retry_count < max_retries:
        logger.info("Retry %d/%d for step '%s'", retry_count + 1, max_retries, failed_step)
        return failed_step

    logger.error("Max retries (%d) exceeded for step '%s'", max_retries, failed_step)
    return "end"


# ── Error handler node ──

async def error_handler_node(state: AgentState) -> dict:
    """Increment retry count and prepare for retry."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ── Graph construction ──

def build_graph() -> StateGraph:
    """Build and compile the 5-agent LangGraph pipeline."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("collector", collector_node)
    workflow.add_node("cleaner", cleaner_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("alerter", alerter_node)
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("error_handler", error_handler_node)

    # Entry point
    workflow.set_entry_point("collector")

    # Success/error routing for each agent node
    workflow.add_conditional_edges("collector", route_result, {
        "success": "cleaner",
        "error": "error_handler",
    })
    workflow.add_conditional_edges("cleaner", route_result, {
        "success": "analyzer",
        "error": "error_handler",
    })
    workflow.add_conditional_edges("analyzer", route_result, {
        "success": "alerter",
        "error": "error_handler",
    })
    workflow.add_conditional_edges("alerter", route_result, {
        "success": "reporter",
        "error": "error_handler",
    })
    workflow.add_conditional_edges("reporter", route_result, {
        "success": END,
        "error": "error_handler",
    })

    # Error handler → retry or give up
    workflow.add_conditional_edges("error_handler", route_retry, {
        "collector": "collector",
        "cleaner": "cleaner",
        "analyzer": "analyzer",
        "alerter": "alerter",
        "reporter": "reporter",
        "end": END,
    })

    return workflow.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    """Get or create the compiled LangGraph pipeline."""
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph pipeline compiled (5 agents + error handler)")
    return _graph
