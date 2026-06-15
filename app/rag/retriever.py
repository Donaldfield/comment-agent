"""RAG retriever — search Milvus and assemble context for LLM.

Query flow:
  1. Embed the query text
  2. Search all 3 Milvus collections in parallel
  3. Assemble context from results
  4. Return structured context for LLM consumption
"""

import logging
from typing import Optional

from app.rag.milvus_store import MilvusStore
from app.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


def query_rag(
    query_text: str,
    product_id: str = "",
    top_k: int = 10,
) -> dict:
    """Search the RAG knowledge base for issues similar to the query.

    Args:
        query_text: Natural language query about product issues.
        product_id: Optional product filter.
        top_k: Number of results per collection.

    Returns:
        Dict with similar_reviews, similar_pain_points, historical_issues.
    """
    embed_fn = get_embedding_function()
    if embed_fn is None:
        logger.warning("RAG query skipped: no embedding function available")
        return {
            "similar_reviews": [],
            "similar_pain_points": [],
            "historical_issues": [],
        }

    store = MilvusStore()
    if not store.is_available:
        logger.warning("RAG query skipped: Milvus not available")
        return {
            "similar_reviews": [],
            "similar_pain_points": [],
            "historical_issues": [],
        }

    # Embed the query
    query_embedding = embed_fn([query_text])[0]

    # Search all collections
    reviews = store.search_reviews(query_embedding, top_k=top_k)
    pain_points = store.search_pain_points(query_embedding, top_k=min(top_k, 5))
    issues = store.search_issues(query_embedding, top_k=min(top_k, 5))

    logger.info("RAG query: %d reviews, %d pain points, %d issues found",
                len(reviews), len(pain_points), len(issues))

    return {
        "similar_reviews": reviews,
        "similar_pain_points": pain_points,
        "historical_issues": issues,
    }


def generate_improvement_suggestions(
    product_id: str,
    pain_points: list[dict],
    top_k: int = 5,
) -> list[str]:
    """Generate product improvement suggestions using RAG context.

    For each pain point, retrieves similar historical issues
    and uses them to generate actionable suggestions.
    """
    embed_fn = get_embedding_function()
    if embed_fn is None:
        return []

    store = MilvusStore()
    if not store.is_available:
        return []

    suggestions = []

    for pp in pain_points[:top_k]:
        desc = pp.get("description", "")
        if not desc:
            continue

        query_embedding = embed_fn([desc])[0]
        issues = store.search_issues(query_embedding, top_k=3)

        if issues:
            # Format suggestion from historical evidence
            resolutions = [
                i.get("metadata", {}).get("resolution", "")
                for i in issues
                if i.get("metadata", {}).get("resolution")
            ]
            if resolutions:
                suggestion = (
                    f"[{pp.get('category', 'other')}] {desc}: "
                    f"Historical similar issues were resolved by: {'; '.join(resolutions[:2])}"
                )
                suggestions.append(suggestion)

    return suggestions


def build_rag_context(query_text: str, product_id: str = "") -> str:
    """Build a formatted context string for LLM consumption.

    Args:
        query_text: User query or pain point description.
        product_id: Optional product filter.

    Returns:
        Formatted context string ready for LLM prompt injection.
    """
    results = query_rag(query_text, product_id=product_id)

    parts = []

    if results["historical_issues"]:
        parts.append("## Historical Similar Issues (with resolutions)")
        for issue in results["historical_issues"]:
            metadata = issue.get("metadata", {})
            parts.append(f"- {issue['text']}")
            if metadata.get("resolution"):
                parts.append(f"  Resolution: {metadata['resolution']}")

    if results["similar_pain_points"]:
        parts.append("\n## Similar Past Pain Points")
        for pp in results["similar_pain_points"]:
            parts.append(f"- {pp['text']} (distance: {pp['distance']:.3f})")

    if results["similar_reviews"]:
        parts.append("\n## Similar Reviews")
        for review in results["similar_reviews"][:5]:
            parts.append(f"- {review['text'][:200]}")

    return "\n".join(parts) if parts else ""
