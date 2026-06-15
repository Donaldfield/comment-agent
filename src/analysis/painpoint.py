"""Pain point mining via two-stage analysis: FAISS clustering + LLM labeling.

Stage 1 (FAISS): Groups similar negative reviews into clusters (free, fast).
Stage 2 (LLM): Labels each cluster with pain point category, description, severity
               (one LLM call per cluster, not per review).

This two-stage design keeps LLM costs proportional to cluster count (O(20)),
not review count (O(10,000)).
"""

import json
import logging
from typing import Optional

import numpy as np

from src.models.review import ReviewRecord
from src.models.analysis import PainPoint, SentimentResult
# LLM client provided by caller
from src.memory.vector_store import FAISSVectorStore
from src.memory.embedding import get_embedding_generator

logger = logging.getLogger(__name__)


def extract_pain_points(
    reviews: list[ReviewRecord],
    negative_reviews: list[ReviewRecord],
    llm,
    vector_store: FAISSVectorStore,
    sentiment_results: list[SentimentResult],
    ecommerce_rules: dict,
    n_clusters: Optional[int] = None,
) -> list[PainPoint]:
    """Extract pain points from negative reviews.

    Two-stage pipeline:
    1. Build FAISS index from all reviews, cluster negative ones
    2. Send each cluster's representative reviews to LLM for labeling

    Args:
        reviews: All reviews (for context).
        negative_reviews: Only negative reviews (rating <= 2 or negative sentiment).
        llm: LLM provider.
        vector_store: FAISS index containing all review embeddings.
        sentiment_results: Sentiment classification results.
        ecommerce_rules: Pain point category keywords from config.
        n_clusters: Number of clusters (auto-computed if None).

    Returns:
        List of PainPoint sorted by frequency descending.
    """
    if not negative_reviews:
        logger.info("No negative reviews to analyze for pain points")
        return []

    # Auto-compute cluster count
    if n_clusters is None:
        n_clusters = max(3, min(20, len(negative_reviews) // 5))
        n_clusters = min(n_clusters, len(negative_reviews))

    logger.info("Clustering %d negative reviews into %d clusters", len(negative_reviews), n_clusters)

    # Build a temporary FAISS index just for negative reviews for clustering
    temp_store = _build_negative_index(negative_reviews, vector_store)

    if temp_store.size < 2:
        # Too few reviews for clustering — use LLM directly on all
        return _direct_llm_extraction(negative_reviews, llm, ecommerce_rules)

    # Stage 1: FAISS clustering
    clusters = temp_store.cluster(n_clusters)

    # Stage 2: LLM labeling per cluster
    pain_points: list[PainPoint] = []
    from src.llm.prompts import PromptRegistry
    prompt_registry = PromptRegistry()

    for cluster_id, review_ids in clusters.items():
        if len(review_ids) < 2:
            continue  # Skip singleton clusters

        # Get cluster reviews
        cluster_reviews = [r for r in negative_reviews if r.id in review_ids]
        if not cluster_reviews:
            continue

        # Find representative reviews (closest to centroid)
        representative_texts = _get_representative_reviews(cluster_reviews, temp_store, cluster_id)

        # Prepare LLM input
        cluster_data = {
            "cluster_id": cluster_id,
            "size": len(cluster_reviews),
            "representative_reviews": representative_texts,
            "all_review_texts": [r.content[:200] for r in cluster_reviews[:10]],
        }

        system_prompt, user_prompt = prompt_registry.render(
            "painpoint_extract.jinja2",
            product_context="Product reviews from e-commerce platform",
            ecommerce_rules=json.dumps(ecommerce_rules, ensure_ascii=False),
            clusters_json=json.dumps([cluster_data], ensure_ascii=False),
        )

        try:
            response = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_format="json_object",
            )
            result = json.loads(response)
            pain_point_items = result.get("pain_points", [])

            for pp in pain_point_items:
                pain_points.append(PainPoint(
                    category=pp.get("category", "other"),
                    description=pp.get("description", ""),
                    frequency=len(cluster_reviews),
                    severity=pp.get("severity", "medium"),
                    is_high_frequency=len(cluster_reviews) >= 15,
                    example_review_ids=review_ids[:5],
                    representative_keywords=[],  # Filled from keyword extraction separately
                ))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM pain point extraction failed for cluster %d: %s", cluster_id, e)
            # Use rule-based fallback for this cluster
            pain_points.append(_rule_based_pain_point(cluster_reviews, ecommerce_rules))

    # Sort by frequency descending
    pain_points.sort(key=lambda p: p.frequency, reverse=True)

    # Mark high-frequency
    for pp in pain_points:
        pp.is_high_frequency = pp.frequency >= 15

    logger.info("Extracted %d pain points", len(pain_points))
    return pain_points


def _build_negative_index(
    negative_reviews: list[ReviewRecord],
    main_store: FAISSVectorStore,
) -> FAISSVectorStore:
    """Build a temporary FAISS index for negative reviews only.

    Reuses embeddings from the main store if available, otherwise generates new ones.
    """
    store = FAISSVectorStore(dimension=384)

    if main_store.size > 0:
        # Try to extract negative review vectors from main store
        main_ids = set(main_store.get_all_ids())
        neg_ids = [r.id for r in negative_reviews if r.id in main_ids]

        if neg_ids:
            # We need to rebuild with just these — for now, generate fresh embeddings
            pass

    # Generate fresh embeddings for negative reviews
    embed_fn = get_embedding_generator()
    texts = [r.content for r in negative_reviews]
    ids = [r.id for r in negative_reviews]

    if texts:
        vectors = embed_fn(texts)
        store.add(ids, vectors)

    return store


def _get_representative_reviews(
    cluster_reviews: list[ReviewRecord],
    store: FAISSVectorStore,
    cluster_id: int,
    top_k: int = 5,
) -> list[str]:
    """Get the most representative review texts for a cluster."""
    # Use reviews closest to centroid as representatives
    ids_in_store = [r.id for r in cluster_reviews if r.id in store.get_all_ids()]
    if ids_in_store:
        results = store.get_centroid_reviews(ids_in_store[:1], top_k=top_k)
        for similar_list in results.values():
            texts = []
            for review_id, _ in similar_list:
                match = next((r for r in cluster_reviews if r.id == review_id), None)
                if match:
                    texts.append(match.content)
            if texts:
                return texts

    # Fallback: just take the first few reviews
    return [r.content for r in cluster_reviews[:top_k]]


def _direct_llm_extraction(
    reviews: list[ReviewRecord],
    llm,
    ecommerce_rules: dict,
) -> list[PainPoint]:
    """LLM extraction on all reviews directly when clustering isn't possible."""
    from src.llm.prompts import PromptRegistry
    prompt_registry = PromptRegistry()

    cluster_data = {
        "cluster_id": 0,
        "size": len(reviews),
        "representative_reviews": [r.content for r in reviews[:10]],
        "all_review_texts": [r.content[:200] for r in reviews[:20]],
    }

    system_prompt, user_prompt = prompt_registry.render(
        "painpoint_extract.jinja2",
        product_context="Product reviews from e-commerce platform",
        ecommerce_rules=json.dumps(ecommerce_rules, ensure_ascii=False),
        clusters_json=json.dumps([cluster_data], ensure_ascii=False),
    )

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format="json_object",
        )
        result = json.loads(response)
        return [
            PainPoint(
                category=pp.get("category", "other"),
                description=pp.get("description", ""),
                frequency=len(reviews),
                severity=pp.get("severity", "medium"),
                is_high_frequency=len(reviews) >= 15,
                example_review_ids=[r.id for r in reviews[:5]],
            )
            for pp in result.get("pain_points", [])
        ]
    except Exception as e:
        logger.warning("Direct LLM extraction failed: %s", e)
        return [_rule_based_pain_point(reviews, ecommerce_rules)]


def _rule_based_pain_point(
    reviews: list[ReviewRecord],
    ecommerce_rules: dict,
) -> PainPoint:
    """Fallback: rule-based pain point detection using keyword matching."""
    categories = ecommerce_rules.get("painpoint_categories", {})
    cat_matches: dict[str, int] = {}

    all_text = " ".join(r.content for r in reviews)
    for category, keywords in categories.items():
        count = sum(1 for kw in keywords if kw in all_text)
        if count > 0:
            cat_matches[category] = count

    if cat_matches:
        best_category = max(cat_matches, key=cat_matches.get)
    else:
        best_category = "other"

    return PainPoint(
        category=best_category,
        description=f"Detected {len(reviews)} reviews related to {best_category}",
        frequency=len(reviews),
        severity="medium",
        is_high_frequency=len(reviews) >= 15,
        example_review_ids=[r.id for r in reviews[:5]],
    )
