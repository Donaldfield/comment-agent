"""Embedding generator with fallback chain.

Tries sentence-transformers first, then DeepSeek API embeddings,
then fails gracefully.
"""

import logging
from typing import Optional, Callable

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_embed_fn: Optional[Callable] = None
_load_attempted: bool = False


def get_embedding_function() -> Optional[Callable[[list[str]], list[list[float]]]]:
    """Get embedding function with automatic fallback.

    Returns:
        A callable that takes list[str] and returns list[list[float]],
        or None if no embedding method is available.
    """
    global _embed_fn, _load_attempted

    if _embed_fn is not None:
        return _embed_fn

    if _load_attempted:
        return None

    _load_attempted = True

    # Attempt 1: local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        model_name = settings.embedding_model
        hf_endpoint = settings.hf_endpoint

        kwargs = {}
        if hf_endpoint:
            kwargs["model_kwargs"] = {"endpoint": hf_endpoint}

        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name, **kwargs)

        def _local_embed(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()

        _embed_fn = _local_embed
        logger.info("Using local sentence-transformers for embeddings")
        return _embed_fn

    except Exception as e:
        logger.warning("Local embedding model unavailable: %s", e)

    # Attempt 2: DeepSeek API embeddings (if API key is set)
    try:
        settings = get_settings()
        if settings.deepseek_api_key:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )

            def _api_embed(texts: list[str]) -> list[list[float]]:
                if not texts:
                    return []
                response = client.embeddings.create(
                    model="text-embedding-ada-002",  # DeepSeek doesn't have embeddings yet
                    input=texts,
                )
                return [d.embedding for d in response.data]

            _embed_fn = _api_embed
            logger.info("Using API-based embeddings")
            return _embed_fn
    except Exception as e:
        logger.warning("API embeddings unavailable: %s", e)

    logger.warning("No embedding method available. RAG disabled.")
    return None


def is_available() -> bool:
    """Check if any embedding method is available."""
    return get_embedding_function() is not None
