"""Keyword extraction using TF-IDF with jieba Chinese tokenization.

Pure Python, no LLM cost. Provides the default keyword extraction path.
LLM-based extraction is available through sentiment analysis which already
returns per-review keywords.
"""

import logging
from collections import Counter
from typing import Optional

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.review import ReviewRecord
from src.models.analysis import KeywordResult, SentimentResult

logger = logging.getLogger(__name__)

# Chinese stop words — common words that carry no topic signal
STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "虽然", "可以", "还是",
    "这个", "那个", "什么", "怎么", "哪里", "哪", "吗", "吧", "呢", "啊",
    "哦", "嗯", "呀", "哈", "嗯嗯", "哈哈", "呵呵",
    # E-commerce specific stop words
    "好评", "差评", "中评", "评价", "用户", "购买", "东西", "收到",
    "还行", "不错", "可以", "一般", "挺", "比较", "非常", "真的",
}


def extract_keywords_tfidf(
    reviews: list[ReviewRecord],
    top_n: int = 30,
    sentiment_results: Optional[list[SentimentResult]] = None,
) -> list[KeywordResult]:
    """Extract top keywords using TF-IDF with jieba tokenization.

    Args:
        reviews: Reviews to analyze.
        top_n: Number of top keywords to return.
        sentiment_results: Optional sentiment data to associate keywords with sentiment.

    Returns:
        List of KeywordResult sorted by frequency descending.
    """
    if not reviews:
        return []

    # Tokenize all reviews with jieba
    texts = []
    for review in reviews:
        tokens = _tokenize(review.content)
        texts.append(" ".join(tokens))

    if not texts:
        return []

    # TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words=list(STOP_WORDS),
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all documents for each term
        scores = tfidf_matrix.sum(axis=0).A1
        term_scores = list(zip(feature_names, scores))
        term_scores.sort(key=lambda x: x[1], reverse=True)

    except ValueError:
        # Fallback: pure jieba frequency
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        counter = Counter(all_tokens)
        term_scores = [
            (word, count)
            for word, count in counter.most_common(top_n * 2)
            if word not in STOP_WORDS
        ]

    # Count raw frequency for each keyword
    all_tokens_flat: list[str] = []
    for text in texts:
        all_tokens_flat.extend(text.split())
    token_counter = Counter(all_tokens_flat)

    # Associate keywords with dominant sentiment
    sentiment_map: dict[str, str] = {}
    if sentiment_results:
        sentiment_map = {s.review_id: s.sentiment for s in sentiment_results}

    results: list[KeywordResult] = []
    for term, score in term_scores[:top_n]:
        if len(term) < 2:  # Skip single characters
            continue
        if term in STOP_WORDS:
            continue

        freq = token_counter.get(term, 0)

        # Find dominant sentiment for reviews containing this keyword
        assoc_sentiment = ""
        sentiment_counter: Counter = Counter()
        for review in reviews:
            if term in review.content:
                sentiment = sentiment_map.get(review.id, "")
                if sentiment:
                    sentiment_counter[sentiment] += 1
        if sentiment_counter:
            assoc_sentiment = sentiment_counter.most_common(1)[0][0]

        results.append(KeywordResult(
            keyword=term,
            frequency=freq,
            associated_sentiment=assoc_sentiment,
        ))

    logger.info("Extracted %d keywords via TF-IDF", len(results))
    return results


def extract_keywords_by_sentiment(
    reviews: list[ReviewRecord],
    sentiment_results: list[SentimentResult],
    top_n: int = 10,
) -> dict[str, list[KeywordResult]]:
    """Extract keywords separately for positive, neutral, negative reviews.

    Returns:
        Dict mapping sentiment -> list of KeywordResult.
    """
    by_sentiment: dict[str, list[ReviewRecord]] = {
        "positive": [],
        "neutral": [],
        "negative": [],
    }

    sentiment_lookup = {s.review_id: s.sentiment for s in sentiment_results}
    for review in reviews:
        sentiment = sentiment_lookup.get(review.id, "neutral")
        if sentiment in by_sentiment:
            by_sentiment[sentiment].append(review)

    return {
        sentiment: extract_keywords_tfidf(revs, top_n=top_n)
        for sentiment, revs in by_sentiment.items()
        if revs
    }


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese text using jieba, filtering stop words."""
    tokens = jieba.cut(text.strip())
    return [
        t.strip()
        for t in tokens
        if len(t.strip()) >= 2 and t.strip() not in STOP_WORDS
    ]
