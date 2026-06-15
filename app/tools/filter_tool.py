"""Function Calling tool — filter ad/spam/duplicate/malicious reviews via LLM.

Exposed as a LangChain @tool for use by the Cleaning Agent.
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def filter_review(review_text: str) -> dict:
    """Classify a review as valid, spam, meaningless, duplicate, or malicious.

    Args:
        review_text: The review content text to classify.

    Returns:
        Dict with keys: is_valid (bool), reason (str), cleaned_text (str).
    """
    import re

    # Quick rule-based checks first (free)
    content = review_text.strip()

    # Too short
    if len(content) < 4:
        return {"is_valid": False, "reason": "meaningless_too_short", "cleaned_text": ""}

    # Pure punctuation / emoji
    if re.match(r"^[\W_]+$", content):
        return {"is_valid": False, "reason": "meaningless_no_content", "cleaned_text": ""}

    # Ad detection
    ad_patterns = [
        r"http[s]?://", r"微信号|微信|wechat|wx[:：]",
        r"加我|加群|扫码|二维码", r"兼职|赚钱|日赚|代理",
        r"1[3-9]\d{9}", r"[\w.]+@[\w.]+",
    ]
    for pattern in ad_patterns:
        if re.search(pattern, content):
            return {"is_valid": False, "reason": "spam_advertisement", "cleaned_text": ""}

    # Looks valid
    return {"is_valid": True, "reason": "", "cleaned_text": content}
