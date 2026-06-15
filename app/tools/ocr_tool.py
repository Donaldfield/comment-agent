"""Function Calling tool — OCR image review text extraction.

Stub implementation. DeepSeek does not natively support vision/image inputs.
When a vision-capable model is available, this tool would extract text
from review images (晒图评价).
"""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def extract_text_from_review_image(image_url: str) -> dict:
    """Extract text content from a review image using OCR.

    Args:
        image_url: URL or local path to the review image.

    Returns:
        Dict with 'text' (extracted text) and 'confidence' (0-1).
    """
    logger.info("OCR requested for: %s (STUB)", image_url)

    # DeepSeek does not support vision. In production with a vision-capable
    # model (GPT-4V, Claude, Qwen-VL), this would:
    # 1. Download the image
    # 2. Send to vision LLM with prompt: "Extract all text from this image"
    # 3. Return extracted text

    return {
        "text": "",
        "confidence": 0,
        "message": "OCR requires a vision-capable model (DeepSeek does not support image inputs). "
                   "Image URL has been logged for manual review.",
    }
