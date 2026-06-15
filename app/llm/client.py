"""DeepSeek LLM adapter — works with langchain-deepseek.

Exposes a compatible `generate()` interface so all src/ modules
can call it without changes.
"""

import json
import logging
from typing import Optional

from langchain_deepseek import ChatDeepSeek

from app.config import get_settings

logger = logging.getLogger(__name__)

_llm_instance: Optional[ChatDeepSeek] = None


def get_llm_client() -> ChatDeepSeek:
    """Get or create the DeepSeek ChatModel singleton."""
    global _llm_instance
    if _llm_instance is None:
        s = get_settings()
        if not s.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Set the environment variable or update config.yaml."
            )
        _llm_instance = ChatDeepSeek(
            model=s.deepseek_model,
            api_key=s.deepseek_api_key,
            api_base=s.deepseek_base_url,
            temperature=s.deepseek_temperature,
            max_tokens=s.deepseek_max_tokens,
        )
        logger.info("DeepSeek LLM initialized: model=%s base_url=%s", s.deepseek_model, s.deepseek_base_url)
    return _llm_instance


def generate(
    prompt: str,
    system_prompt: str = "",
    response_format: str = "text",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a completion — compatible with the old BaseLLM.generate().

    Args:
        prompt: User message.
        system_prompt: System message.
        response_format: "text" or "json_object".
        temperature: Override default temperature.
        max_tokens: Override default max_tokens.

    Returns:
        Raw response text from the LLM.
    """
    llm = get_llm_client()

    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("user", prompt))

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if response_format == "json_object":
        # DeepSeek supports OpenAI-compatible response_format
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    try:
        response = llm.invoke(messages, **kwargs)
        content = response.content if hasattr(response, "content") else str(response)
        logger.debug("LLM call completed, tokens: %s", getattr(response, "usage_metadata", "unknown"))
        return content
    except Exception as e:
        logger.error("DeepSeek LLM call failed: %s", e)
        raise
