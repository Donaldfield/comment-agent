"""Review data cleaner — rule-based filters + optional LLM classification.

Strategy: Rule-based filters catch ~80% of junk for free (regex, length,
dedup). LLM only sees the remaining ambiguous cases.
"""

import hashlib
import json
import logging
import re
from typing import Optional

from src.models.review import ReviewRecord

logger = logging.getLogger(__name__)

# Minimum content length (Chinese characters) to be considered meaningful
MIN_CONTENT_LENGTH = 4

# Patterns for meaningless content
MEANINGLESS_PATTERNS = [
    re.compile(r"^[，。！？、…\s]+$"),          # Pure punctuation
    re.compile(r"^[\U0001F300-\U0001FAFF]+$"), # Pure emoji
    re.compile(r"^[0-9]+$"),                     # Pure numbers
    re.compile(r"^(.)\1{3,}$"),                  # Same char repeated (aaaaa)
    re.compile(r"^(好|差|行|可|嗯|哦|是|对|的|了|吧|吗|呢|啊|呀|哈)+$"),  # Single filler words
]

# Patterns for ad/spam detection (rule-based)
AD_PATTERNS = [
    re.compile(r"http[s]?://"),                # URLs
    re.compile(r"微信号|微信|wechat|wx[:：]"),    # WeChat contact
    re.compile(r"加我|加群|扫码|二维码"),          # Contact farming
    re.compile(r"兼职|赚钱|日赚|月入|代理|加盟"),   # MLM/recruitment
    re.compile(r"免费领取|免费送|点击领取"),         # Freebie scams
    re.compile(r"[\w.]+@[\w.]+"),               # Email addresses
    re.compile(r"1[3-9]\d{9}"),                 # Phone numbers (Chinese)
    re.compile(r"关注.+公众号|关注.+抖音|关注.+微博"), # Social follow farming
]


def clean_rule_based(records: list[ReviewRecord]) -> list[ReviewRecord]:
    """Apply rule-based filters. Fast, free, catches ~80% of junk.

    Marks invalid records by setting is_valid=False.
    Does NOT delete anything — keeps audit trail.
    """
    seen_hashes: set[str] = set()
    cleaned: list[ReviewRecord] = []

    for record in records:
        content = record.content.strip()
        is_valid = True

        # Check 1: Minimum length
        if len(content) < MIN_CONTENT_LENGTH:
            is_valid = False
            logger.debug("Too short: %s", record.id)

        # Check 2: Meaningless patterns
        elif _matches_any(content, MEANINGLESS_PATTERNS):
            is_valid = False
            logger.debug("Meaningless: %s", record.id)

        # Check 3: Ad/spam patterns (rule-based pre-filter)
        elif _matches_any(content, AD_PATTERNS):
            is_valid = False
            logger.debug("Ad/spam detected: %s", record.id)

        # Check 4: Exact duplicate detection
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in seen_hashes:
            is_valid = False
            logger.debug("Duplicate: %s", record.id)
        else:
            seen_hashes.add(content_hash)

        # Create cleaned copy
        cleaned_record = ReviewRecord(
            id=record.id,
            platform=record.platform,
            product_id=record.product_id,
            content=content,
            rating=record.rating,
            review_type=record.review_type,
            created_at=record.created_at,
            imported_at=record.imported_at,
            metadata=dict(record.metadata),
            is_valid=is_valid,
        )
        cleaned.append(cleaned_record)

    valid_count = sum(1 for r in cleaned if r.is_valid)
    logger.info(
        "Rule-based cleaning: %d/%d valid (%.1f%%)",
        valid_count, len(cleaned),
        valid_count / max(len(cleaned), 1) * 100,
    )
    return cleaned


def clean_with_llm(
    records: list[ReviewRecord],
    llm,
    batch_size: int = 20,
) -> list[ReviewRecord]:
    """Use LLM to classify borderline reviews as valid/ad/spam/meaningless.

    Only sends reviews that passed rule-based checks but may still be
    low-quality. This minimizes API cost.
    """
    if not records:
        return records

    prompt_registry = None  # Lazy import to avoid circular dependency
    from src.llm.prompts import PromptRegistry
    prompt_registry = PromptRegistry()

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        reviews_json = json.dumps(
            [{"id": r.id, "content": r.content} for r in batch],
            ensure_ascii=False,
        )

        system_prompt, user_prompt = prompt_registry.render(
            "review_clean.jinja2", reviews_json=reviews_json
        )

        try:
            response = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_format="json_object",
            )
            result = json.loads(response)

            # Build lookup for fast update
            llm_results = {
                item["review_id"]: item
                for item in result.get("results", [])
            }

            for record in batch:
                llm_result = llm_results.get(record.id)
                if llm_result:
                    record.is_valid = llm_result.get("is_valid", True)
                    if not record.is_valid:
                        record.metadata["invalid_reason"] = llm_result.get(
                            "invalid_reason", ""
                        )
                    cleaned = llm_result.get("cleaned_content", "")
                    if cleaned and cleaned != record.content:
                        record.content = cleaned

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM cleaning parse error for batch %d: %s", i, e)
            # Keep records as-is on LLM failure (fail open)
            continue
        except Exception as e:
            logger.error("LLM cleaning failed for batch %d: %s", i, e)
            continue

    valid_count = sum(1 for r in records if r.is_valid)
    logger.info(
        "LLM cleaning: %d/%d valid (%.1f%%)",
        valid_count, len(records),
        valid_count / max(len(records), 1) * 100,
    )
    return records


def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    """Check if text matches any compiled regex pattern."""
    return any(p.search(text) for p in patterns)
