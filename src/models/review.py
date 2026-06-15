"""ReviewRecord — Pydantic v2 model for a single product review."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class ReviewRecord(BaseModel):
    """A single product review after ingestion and cleaning."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: str = "unknown"
    product_id: str = ""
    content: str = ""
    rating: int = Field(default=3, ge=1, le=5)
    review_type: str = ""  # positive, neutral, negative, follow_up
    created_at: datetime = Field(default_factory=datetime.now)
    imported_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)
    is_valid: bool = True

    model_config = {"frozen": False}
