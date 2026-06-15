"""Application configuration via Pydantic Settings.

Reads from config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All application settings, sourced from config.yaml + env vars."""

    # ── LLM (DeepSeek) ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.1
    deepseek_max_tokens: int = 2048

    # ── Milvus ──
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_reviews: str = "review_embeddings"
    milvus_collection_pain_points: str = "pain_point_embeddings"
    milvus_collection_issues: str = "historical_issues"
    milvus_dimension: int = 1536

    # ── Embedding ──
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    hf_endpoint: str = ""

    # ── Data ──
    db_path: str = "data/reviews.db"
    import_dir: str = "data/imports"
    report_dir: str = "data/reports"

    # ── Alerts ──
    alerts_enabled: bool = True
    alert_threshold_high_freq: int = 15
    alert_time_window_hours: int = 24
    alert_sentiment_stddev: float = 2.0
    alert_volume_multiplier: float = 3.0

    # ── E-commerce Rules ──
    painpoint_categories: dict = {}

    # ── Logging ──
    log_level: str = "INFO"

    model_config = {"extra": "ignore"}

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Settings":
        """Load settings from config.yaml, with env var overrides."""
        kwargs: dict = {}

        # Load yaml defaults
        if Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

            llm = yaml_data.get("llm", {})
            if llm:
                kwargs["deepseek_api_key"] = _resolve_env(llm.get("api_key", ""))
                kwargs["deepseek_base_url"] = llm.get("base_url", "https://api.deepseek.com/v1")
                kwargs["deepseek_model"] = llm.get("model", "deepseek-chat")
                kwargs["deepseek_temperature"] = llm.get("temperature", 0.1)
                kwargs["deepseek_max_tokens"] = llm.get("max_tokens", 2048)

            milvus = yaml_data.get("milvus", {})
            if milvus:
                kwargs["milvus_host"] = milvus.get("host", "localhost")
                kwargs["milvus_port"] = milvus.get("port", 19530)

            ecommerce = yaml_data.get("ecommerce_rules", {})
            if ecommerce:
                kwargs["painpoint_categories"] = ecommerce.get("painpoint_categories", {})

            alerts = yaml_data.get("alerts", {})
            if alerts:
                kwargs["alerts_enabled"] = alerts.get("enabled", True)
                rules = alerts.get("rules", {})
                hfn = rules.get("high_freq_negative", {})
                kwargs["alert_threshold_high_freq"] = hfn.get("threshold", 15)
                ss = rules.get("sentiment_shift", {})
                kwargs["alert_sentiment_stddev"] = ss.get("stddev_multiplier", 2.0)
                vs = rules.get("volume_spike", {})
                kwargs["alert_volume_multiplier"] = vs.get("multiplier", 3.0)

            data_cfg = yaml_data.get("data", {})
            if data_cfg:
                kwargs["db_path"] = data_cfg.get("db_path", "data/reviews.db")
                kwargs["import_dir"] = data_cfg.get("import_dir", "data/imports")
                kwargs["report_dir"] = data_cfg.get("report_dir", "data/reports")

            logging_cfg = yaml_data.get("logging", {})
            if logging_cfg:
                kwargs["log_level"] = logging_cfg.get("level", "INFO")

        # Env vars override yaml
        env_map = {
            "DEEPSEEK_API_KEY": "deepseek_api_key",
            "DEEPSEEK_BASE_URL": "deepseek_base_url",
            "DEEPSEEK_MODEL": "deepseek_model",
            "MILVUS_HOST": "milvus_host",
            "MILVUS_PORT": "milvus_port",
            "HF_ENDPOINT": "hf_endpoint",
            "LOG_LEVEL": "log_level",
        }
        for env_var, field_name in env_map.items():
            val = os.environ.get(env_var)
            if val:
                kwargs[field_name] = val

        return cls(**kwargs)


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings
