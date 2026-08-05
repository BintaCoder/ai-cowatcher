"""Application configuration — all provider/model choices from environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


WhisperModelSize = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]
WhisperComputeType = Literal["int8", "int8_float16", "float16", "float32"]
EscalationStrategy = Literal["heuristic", "prompt"]


class Settings(BaseSettings):
    """Central config for the co-watcher pilot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Global mock switch ───────────────────────────────────────────────────
    # When true, AI components use cheap local mocks; infra health checks stay real.
    mock_mode: bool = Field(default=True, alias="MOCK_MODE")

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = Field(default="ai-cowatcher", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="cowatcher", alias="POSTGRES_USER")
    postgres_password: str = Field(default="cowatcher", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="cowatcher", alias="POSTGRES_DB")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="title_segments", alias="QDRANT_COLLECTION")

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="cowatcher", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # ── FFmpeg (subprocess) ───────────────────────────────────────────────────
    ffmpeg_bin: str = Field(default="ffmpeg", alias="FFMPEG_BIN")

    # ── Speech recognition (pilot: small/medium, int8, CPU) ─────────────────
    whisper_model_size: WhisperModelSize = Field(default="small", alias="WHISPER_MODEL_SIZE")
    whisper_compute_type: WhisperComputeType = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_num_workers: int = Field(default=1, alias="WHISPER_NUM_WORKERS")

    # ── Face recognition ──────────────────────────────────────────────────────
    insightface_model: str = Field(default="buffalo_l", alias="INSIGHTFACE_MODEL")
    insightface_ctx_id: int = Field(default=-1, alias="INSIGHTFACE_CTX_ID")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ── Vision captioning (LiteLLM) ───────────────────────────────────────────
    vision_model: str = Field(default="gemini/gemini-2.0-flash-lite", alias="VISION_MODEL")
    vision_max_tokens: int = Field(default=256, alias="VISION_MAX_TOKENS")

    # ── Conversation LLM (LiteLLM) ────────────────────────────────────────────
    llm_primary_model: str = Field(default="openai/gpt-4o-mini", alias="LLM_PRIMARY_MODEL")
    llm_fallback_model: str = Field(
        default="anthropic/claude-3-5-haiku-latest",
        alias="LLM_FALLBACK_MODEL",
    )
    llm_mock_model: str = Field(default="mock-llm", alias="LLM_MOCK_MODEL")
    llm_tier_fast_model: str = Field(default="openai/gpt-4o-mini", alias="LLM_TIER_FAST_MODEL")
    llm_tier_escalated_model: str = Field(
        default="openai/gpt-4o",
        alias="LLM_TIER_ESCALATED_MODEL",
    )
    llm_mock_tier_fast_model: str = Field(
        default="mock-llm-fast",
        alias="LLM_MOCK_TIER_FAST_MODEL",
    )
    llm_mock_tier_escalated_model: str = Field(
        default="mock-llm-escalated",
        alias="LLM_MOCK_TIER_ESCALATED_MODEL",
    )
    llm_escalation_strategy: EscalationStrategy = Field(
        default="heuristic",
        alias="LLM_ESCALATION_STRATEGY",
    )
    llm_escalation_min_chars: int = Field(default=120, alias="LLM_ESCALATION_MIN_CHARS")
    llm_escalation_keywords: str = Field(
        default="why,how,explain,compare,motivation,relationship,theme,symbolism,foreshadow",
        alias="LLM_ESCALATION_KEYWORDS",
    )
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    # ── Real-time retrieval ─────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")

    # ── Provider API keys ─────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_escalation_keyword_list(self) -> list[str]:
        return [
            keyword.strip().lower()
            for keyword in self.llm_escalation_keywords.split(",")
            if keyword.strip()
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_fast_model(self) -> str:
        if self.mock_mode:
            return self.llm_mock_tier_fast_model
        return self.llm_tier_fast_model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_escalated_model(self) -> str:
        if self.mock_mode:
            return self.llm_mock_tier_escalated_model
        return self.llm_tier_escalated_model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_llm_model(self) -> str:
        """Default conversation tier (fast). Kept for backward compatibility."""
        return self.conversation_fast_model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_vision_model(self) -> str:
        """Model routed through LiteLLM for offline scene captioning."""
        if self.mock_mode:
            return self.llm_mock_model
        return self.vision_model

    def validate_pilot_whisper_config(self) -> None:
        """Pilot guardrail — block expensive large-v3 until engagement is proven."""
        if self.whisper_model_size in ("large-v2", "large-v3"):
            raise ValueError(
                "WHISPER_MODEL_SIZE must be small or medium for the pilot "
                "(large models are cost-blocked until engagement is proven)."
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_pilot_whisper_config()
    return settings
