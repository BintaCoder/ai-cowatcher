"""Application configuration — all provider/model choices from environment."""

import json
import logging
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


WhisperModelSize = Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"]
WhisperComputeType = Literal["int8", "int8_float16", "float16", "float32"]
EscalationStrategy = Literal["heuristic", "prompt"]
UtteranceGateStrategy = Literal["heuristic", "prompt", "merged"]


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
    qdrant_knowledge_collection: str = Field(
        default="title_knowledge", alias="QDRANT_KNOWLEDGE_COLLECTION"
    )

    # ── Q&A response cache (exact Redis + semantic Qdrant) ─────────────────────
    # Default off for latency isolation; demos: make api QA_CACHE_ENABLED=true
    qa_cache_enabled: bool = Field(default=False, alias="QA_CACHE_ENABLED")
    qa_cache_collection: str = Field(default="qa_cache", alias="QA_CACHE_COLLECTION")
    # Playhead bucket width. 45s better matches multi-scene sitcom beats (>30s)
    # without serving cross-context answers; override per title if needed.
    qa_cache_ts_bucket_sec: int = Field(default=45, alias="QA_CACHE_TS_BUCKET_SEC")
    qa_cache_exact_ttl_sec: int = Field(default=7200, alias="QA_CACHE_EXACT_TTL_SEC")
    qa_cache_semantic_ttl_sec: int = Field(
        default=7200, alias="QA_CACHE_SEMANTIC_TTL_SEC"
    )
    # Cosine floor for semantic near-dup hits. Tuned from 0.90 → 0.88 on labeled
    # near-dup vs different pairs (see scripts/tune_qa_cache_threshold.py).
    qa_cache_semantic_threshold: float = Field(
        default=0.88, alias="QA_CACHE_SEMANTIC_THRESHOLD"
    )
    qa_cache_semantic_top_k: int = Field(default=3, alias="QA_CACHE_SEMANTIC_TOP_K")

    # ── Neo4j (character intelligence graph) ──────────────────────────────────
    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="cowatcher", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = Field(default="localhost:19000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="cowatcher", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    # local | minio — local uses OBJECT_STORE_LOCAL_DIR (default under project)
    object_store_backend: Literal["local", "minio"] = Field(
        default="local",
        alias="OBJECT_STORE_BACKEND",
    )
    object_store_local_dir: str = Field(
        default=".cowatcher-objects",
        alias="OBJECT_STORE_LOCAL_DIR",
    )
    # Max seconds of scene audio stored / sent to multimodal LLM
    scene_audio_max_sec: float = Field(default=20.0, alias="SCENE_AUDIO_MAX_SEC")
    scene_audio_enabled: bool = Field(default=True, alias="SCENE_AUDIO_ENABLED")
    # Attach retrieved clips to the conversation answer model
    multimodal_scene_audio_enabled: bool = Field(
        default=False,  # off for pilot latency — text scenes usually enough
        alias="MULTIMODAL_SCENE_AUDIO_ENABLED",
    )
    multimodal_max_clips: int = Field(default=2, alias="MULTIMODAL_MAX_CLIPS")
    # Pilot: minimize tool hops / model reasoning / multimodal.
    # Production startup fails if this is false (legacy multi-tool path reachable).
    pilot_low_latency: bool = Field(default=True, alias="PILOT_LOW_LATENCY")
    # Cap scenes + transcript size sent into the merged Gemini prompt (cost + TTFT).
    # 2 scenes × 200 chars/field keeps spoiler-safe grounding while cutting prompt size.
    evidence_max_scenes: int = Field(default=2, alias="EVIDENCE_MAX_SCENES")
    evidence_max_chars_per_field: int = Field(
        default=200, alias="EVIDENCE_MAX_CHARS_PER_FIELD"
    )
    # Short-lived reuse of BGE query vectors for similar/repeated questions.
    query_embedding_cache_ttl_sec: float = Field(
        default=90.0, alias="QUERY_EMBEDDING_CACHE_TTL_SEC"
    )
    query_embedding_cache_max: int = Field(
        default=256, alias="QUERY_EMBEDDING_CACHE_MAX"
    )
    # Cap completion tokens for brief who/what lines (TTFT-friendly under Gemini 3).
    # Pilot answers are ~1 short sentence; keep headroom for reasoning tokens first.
    llm_short_answer_max_tokens: int = Field(
        default=160, alias="LLM_SHORT_ANSWER_MAX_TOKENS"
    )

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

    # ── Speaker diarization (pyannote.audio) ──────────────────────────────────
    # Optional: install with `pip install '.[diarization]'` and set HUGGINGFACE_TOKEN.
    # If the package is missing, ingest continues with empty speaker clusters.
    diarization_enabled: bool = Field(default=True, alias="DIARIZATION_ENABLED")
    diarization_model: str = Field(
        default="pyannote/speaker-diarization-3.1", alias="DIARIZATION_MODEL"
    )
    huggingface_token: str = Field(default="", alias="HUGGINGFACE_TOKEN")

    # ── Character intelligence graph (offline enrichment) ─────────────────────
    # Minimum face/speaker co-occurrence count to link a voice to a face.
    character_link_min_cooccur: int = Field(default=1, alias="CHARACTER_LINK_MIN_COOCCUR")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ── Vision captioning (LiteLLM) ───────────────────────────────────────────
    vision_model: str = Field(default="gemini/gemini-3.5-flash-lite", alias="VISION_MODEL")
    vision_max_tokens: int = Field(default=256, alias="VISION_MAX_TOKENS")
    vision_caption_delay_sec: float = Field(default=1.0, alias="VISION_CAPTION_DELAY_SEC")
    vision_caption_max_retries: int = Field(default=8, alias="VISION_CAPTION_MAX_RETRIES")
    vision_frame_max_size: int = Field(default=512, alias="VISION_FRAME_MAX_SIZE")

    # ── Conversation LLM (LiteLLM) ────────────────────────────────────────────
    llm_primary_model: str = Field(default="gemini/gemini-3.6-flash", alias="LLM_PRIMARY_MODEL")
    llm_fallback_model: str = Field(
        default="gemini/gemini-3.6-flash",
        alias="LLM_FALLBACK_MODEL",
    )
    llm_mock_model: str = Field(default="mock-llm", alias="LLM_MOCK_MODEL")
    # Prefer online multimodal-capable models for co-watch answers (audio+text).
    # Override to ollama/* only if you accept text-only fall back when clips need mm.
    llm_tier_fast_model: str = Field(
        default="gemini/gemini-3.6-flash",
        alias="LLM_TIER_FAST_MODEL",
    )
    llm_tier_escalated_model: str = Field(
        default="gemini/gemini-3.6-flash",
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
    # Global answer budget. Keep ≥512 so Gemini 3 reasoning doesn't starve speech.
    # Short who/what lines use LLM_SHORT_ANSWER_MAX_TOKENS via _answer_max_tokens.
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")
    # Tool-call JSON + multi-step needs more headroom than spoken answer size.
    llm_tool_max_tokens: int = Field(default=768, alias="LLM_TOOL_MAX_TOKENS")
    llm_temperature: float = Field(default=0.35, alias="LLM_TEMPERATURE")
    # LiteLLM/Gemini 3: lower thinking so max_tokens isn't all reasoning.
    # Applied globally in LiteLLMCompletionClient — not overridden per persona.
    llm_reasoning_effort: str = Field(default="minimal", alias="LLM_REASONING_EFFORT")
    # Pilot cost instrumentation (USD per 1M tokens; not billing). Flash-lite class.
    llm_input_usd_per_mtok: float = Field(
        default=0.10, alias="LLM_INPUT_USD_PER_MTOK"
    )
    llm_output_usd_per_mtok: float = Field(
        default=0.40, alias="LLM_OUTPUT_USD_PER_MTOK"
    )
    # Soft alert when a viewer's cumulative estimated spend exceeds this (USD).
    session_cost_budget_usd: float = Field(
        default=0.50, alias="SESSION_COST_BUDGET_USD"
    )
    ollama_api_base: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_API_BASE",
    )
    # ── Utterance gate (mic / typed noise + intents) ──────────────────────────
    utterance_gate_enabled: bool = Field(default=True, alias="UTTERANCE_GATE_ENABLED")
    # heuristic = rules only
    # prompt = rules + ambiguous YES/NO via a second fast-LLM call
    # merged = free heuristics for clear cases; ambiguous + content go through one
    #          tagged completion (intent + answer in the same stream/call)
    utterance_gate_strategy: UtteranceGateStrategy = Field(
        default="merged",
        alias="UTTERANCE_GATE_STRATEGY",
    )

    # ── Real-time retrieval ─────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    navigation_top_k: int = Field(default=15, alias="NAVIGATION_TOP_K")
    knowledge_top_k: int = Field(default=5, alias="KNOWLEDGE_TOP_K")
    knowledge_dir: str = Field(default="knowledge", alias="KNOWLEDGE_DIR")

    # ── Event-driven ingestion (Kafka / RabbitMQ / in-memory) ─────────────────
    message_broker: Literal["memory", "rabbitmq", "kafka"] = Field(
        default="memory", alias="MESSAGE_BROKER"
    )
    ingest_topic: str = Field(default="title.ingest", alias="INGEST_TOPIC")
    ingest_queue_name: str = Field(default="title.ingest", alias="INGEST_QUEUE_NAME")
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/", alias="RABBITMQ_URL"
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_consumer_group: str = Field(
        default="cowatcher-ingest", alias="KAFKA_CONSUMER_GROUP"
    )

    # ── Prometheus / Grafana observability ──────────────────────────────────────
    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")
    worker_metrics_port: int = Field(default=9100, alias="WORKER_METRICS_PORT")

    # ── User conversation memory ────────────────────────────────────────────────
    user_memory_max_turns: int = Field(default=10, alias="USER_MEMORY_MAX_TURNS")
    user_memory_cache_turns: int = Field(default=20, alias="USER_MEMORY_CACHE_TURNS")
    user_memory_redis_ttl_sec: int = Field(default=3600, alias="USER_MEMORY_REDIS_TTL_SEC")

    # ── Companion persona (tone pack under ai_cowatcher/personas/) ─────────────
    default_persona_id: str = Field(
        default="easygoing_friend",
        alias="DEFAULT_PERSONA_ID",
    )

    # ── Cast / actor lookup (TMDB) ────────────────────────────────────────────
    # Public cast metadata is not a plot spoiler, so this is safe to expose.
    tmdb_api_key: str = Field(default="", alias="TMDB_API_KEY")
    tmdb_base_url: str = Field(default="https://api.themoviedb.org/3", alias="TMDB_BASE_URL")
    tmdb_timeout_sec: float = Field(default=8.0, alias="TMDB_TIMEOUT_SEC")
    tmdb_max_cast: int = Field(default=10, alias="TMDB_MAX_CAST")
    # Intermittent TLS resets (ISP/middlebox filtering) are common for TMDB on some
    # networks; retry transient connection failures before giving up.
    tmdb_max_retries: int = Field(default=5, alias="TMDB_MAX_RETRIES")
    tmdb_retry_backoff_sec: float = Field(default=0.5, alias="TMDB_RETRY_BACKOFF_SEC")
    # Hot cache TTL for cast after ingest/write-through (Redis).
    cast_cache_redis_ttl_sec: int = Field(
        default=7 * 24 * 3600, alias="CAST_CACHE_REDIS_TTL_SEC"
    )
    # JSON map of internal title_id -> human title used for TMDB search,
    # e.g. {"demo": "Kids", "thriller-001": "Knives Out (2019)"}
    title_names: str = Field(default="{}", alias="TITLE_NAMES")

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
    def cast_lookup_enabled(self) -> bool:
        """Cast tool is always attached: prefer ingest cache; fall back to TMDB when keyed."""
        return True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def neo4j_enabled(self) -> bool:
        """Character graph persists to Neo4j only when a URI is configured."""
        return bool(self.neo4j_uri)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def character_graph_enabled(self) -> bool:
        """Run the offline character-graph enrichment when Neo4j is configured."""
        return bool(self.neo4j_uri)

    @property
    def title_name_map(self) -> dict[str, str]:
        try:
            parsed = json.loads(self.title_names or "{}")
        except json.JSONDecodeError:
            logger.warning("TITLE_NAMES is not valid JSON; ignoring")
            return {}
        if not isinstance(parsed, dict):
            logger.warning("TITLE_NAMES must be a JSON object; ignoring")
            return {}
        return {str(key): str(value) for key, value in parsed.items()}

    def title_display_name(self, title_id: str) -> str | None:
        return self.title_name_map.get(title_id)

    def resolve_title_display_name(
        self, title_id: str, db_display_name: str | None = None
    ) -> str | None:
        """Prefer DB display_name, fall back to TITLE_NAMES env map. May return None."""
        if db_display_name:
            return db_display_name
        return self.title_display_name(title_id)

    @staticmethod
    def derive_title_from_id(title_id: str) -> str | None:
        """Best-effort human title guessed from an internal title_id.

        Examples: "Friends-1v" -> "Friends", "the_office_s01" -> "the office".
        Used only as a fallback when no display_name / TITLE_NAMES entry exists,
        so the cast lookup still has a reasonable term to search TMDB with.
        """
        if not title_id:
            return None
        tokens = re.split(r"[-_\s]+", title_id.strip())
        kept: list[str] = []
        for token in tokens:
            low = token.lower()
            if not low:
                continue
            if low.isdigit():
                continue
            # version / sequence markers: 1v, v1, s01, e01, 001, etc.
            if re.fullmatch(r"[a-z]?\d+[a-z]?", low):
                continue
            if low in {"clip", "clips", "part", "ep", "episode", "season", "v", "vid", "video"}:
                continue
            kept.append(token)
        guess = " ".join(kept).strip()
        return guess or None

    def effective_search_title(
        self, title_id: str, db_display_name: str | None = None
    ) -> str | None:
        """Confident title (DB/env) if available, else a guess derived from title_id."""
        resolved = self.resolve_title_display_name(title_id, db_display_name)
        if resolved:
            return resolved
        return self.derive_title_from_id(title_id)

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

    def legacy_multi_tool_path_reachable(self) -> bool:
        """True when the multi-round tool LLM loop can run (not merged-only)."""
        if self.pilot_low_latency:
            return False
        # Gate merged still avoids multi-tool for content; other strategies do not.
        return self.utterance_gate_strategy != "merged"

    def validate_pilot_latency_config(self) -> None:
        """Prod must stay on 1-retrieve + 1-LLM merged path; dev logs when legacy is on."""
        env = (self.app_env or "").strip().lower()
        is_prod = env in ("production", "prod")
        if is_prod:
            if not self.pilot_low_latency:
                raise ValueError(
                    "PILOT_LOW_LATENCY must be true in production "
                    "(legacy multi-tool agent path is not allowed)."
                )
            if self.utterance_gate_strategy != "merged":
                raise ValueError(
                    "UTTERANCE_GATE_STRATEGY must be 'merged' in production "
                    "(legacy multi-tool / gate-LLM paths are not allowed)."
                )
            return
        if not self.pilot_low_latency or self.utterance_gate_strategy != "merged":
            logger.warning(
                "legacy multi-tool path is reachable "
                "(PILOT_LOW_LATENCY=%s UTTERANCE_GATE_STRATEGY=%s APP_ENV=%s)",
                self.pilot_low_latency,
                self.utterance_gate_strategy,
                self.app_env,
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_pilot_whisper_config()
    settings.validate_pilot_latency_config()
    return settings
