from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class VectorDBType(str, Enum):
    """Supported vector database backend types."""

    LANCEDB = "lancedb"
    WEAVIATE = "weaviate"
    WEAVIATE_SAAS = "weaviate_saas"
    CHROMADB = "chromadb"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    PGVECTOR = "pgvector"
    ELASTICSEARCH = "elasticsearch"
    OPEN_SEARCH = "open_search"
    REDIS = "redis"
    FAISS = "faiss"
    TYPESENSE = "typesense"
    ZILLIZ = "zilliz"


class ModelConfig(BaseModel):
    id: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 180.0
    abilities: Optional[List[str]] = None
    description: Optional[str] = None
    max_retries: int = 10


class ChatModelConfig(ModelConfig):
    model_provider: str = "openai"  # openai, zhipu, dashscope, etc.
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None
    context_window: Optional[int] = None  # Total context window in tokens
    thinking_mode: bool = False
    # Runtime-only settings for a configured router model. They are populated
    # from the user's Auto configuration and are intentionally not stored on a
    # concrete provider model row.
    router_config_name: Optional[str] = None
    router_candidate_models: Optional[List[str]] = None
    router_fallback_model: Optional[str] = None
    # Runtime-only, and deliberately narrow: not a general "no ambient
    # credentials" switch. When set, ``create_base_llm`` refuses a missing or
    # placeholder ``api_key`` (which adapters would otherwise replace from the
    # process environment) and Auto models, and the Azure adapter never sends
    # an Entra token from ``AZURE_OPENAI_AD_TOKEN``. Only ``create_base_llm``
    # reads the flag: the LangChain factory (``chat/langchain.py``) ignores
    # it, Entra token included. An adapter that can add another ambient
    # credential must honor the flag the same way. Nothing else is enforced:
    # adapters and provider SDKs still read other settings from the
    # environment, for example
    # - ``ANTHROPIC_AUTH_TOKEN``: older anthropic releases (0.84 among them)
    #   send it as a Bearer token next to the key;
    # - ``OPENAI_CUSTOM_HEADERS``/``ANTHROPIC_CUSTOM_HEADERS``: newer releases
    #   send them on every request, auth headers included;
    # - endpoints whenever no ``base_url`` reaches the client: ``*_BASE_URL``
    #   variables such as ``OPENAI_BASE_URL`` (also for the official OpenAI
    #   URL, which the adapter passes as None), ``ANTHROPIC_BASE_URL``,
    #   ``GOOGLE_GEMINI_BASE_URL``, ``ZHIPU_BASE_URL``, ``ZAI_BASE_URL``,
    #   ``DEEPSEEK_BASE_URL`` and ``DASHSCOPE_BASE_URL``, and Azure's
    #   ``AZURE_OPENAI_ENDPOINT``/``OPENAI_API_BASE``;
    # - ``GOOGLE_GENAI_USE_VERTEXAI`` (sends Gemini requests to Vertex AI),
    #   Azure's ``OPENAI_API_VERSION``, and ``OPENAI_ORG_ID``/
    #   ``OPENAI_PROJECT_ID``.
    # A host keeps these out of the process that runs caller keys.
    # Hosts running configurations owned by someone other than the deployment
    # (for example a model a user configured with their own key) set this;
    # deployment-owned models keep the default and their environment-based
    # credentials.
    explicit_credentials_only: bool = False
    # Runtime-only native Bedrock settings. Persistence/UI wiring is owned by
    # the dedicated provider-settings change; these fields let the factory and
    # adapter carry the values without overloading unrelated AWS environment
    # variables or process-global bearer-token state.
    bedrock_region: Optional[str] = None
    bedrock_auth_mode: str = "auto"


class ImageModelConfig(ModelConfig):
    model_provider: str = "openai"  # openai, zhipu, dashscope, etc.
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None


class VideoModelConfig(ModelConfig):
    model_provider: str = "volcengine-ark"  # Volcengine/BytePlus ModelArk video


class EmbeddingModelConfig(ModelConfig):
    model_provider: str = "dashscope"  # openai, zhipu, dashscope, etc.
    dimension: Optional[int] = None
    instruct: Optional[str] = None


class RerankModelConfig(ModelConfig):
    model_provider: str = "dashscope"  # dashscope, xinference, etc.
    top_n: Optional[int] = None
    instruct: Optional[str] = None


class SpeechModelConfig(ModelConfig):
    """Configuration for speech models (ASR and TTS)."""

    model_provider: str = "xinference"  # xinference, etc.
    language: Optional[str] = None  # Default language code (e.g., 'zh', 'en')
    # TTS-specific configuration
    voice: Optional[str] = (
        None  # Default voice/speaker for TTS (e.g., 'female', 'male')
    )
    format: Optional[str] = None  # Audio format for TTS (e.g., 'mp3', 'wav', 'pcm')
    sample_rate: Optional[int] = None  # Sample rate for TTS in Hz (e.g., 24000, 48000)


class SoundEffectModelConfig(ModelConfig):
    """Configuration for text-to-sound-effect models."""

    model_provider: str = "elevenlabs"


class MusicModelConfig(ModelConfig):
    """Configuration for prompt-to-music models."""

    model_provider: str = "elevenlabs"


class VectorDBConfig(ModelConfig):
    """Configuration for vector database backend (e.g. LanceDB, Weaviate).

    Note: When persisted via SQLAlchemyModelHub, the optional extra config dict
    is stored in the base model's ``abilities`` JSON column (semantic repurpose;
    for other categories ``abilities`` is Optional[List[str]]).
    """

    db_type: VectorDBType = VectorDBType.LANCEDB
    config: dict = Field(default_factory=dict)
