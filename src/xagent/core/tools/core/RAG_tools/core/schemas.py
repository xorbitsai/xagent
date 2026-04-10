"""Core data models and schemas for RAG tools.

This module defines Pydantic models for all RAG tools data structures.
These models ensure type safety and validation across the core layer.

All models use Pydantic v2 ConfigDict for future compatibility.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# Default configurable values (avoid scattering literals)
DEFAULT_SEARCH_TOP_K: int = 5
DEFAULT_RERANK_TOP_K: int = 5
DEFAULT_RRF_K: int = 60
DEFAULT_DENSE_WEIGHT: float = 0.5
DEFAULT_SPARSE_WEIGHT: float = 0.5
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_EMBEDDING_BATCH_SIZE: int = 10
DEFAULT_EMBEDDING_CONCURRENT: int = 10
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_DELAY_SECONDS: float = 1.0

# ------------------------- Enums -------------------------


class ParseMethod(Enum):
    """Available parsing methods"""

    DEFAULT = "default"
    PYPDF = "pypdf"
    PDFPLUMBER = "pdfplumber"
    UNSTRUCTURED = "unstructured"
    PYMUPDF = "pymupdf"
    DEEPDOC = "deepdoc"

    def __str__(self) -> str:
        return self.value


class ChunkStrategy(Enum):
    """Available chunk strategies"""

    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    MARKDOWN = "markdown"

    def __str__(self) -> str:
        return self.value


class StepType(Enum):
    """Processing stage types for version management"""

    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"

    def __str__(self) -> str:
        return self.value


class DocumentProcessingStatus(str, Enum):
    """States representing document ingestion lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    CHUNKED = "chunked"  # Document has been chunked but not yet embedded
    PARTIALLY_EMBEDDED = "partially_embedded"  # Document has some embeddings but not all chunks are embedded
    SUCCESS = "success"  # Document fully processed: all chunks have embeddings
    FAILED = "failed"
    CANCELLED = "cancelled"  # Task was cancelled by user or system

    def __str__(self) -> str:
        return self.value


class TaskProgress(BaseModel):
    """Progress state for a single RAG task (ingestion, retrieval, etc.)."""

    task_id: str = Field(..., description="Unique task identifier")
    user_id: Optional[int] = Field(default=None, description="User ID for isolation")
    task_type: str = Field(..., description="Type of task (e.g. ingestion, retrieval)")
    status: DocumentProcessingStatus = Field(
        default=DocumentProcessingStatus.PENDING,
        description="Current task status",
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Human-readable current step",
    )
    overall_progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Progress in [0, 1]",
    )
    start_time: Optional[float] = Field(
        default=None, description="Unix timestamp when started"
    )
    end_time: Optional[float] = Field(
        default=None, description="Unix timestamp when ended"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra task metadata",
    )


class ProgressUpdateEvent(BaseModel):
    """Event payload for real-time progress WebSocket broadcasts."""

    task_id: str = Field(..., description="Task identifier")
    task_type: str = Field(..., description="Task type")
    status: DocumentProcessingStatus = Field(
        ...,
        description="Current status",
    )
    current_step: Optional[str] = Field(default=None, description="Current step label")
    overall_progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Progress in [0, 1]",
    )
    timestamp: float = Field(..., description="Event time (Unix timestamp)")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra event data (e.g. start_time, end_time, metadata)",
    )
    event_type: str = Field(
        default="progress",
        description="Event kind (e.g. progress, custom)",
    )


class IndexType(Enum):
    """Available index types for vector search."""

    HNSW = "IVF_HNSW_SQ"
    IVFPQ = "IVF_PQ"

    def __str__(self) -> str:
        return self.value


class IndexMetric(Enum):
    """Available distance metrics for vector indexes."""

    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"

    def __str__(self) -> str:
        return self.value


class IndexStatus(Enum):
    """Index status enumeration for tracking index lifecycle."""

    INDEX_READY = "index_ready"
    INDEX_BUILDING = "index_building"
    NO_INDEX = "no_index"
    INDEX_CORRUPTED = "index_corrupted"
    BELOW_THRESHOLD = "below_threshold"
    READONLY = "readonly"

    def __str__(self) -> str:
        return self.value


class FTSIndexStatus(Enum):
    """Status of Full-Text Search index."""

    FTS_INDEX_READY = "fts_index_ready"
    FTS_INDEX_MISSING = "fts_index_missing"
    FTS_INDEX_CREATING = "fts_index_creating"
    FTS_INDEX_FAILED = "fts_index_failed"


class IndexOperation(Enum):
    """Index operation result types (e.g. for embedding write response)."""

    CREATED = "created"
    READY = "ready"
    SKIPPED = "skipped"
    SKIPPED_THRESHOLD = "skipped_threshold"
    FAILED = "failed"
    UPDATED = "updated"

    def __str__(self) -> str:
        return self.value


class FusionStrategy(Enum):
    """Fusion strategies for hybrid search."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Linear weighted combination

    def __str__(self) -> str:
        return self.value


class SearchErrorLevel(Enum):
    """Error severity levels for search operations."""

    CRITICAL = "critical"  # Block execution, must raise error
    WARNING = "warning"  # Degrade gracefully, continue execution
    INFO = "info"  # Log only, normal execution

    def __str__(self) -> str:
        return self.value


class SearchFallbackAction(Enum):
    """Fallback actions when index is unavailable."""

    BRUTE_FORCE = "brute_force"  # Fall back to brute-force search
    REBUILD_INDEX = "rebuild_index"  # Trigger index rebuild
    SAMPLE_SEARCH = "sample_search"  # Search on data sample
    PARTIAL_RESULTS = "partial_results"  # Return available results only

    def __str__(self) -> str:
        return self.value


class SearchType(Enum):
    """Unified search types for the document search pipeline."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value


# ------------------------- Register schemas -------------------------


class RegisterDocumentRequest(BaseModel):
    """Request model for document registration.

    This model defines the input parameters for registering a document into the LanceDB system.
    """

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")
    file_id: Optional[str] = Field(
        None, description="UploadedFile file_id for stable file association"
    )
    source_path: str = Field(..., description="Absolute path to uploaded file")

    file_type: Optional[str] = Field(
        None, description="File type (auto-detected if not provided)"
    )
    doc_id: Optional[str] = Field(
        None, description="Document ID (auto-generated if not provided)"
    )
    uploaded_at: Optional[datetime] = Field(
        None,
        description="Upload timestamp (defaults to now)",
    )
    user_id: Optional[int] = Field(
        None, description="User ID for multi-tenancy (None for legacy data)"
    )


class RegisterDocumentResponse(BaseModel):
    """Response model for document registration.

    This model defines the output structure returned after successful document registration.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="The document ID (generated or provided)")

    created: bool = Field(
        ..., description="True if new document created, False if existed"
    )

    content_hash: str = Field(..., description="SHA256 hash of the document content")


class ParsedParagraph(BaseModel):
    """Model for a parsed paragraph from a document.

    This model represents a single paragraph extracted from a document during parsing.
    It includes the text content and associated metadata.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="The paragraph text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the paragraph"
    )


class ParseDocumentRequest(BaseModel):
    """Request model for document parsing.

    This model defines the input parameters for parsing a document.
    """

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")
    doc_id: str = Field(..., description="Document ID to parse")
    parse_method: ParseMethod = Field(..., description="Parsing method to use")
    params: Optional[Dict[str, Any]] = Field(
        None, description="Optional parameters for parsing"
    )
    user_id: Optional[int] = Field(
        None, description="User ID for multi-tenancy (None for legacy data)"
    )
    is_admin: bool = Field(
        False, description="Whether user is admin (can access all data)"
    )


class ParseDocumentResponse(BaseModel):
    """Response model for document parsing.

    This model defines the output structure returned after successful document parsing.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="The document ID that was parsed")
    parse_hash: str = Field(..., description="SHA256 hash of parse configuration")
    paragraphs: List[ParsedParagraph] = Field(
        ..., description="List of parsed paragraphs"
    )
    written: bool = Field(
        ..., description="True if parse record was written to database"
    )


# ------------------------- Parse Result Display schemas -------------------------


class ParsedTextSegmentDisplay(BaseModel):
    """Display model for a parsed text segment."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content of the segment")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata including positions, style, etc."
    )


class ParsedTableDisplay(BaseModel):
    """Display model for a parsed table."""

    model_config = ConfigDict(frozen=True)

    type: Literal["table"] = "table"
    html: Optional[str] = Field(None, description="HTML representation of the table")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata including positions, caption, etc."
    )


class ParsedFigureDisplay(BaseModel):
    """Display model for a parsed figure."""

    model_config = ConfigDict(frozen=True)

    type: Literal["figure"] = "figure"
    text: str = Field(..., description="Caption or text associated with the figure")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including positions, image_path, etc.",
    )


ParsedElementDisplay = Annotated[
    Union[ParsedTextSegmentDisplay, ParsedTableDisplay, ParsedFigureDisplay],
    Field(discriminator="type"),
]


class ParseResultResponse(BaseModel):
    """Response model for displaying parse results with pagination."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="The document ID")
    parse_hash: str = Field(..., description="SHA256 hash of parse configuration")
    elements: List[ParsedElementDisplay] = Field(
        default_factory=list, description="Ordered list of parsed elements"
    )
    pagination: Dict[str, Any] = Field(
        ...,
        description="Pagination information including page, page_size, total counts",
    )


# ------------------------- Chunk schemas -------------------------


class ChunkDocumentRequest(BaseModel):
    """Request model for chunking a parsed document version."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")
    doc_id: str = Field(..., description="Document ID to chunk")
    parse_hash: str = Field(..., description="Parse version hash to use as source")
    chunk_strategy: ChunkStrategy = Field(
        ..., description="Chunking strategy identifier"
    )
    chunk_size: int = Field(
        ..., gt=0, description="Target chunk size in characters; must be positive"
    )
    chunk_overlap: int = Field(
        ...,
        ge=0,
        description="Overlap between consecutive chunks in characters; must be non-negative",
    )
    headers_to_split_on: Optional[List[tuple[str, str]]] = Field(
        default=None, description="Markdown headers split rules if strategy requires"
    )
    separators: Optional[List[str]] = Field(
        default=None, description="Custom separators list for recursive splitting"
    )


class ChunkDocumentResponse(BaseModel):
    """Response model for chunking results, aligned to current implementation."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Document ID that was chunked")
    parse_hash: str = Field(..., description="Parse version hash that was used")
    chunk_count: int = Field(
        ..., ge=0, description="Number of chunks produced; must be non-negative"
    )
    stats: Dict[str, Any] = Field(..., description="Chunk statistics")
    created: bool = Field(
        ..., description="True if chunk records were written to database"
    )


# ------------------------- Embed schemas -------------------------


class ChunkForEmbedding(BaseModel):
    """Model for chunk data ready for embedding."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Document ID")

    chunk_id: str = Field(..., description="Unique chunk identifier")
    parse_hash: str = Field(..., description="Parse version hash")
    text: str = Field(..., description="Chunk text content")
    chunk_hash: str = Field(..., description="Hash of chunk content and params")
    index: int = Field(..., ge=0, description="Chunk position index (0-based)")
    page_number: Optional[int] = Field(
        None,
        gt=0,
        description="Page number if available; must be positive if provided",
    )
    section: Optional[str] = Field(None, description="Section name if available")
    anchor: Optional[str] = Field(None, description="Anchor reference if available")
    json_path: Optional[str] = Field(None, description="JSON path if available")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the chunk"
    )


class EmbeddingReadRequest(BaseModel):
    """Request model for reading chunks for embedding."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")
    doc_id: str = Field(..., description="Document ID to read chunks from")
    parse_hash: str = Field(..., description="Parse version hash to filter chunks")
    model: str = Field(..., description="Model name for embedding")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional filters for chunk selection"
    )


class EmbeddingReadResponse(BaseModel):
    """Response model for reading chunks for embedding."""

    model_config = ConfigDict(frozen=True)

    chunks: List[ChunkForEmbedding] = Field(
        ..., description="Chunks ready for embedding"
    )
    total_count: int = Field(
        ..., ge=0, description="Total number of chunks found; must be non-negative"
    )
    pending_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks needing embedding; must be non-negative",
    )


class ChunkEmbeddingData(BaseModel):
    """Model for chunk embedding data to be written."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    parse_hash: str = Field(..., description="Parse version hash")
    model: str = Field(..., description="Model name used for embedding")
    vector: List[float] = Field(..., description="Embedding vector")
    text: str = Field(..., description="Original chunk text")
    chunk_hash: str = Field(..., description="Hash of chunk content and params")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the chunk"
    )


class EmbeddingWriteRequest(BaseModel):
    """Request model for writing embeddings to database."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")

    embeddings: List[ChunkEmbeddingData] = Field(
        ..., description="List of embedding data to write"
    )
    create_index: bool = Field(
        True, description="Whether to create/update index after writing"
    )


class EmbeddingWriteResponse(BaseModel):
    """Response model for writing embeddings."""

    model_config = ConfigDict(frozen=True)

    upsert_count: int = Field(
        ..., ge=0, description="Number of embeddings written; must be non-negative"
    )
    deleted_stale_count: int = Field(
        ...,
        ge=0,
        description="Number of stale embeddings deleted; must be non-negative",
    )
    index_status: str = Field(
        ..., description="Index operation status: created/updated/skipped"
    )


# ------------------------- Search/Retrieval schemas -------------------------


class SearchResult(BaseModel):
    """Individual search result item."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID within the document")
    text: str = Field(..., description="Text content of the chunk")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1, higher is better); must be between 0 and 1 inclusive",
    )
    parse_hash: str = Field(..., description="Parse version hash")
    model_tag: str = Field(..., description="Embedding model identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Chunk creation timestamp",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the chunk (page_number, section, source, etc.)",
    )
    # Original scores and ranks for RRF fallback (optional, populated during hybrid search)
    vector_score: Optional[float] = Field(
        None,
        description="Original vector search score (before rerank/fusion)",
    )
    fts_score: Optional[float] = Field(
        None,
        description="Original FTS search score (before rerank/fusion)",
    )
    vector_rank: Optional[int] = Field(
        None,
        ge=1,
        description="Original rank in vector search results (1-based)",
    )
    fts_rank: Optional[int] = Field(
        None,
        ge=1,
        description="Original rank in FTS search results (1-based)",
    )


class DenseSearchRequest(BaseModel):
    """Request model for dense vector search."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection name for data isolation")
    model_tag: str = Field(..., description="Embedding model identifier")
    query_vector: List[float] = Field(
        ..., description="Query vector for similarity search"
    )
    top_k: int = Field(default=10, description="Number of top results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters for search results"
    )
    readonly: bool = Field(
        default=False, description="Readonly mode - don't trigger index building"
    )


class SearchWarning(BaseModel):
    """Warning information for degraded search operations."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="Warning code identifier")
    message: str = Field(..., description="Human-readable warning message")
    affected_models: List[str] = Field(
        default_factory=list, description="List of affected model tags"
    )
    fallback_action: SearchFallbackAction = Field(
        ..., description="Fallback action that was applied"
    )


class PerformanceImpact(BaseModel):
    """Performance impact information for degraded operations."""

    model_config = ConfigDict(frozen=True)

    expected_latency_ms: float = Field(
        ..., description="Expected latency in milliseconds"
    )
    actual_latency_ms: float = Field(..., description="Actual latency in milliseconds")
    degradation_reason: str = Field(
        ..., description="Reason for performance degradation"
    )


class FallbackInfo(BaseModel):
    """Fallback information for search operations."""

    model_config = ConfigDict(frozen=True)

    applied: bool = Field(..., description="Whether fallback was applied")
    reason: str = Field(..., description="Reason for fallback")
    performance_impact: PerformanceImpact = Field(
        ..., description="Performance impact details"
    )


class DenseSearchResponse(BaseModel):
    """Response model for dense vector search."""

    model_config = ConfigDict(frozen=True)

    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of results returned; must be non-negative",
    )
    status: str = Field(
        default="success",
        description="Operation status: success|partial_success|failed",
    )
    warnings: List[SearchWarning] = Field(
        default_factory=list, description="Warnings for degraded operations"
    )
    index_status: IndexStatus = Field(..., description="Current index status")
    index_advice: Optional[str] = Field(
        default=None, description="Human-readable index advice"
    )
    idempotency_key: Optional[str] = Field(
        default=None, description="Idempotency key for task deduplication"
    )
    fallback_info: Optional[FallbackInfo] = Field(
        default=None, description="Fallback information if applied"
    )
    nprobes: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of partitions to probe for ANN search; must be positive if provided",
    )
    refine_factor: Optional[int] = Field(
        default=None,
        gt=0,
        description="Refine factor for re-ranking results in memory; must be positive if provided",
    )


class SparseSearchResponse(BaseModel):
    """Response model for sparse (FTS) search operations."""

    model_config = ConfigDict(frozen=True)

    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of results returned; must be non-negative",
    )
    status: str = Field(
        default="success",
        description="Operation status: success|partial_success|failed",
    )
    warnings: List[SearchWarning] = Field(
        default_factory=list, description="Warnings for degraded operations"
    )
    fts_enabled: bool = Field(..., description="Whether FTS index is available")
    query_text: str = Field(..., description="Original query text used for search")


class FusionConfig(BaseModel):
    """Configuration for hybrid search fusion."""

    model_config = ConfigDict(frozen=True)

    strategy: FusionStrategy = Field(
        default=FusionStrategy.RRF, description="Fusion strategy to use"
    )
    rrf_k: int = Field(
        default=DEFAULT_RRF_K,
        gt=0,
        description="RRF constant (k parameter); must be a positive integer",
    )
    dense_weight: float = Field(
        default=DEFAULT_DENSE_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for dense results in linear fusion (0-1 inclusive)",
    )
    sparse_weight: float = Field(
        default=DEFAULT_SPARSE_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for sparse results in linear fusion (0-1 inclusive)",
    )
    normalize_scores: bool = Field(
        default=True,
        description="Whether to normalize scores before fusion (Min-Max)",
    )


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search operations."""

    model_config = ConfigDict(frozen=True)

    results: List[SearchResult] = Field(..., description="Fused search results")
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of results returned; must be non-negative",
    )
    status: str = Field(
        default="success",
        description="Operation status: success|partial_success|failed",
    )
    warnings: List[SearchWarning] = Field(
        default_factory=list, description="Warnings for degraded operations"
    )
    fusion_config: FusionConfig = Field(..., description="Fusion configuration used")
    dense_count: int = Field(
        ...,
        ge=0,
        description="Number of dense results contributed; must be non-negative",
    )
    sparse_count: int = Field(
        ...,
        ge=0,
        description="Number of sparse results contributed; must be non-negative",
    )
    index_status: IndexStatus = Field(..., description="Dense index status")
    index_advice: Optional[str] = Field(
        default=None, description="Human-readable index advice"
    )


class IndexResult(BaseModel):
    """Structured result from index creation operations.

    This model replaces the previous string-based return format for create_index,
    providing type-safe access to index status, advice, and FTS enabled state.

    Attributes:
        status: Index creation status (e.g., "index_ready", "readonly", "failed")
        advice: Optional advice message for further actions
        fts_enabled: Whether FTS index is actually enabled (separate from vector index)
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Index creation status")
    advice: Optional[str] = Field(
        default=None, description="Human-readable index advice"
    )
    fts_enabled: bool = Field(
        default=False, description="Whether FTS index is enabled on text column"
    )


class SearchConfig(BaseModel):
    """Configuration for the unified document search pipeline."""

    model_config = ConfigDict(frozen=True)

    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Requested search strategy"
    )
    top_k: int = Field(
        default=DEFAULT_SEARCH_TOP_K,
        ge=1,
        le=100,
        description="Maximum number of search results to return",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters applied during retrieval"
    )
    fusion_config: Optional[FusionConfig] = Field(
        default=None, description="Optional override for hybrid fusion configuration"
    )
    embedding_model_id: str = Field(
        ...,
        description=(
            "Embedding model identifier registered in model hub (required for dense/hybrid)."
        ),
    )
    rerank_model_id: Optional[str] = Field(
        default=None,
        description="Optional rerank model identifier registered in model hub",
    )
    rerank_top_k: Optional[int] = Field(
        default=DEFAULT_RERANK_TOP_K,
        description="Optional override for rerank result count",
    )
    readonly: bool = Field(
        default=False,
        description="Whether retrieval operations should avoid index modifications",
    )
    nprobes: Optional[int] = Field(
        default=None, description="Number of partitions to probe for ANN searches"
    )
    refine_factor: Optional[int] = Field(
        default=None, description="Refine factor for ANN search re-ranking"
    )
    fallback_to_sparse: bool = Field(
        default=True,
        description="Allow hybrid search to fallback to sparse when embedding fails",
    )


class SearchPipelineResult(BaseModel):
    """Unified response payload for document search pipeline."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ...,
        description="Pipeline status: success|partial_success|error",
    )
    search_type: SearchType = Field(
        ..., description="Actual search strategy executed (post-fallback if any)"
    )
    results: List[SearchResult] = Field(
        default_factory=list, description="Search results (possibly reranked)"
    )
    result_count: int = Field(..., description="Number of results returned")
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings and fallback messages"
    )
    message: str = Field(..., description="Human-readable pipeline outcome message")
    used_rerank: bool = Field(
        default=False, description="Whether rerank model was applied to the results"
    )


# ------------------------- Generation schemas -------------------------


class GenerateResponse(BaseModel):
    """Response schema for text generation operations.

    Attributes:
        generated_text: The main text content generated by the LLM.
        status: The status of the generation operation (e.g., 'success', 'failed', 'partial_success').
        model_name: The name of the LLM model used for generation.
        warnings: Optional list of dictionaries containing warning details during generation.
        error: Optional error message if the generation failed.
        latency_ms: Optional. The time taken for the generation operation in milliseconds.
        prompt_tokens: Optional. The number of tokens in the input prompt.
        completion_tokens: Optional. The number of tokens in the generated completion.
        total_tokens: Optional. The total number of tokens (prompt + completion).
    """

    model_config = ConfigDict(frozen=True)

    generated_text: str = Field(
        description="The main text content generated by the LLM."
    )
    status: Literal["success", "failed", "partial_success"] = Field(
        default="success", description="The status of the generation operation."
    )
    model_name: str = Field(
        description="The name of the LLM model used for generation."
    )
    warnings: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Optional list of warning details during generation.",
    )
    error: Optional[str] = Field(
        default=None, description="Optional error message if the generation failed."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Optional. The time taken for the generation operation in milliseconds.",
    )
    prompt_tokens: Optional[int] = Field(
        default=None, description="Optional. The number of tokens in the input prompt."
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Optional. The number of tokens in the generated completion.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Optional. The total number of tokens (prompt + completion).",
    )


# ------------------------- Prompt Manager schemas -------------------------


class PromptTemplate(BaseModel):
    """Model for prompt template with version management.

    This model represents a prompt template stored in LanceDB with support
    for version management and metadata tracking.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the prompt template (UUID as string)",
    )
    name: str = Field(..., description="Human-readable name for the prompt template")
    template: str = Field(..., description="The actual prompt template content")
    version: int = Field(default=1, description="Version number of the prompt template")
    is_latest: bool = Field(
        default=True, description="Whether this is the latest version"
    )
    metadata: Optional[str] = Field(
        default=None,
        description="Optional metadata as JSON string (author, description, tags, etc.)",
    )
    user_id: Optional[int] = Field(
        default=None,
        description="User ID for multi-tenancy. None for legacy data.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Last update timestamp",
    )


# ------------------------- Ingestion pipeline schemas -------------------------


class IngestionConfig(BaseModel):
    """Configuration values for the document ingestion pipeline."""

    model_config = ConfigDict(frozen=True)

    # DeepDoc parsing/runtime configuration (applied via environment during parse)
    deepdoc_processing_mode: Optional[str] = Field(
        default=None,
        description="DeepDoc processing mode (e.g., 'pipeline', 'default').",
    )
    deepdoc_parallel_threads: Optional[int] = Field(
        default=None,
        ge=1,
        description="DeepDoc parallel threads (DEEPDOC_PARALLEL_THREADS).",
    )
    deepdoc_reserve_cpu: Optional[int] = Field(
        default=None,
        ge=0,
        description="DeepDoc reserved CPU cores (DEEPDOC_RESERVE_CPU).",
    )
    deepdoc_limiter_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        description="DeepDoc CapacityLimiter capacity (DEEPDOC_LIMITER_CAPACITY).",
    )
    deepdoc_pipeline_monitor: Optional[bool] = Field(
        default=None,
        description="Enable DeepDoc pipeline monitor (DEEPDOC_PIPELINE_MONITOR).",
    )
    deepdoc_pipeline_s1_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="DeepDoc S1 worker count (DEEPDOC_PIPELINE_S1_WORKERS).",
    )
    deepdoc_gpu_sessions: Optional[int] = Field(
        default=None,
        ge=0,
        description="DeepDoc GPU sessions count/preference (DEEPDOC_GPU_SESSIONS).",
    )

    # DashScope embedding configuration (passed as params to resolver with hub > env fallback)
    embedding_base_url: Optional[str] = Field(
        default=None,
        description="Override DashScope base URL for embedding requests.",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="Override DashScope API key for embedding requests.",
    )
    embedding_timeout_sec: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Override embedding request timeout (seconds).",
    )

    parse_method: ParseMethod = Field(
        ParseMethod.DEFAULT, description="Parse method used during parse_document step"
    )
    chunk_strategy: ChunkStrategy = Field(
        ChunkStrategy.RECURSIVE, description="Chunk strategy passed to chunk_document"
    )
    chunk_method: Optional[str] = Field(
        default=None,
        description="Custom chunk method identifier. If provided, takes precedence over chunk_strategy",
    )
    chunk_size: Optional[int] = Field(
        default=DEFAULT_CHUNK_SIZE,
        gt=0,
        description="Chunk size passed to chunk_document; must be a positive integer. If None, semantic splitting is used without size limits.",
    )
    chunk_overlap: int = Field(
        DEFAULT_CHUNK_OVERLAP,
        ge=0,
        description="Chunk overlap passed to chunk_document; must be non-negative",
    )
    headers_to_split_on: Optional[List[tuple[str, str]]] = Field(
        default=None, description="Markdown headers split rules for markdown strategy"
    )
    separators: Optional[List[str]] = Field(
        default=None, description="Custom separators for recursive/markdown strategies"
    )
    use_token_count: bool = Field(
        default=False,
        description="If True, chunk_size and chunk_overlap are in tokens (tiktoken); only applies to RECURSIVE strategy",
    )
    tiktoken_encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding name when use_token_count=True (e.g. cl100k_base for GPT-4/3.5). Should align with config.DEFAULT_TIKTOKEN_ENCODING.",
    )
    enable_protected_content: bool = Field(
        default=True,
        description="If True, do not split inside code blocks, formulas, tables (P1).",
    )
    protected_patterns: Optional[List[str]] = Field(
        default=None,
        description="Optional regex patterns for protected regions; None uses config default.",
    )
    table_context_size: int = Field(
        default=0,
        ge=0,
        description="Chars from prev/next chunk to attach to table chunks; 0 = off (P2).",
    )
    image_context_size: int = Field(
        default=0,
        ge=0,
        description="Chars from prev/next chunk to attach to image chunks; 0 = off (P2).",
    )
    embedding_model_id: Optional[str] = Field(
        default=None,
        description=(
            "Embedding model identifier registered in AgentOS model hub. If omitted, "
            "the pipeline attempts to auto-detect a single available embedding model."
        ),
    )

    # Collection configuration management
    collection_locked: bool = Field(
        default=False,
        description="Whether to lock collection configuration. When True, enforces strict config validation.",
    )
    allow_mixed_parse_methods: bool = Field(
        default=False,
        description="Whether to allow mixed parse methods within the collection. When False, enforces type-based parse method consistency.",
    )
    skip_config_validation: bool = Field(
        default=False,
        description="Skip collection configuration validation. Use with caution.",
    )
    embedding_batch_size: int = Field(
        DEFAULT_EMBEDDING_BATCH_SIZE,
        gt=0,
        description="Batch size for embedding provider requests; must be positive",
    )
    embedding_concurrent: int = Field(
        DEFAULT_EMBEDDING_CONCURRENT,
        gt=0,
        description=(
            "Maximum concurrent requests for embedding computation when using "
            "async mode (for models that don't support batch processing, e.g., text-embedding-v4). "
            "Must be positive. Adjust based on machine configuration and API rate limits."
        ),
    )
    embedding_use_async: bool = Field(
        False,
        description=(
            "Whether to use async concurrent processing for embeddings. "
            "Set to True for models that don't support batch processing (e.g., text-embedding-v4). "
            "When True, embeddings are processed concurrently using asyncio instead of batch API calls."
        ),
    )
    max_retries: int = Field(
        DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum number of retries for embedding provider failures; must be non-negative",
    )
    retry_delay: float = Field(
        DEFAULT_RETRY_DELAY_SECONDS,
        ge=0.0,
        description="Delay in seconds between embedding retries; must be non-negative",
    )


class IngestionStepResult(BaseModel):
    """Metadata for a successfully completed pipeline step."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Step identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata captured for the step"
    )


class IngestionResult(BaseModel):
    """Structured response for the document ingestion pipeline."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Pipeline status: success|error|partial")
    doc_id: Optional[str] = Field(
        None, description="Document identifier produced by register_document"
    )
    parse_hash: Optional[str] = Field(
        None, description="Parse hash produced during parse_document step"
    )
    chunk_count: int = Field(
        0, ge=0, description="Number of chunks created; must be non-negative"
    )
    embedding_count: int = Field(
        0, ge=0, description="Number of embeddings generated; must be non-negative"
    )
    vector_count: int = Field(
        0,
        ge=0,
        description="Number of vectors written to storage; must be non-negative",
    )
    completed_steps: List[IngestionStepResult] = Field(
        default_factory=list, description="List of successfully completed steps"
    )
    failed_step: Optional[str] = Field(
        None, description="Pipeline step where failure occurred, if any"
    )
    message: str = Field(..., description="Human-readable summary of pipeline result")
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings encountered"
    )
    file_id: Optional[str] = Field(
        None,
        description="Uploaded file ID for preview/download via /api/files (when ingest registers the file)",
    )


# ------------------------- Management -------------------------


class DocumentProcessingRecord(BaseModel):
    """Record of how a specific document was processed in the collection."""

    model_config = ConfigDict(frozen=True)

    # Document identifiers
    doc_id: str = Field(..., description="Document identifier")
    collection_name: str = Field(..., description="Collection this document belongs to")

    # 📄 Original information
    source_path: str = Field(..., description="Original file path")
    file_type: str = Field(..., description="File extension/type")
    file_size: int = Field(..., description="File size in bytes")

    # 🔧 Processing configuration (deterministic)
    parsing_method: str = Field(..., description="Parsing method used")
    parsing_config: Dict[str, Any] = Field(
        default_factory=dict, description="Specific parsing parameters used"
    )

    chunking_method: str = Field(..., description="Chunking method used")
    chunking_config: Dict[str, Any] = Field(
        default_factory=dict, description="Specific chunking parameters used"
    )

    # 📊 Processing result statistics
    parse_success: bool = Field(True, description="Whether parsing succeeded")
    chunks_generated: int = Field(0, description="Number of chunks created")
    embeddings_generated: int = Field(0, description="Number of vectors created")

    # 🕒 Processing timestamps
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    parse_duration_ms: Optional[int] = Field(
        None, description="Parsing time in milliseconds"
    )
    chunk_duration_ms: Optional[int] = Field(
        None, description="Chunking time in milliseconds"
    )
    embed_duration_ms: Optional[int] = Field(
        None, description="Embedding time in milliseconds"
    )

    # 🏷️ Version control
    processing_version: str = Field(
        ..., description="Version of processing pipeline used"
    )


class CollectionInfo(BaseModel):
    """Aggregate metadata for a single collection with embedding binding."""

    model_config = ConfigDict(frozen=False)  # Allow updates for lazy initialization

    # 🏷️ Version control for backward compatibility
    schema_version: str = Field(
        default="1.0.0", description="Schema version for migration compatibility"
    )

    # Basic identifier
    name: str = Field(..., description="Collection identifier")

    # 🎯 Core binding: Embedding configuration (lazy initialization)
    embedding_model_id: Optional[str] = Field(
        default=None,  # None indicates not initialized
        description="Fixed embedding model ID. Set during first document ingestion.",
    )
    embedding_dimension: Optional[int] = Field(
        default=None,  # None indicates not initialized
        ge=0,
        description="Vector dimension. Auto-detected from embedding model.",
    )

    # 📊 Statistics
    documents: int = Field(0, description="Total number of registered documents")
    processed_documents: int = Field(
        0, description="Number of successfully processed documents"
    )
    parses: int = Field(0, description="Number of parse records")
    chunks: int = Field(0, description="Number of chunk records")
    embeddings: int = Field(0, description="Number of embedding vectors")

    # 📋 Document list
    document_names: List[str] = Field(
        default_factory=list,
        description="Distinct source paths for documents within the collection",
    )

    # ⚙️ Configuration management
    collection_locked: bool = Field(
        default=False,
        description="Whether to lock collection configuration. When True, enforces strict config validation.",
    )
    allow_mixed_parse_methods: bool = Field(
        default=False,
        description="Whether to allow mixed parse methods within the collection. When False, enforces type-based parse method consistency.",
    )
    skip_config_validation: bool = Field(
        default=False,
        description="Whether to skip configuration validation during ingestion. Use with caution.",
    )

    # 📝 Stored Ingestion Config
    ingestion_config: Optional[IngestionConfig] = Field(
        default=None,
        description="Default ingestion configuration for the collection.",
    )

    # 🕒 Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    last_accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    # 🔮 Extension fields for future versions
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for future extensions"
    )

    @property
    def is_initialized(self) -> bool:
        """Check if collection has been initialized with embedding config."""
        return (
            self.embedding_model_id is not None and self.embedding_dimension is not None
        )

    @classmethod
    def from_storage(cls, data: dict) -> "CollectionInfo":
        """Load from storage dict with in-memory schema normalization.

        Legacy rows (e.g. ``schema_version`` missing / ``0.0.0``) are upgraded
        **in memory only** via :func:`~.migration_utils.migrate_collection_metadata`
        with ``infer_embedding=False`` so this path does **not** open LanceDB or
        scan embedding tables (read-side-effect-free). For full migration with
        embedding inference, call ``migrate_collection_metadata(data)`` explicitly
        (e.g. admin repair or write pipeline).
        """
        import json
        import math

        from ..utils.migration_utils import migrate_collection_metadata

        # 1. Deserialize JSON fields
        if isinstance(data.get("extra_metadata"), str):
            data["extra_metadata"] = json.loads(data["extra_metadata"])
        if isinstance(data.get("document_names"), str):
            data["document_names"] = json.loads(data["document_names"])
        if isinstance(data.get("ingestion_config"), str):
            raw_ingestion_config = data["ingestion_config"].strip()
            if raw_ingestion_config:
                data["ingestion_config"] = json.loads(raw_ingestion_config)
            else:
                data["ingestion_config"] = None

        # 2. Convert NaN values to None (LanceDB stores NULL as NaN for numeric fields)
        for key, value in data.items():
            if isinstance(value, float) and math.isnan(value):
                data[key] = None

        # Handle empty string fallback for string fields that might have been stored as "" to avoid non-null errors
        if data.get("embedding_model_id") == "":
            data["embedding_model_id"] = None

        if data.get("embedding_dimension") == -1:
            data["embedding_dimension"] = None

        # 3. Check version and migrate if needed (no DB access on read path)
        current_version = "1.0.0"
        data_version = data.get("schema_version", "0.0.0")

        if data_version < current_version:
            data = migrate_collection_metadata(data, infer_embedding=False)

        return cls(**data)

    def to_storage(self) -> dict:
        """Serialize for LanceDB storage."""
        import json

        data = self.model_dump()

        # Serialize complex types to JSON strings for LanceDB
        data["extra_metadata"] = json.dumps(data["extra_metadata"])
        data["document_names"] = json.dumps(data["document_names"])

        # Serialize ingestion_config if present
        if data.get("ingestion_config"):
            data["ingestion_config"] = json.dumps(data["ingestion_config"])
        else:
            data["ingestion_config"] = (
                ""  # Use empty string instead of None to prevent LanceDB non-null schema errors
            )

        if data.get("embedding_model_id") is None:
            data["embedding_model_id"] = ""

        if data.get("embedding_dimension") is None:
            data["embedding_dimension"] = -1

        return data


class ListCollectionsResult(BaseModel):
    """Response payload for the list collections operation."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Operation status: success|error")
    collections: List[CollectionInfo] = Field(
        default_factory=list, description="Collection statistics"
    )
    total_count: int = Field(..., description="Number of collections discovered")
    message: str = Field(..., description="Human-readable status message")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during aggregation",
    )


class DocumentStats(BaseModel):
    """Aggregate statistics for a single document within a collection."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection identifier")
    doc_id: str = Field(..., description="Document identifier inside the collection")
    document_exists: bool = Field(..., description="Whether the document record exists")
    parse_count: int = Field(0, description="Number of parse records for the document")
    chunk_count: int = Field(
        0, description="Number of chunks generated for the document"
    )
    embedding_count: int = Field(
        0, description="Total number of embeddings across all matched tables"
    )
    embedding_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-model-tag embedding counts keyed by table name",
    )
    status: DocumentProcessingStatus = Field(
        default=DocumentProcessingStatus.PENDING,
        description="Current processing status derived from the latest ingestion run",
    )
    last_message: Optional[str] = Field(
        default=None,
        description="Most recent status message or error information for the document",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Timestamp of the most recent document status update",
    )


class DocumentStatsResult(BaseModel):
    """Response payload for document statistics query."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Operation status: success|error")
    data: DocumentStats | None = Field(
        default=None, description="Document statistics when query succeeds"
    )
    message: str = Field(..., description="Human-readable status message")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while collecting statistics",
    )


class DocumentInfo(BaseModel):
    """Document information with status."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Document identifier")
    collection: str = Field(..., description="Collection identifier")
    source_path: Optional[str] = Field(None, description="Source file path")
    file_type: Optional[str] = Field(None, description="File type")
    status: Optional[str] = Field(
        None, description="Processing status from ingestion_runs table"
    )
    message: Optional[str] = Field(
        None, description="Status message from ingestion_runs table"
    )
    uploaded_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Upload timestamp",
    )


class DocumentSummary(BaseModel):
    """Lightweight summary for document list views."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(..., description="Collection identifier")
    doc_id: str = Field(..., description="Document identifier inside the collection")
    source_path: Optional[str] = Field(
        default=None, description="Original document source path if available"
    )
    status: DocumentProcessingStatus = Field(
        default=DocumentProcessingStatus.PENDING,
        description="Current processing status for the document",
    )
    message: Optional[str] = Field(
        default=None,
        description="Latest status message or error summary for the document",
    )
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Document registration timestamp if tracked",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="Last time the document status was updated",
    )
    chunk_count: int = Field(
        0, ge=0, description="Most recent chunk count snapshot for the document"
    )
    embedding_count: int = Field(
        0, ge=0, description="Most recent embedding count snapshot for the document"
    )


class DocumentListResult(BaseModel):
    """Response payload for document summaries within a collection."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Operation status: success|error")
    documents: List[DocumentSummary] = Field(
        default_factory=list, description="Documents associated with the collection"
    )
    total_count: int = Field(
        ..., ge=0, description="Total number of documents returned in the response"
    )
    message: str = Field(..., description="Human-readable status message")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while enumerating documents",
    )


class DocumentOperationResult(BaseModel):
    """Standard response for document management operations."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="Operation status: success|error")
    collection: str = Field(..., description="Collection identifier")
    doc_id: str = Field(
        ..., description="Document identifier affected by the operation"
    )
    new_status: DocumentProcessingStatus = Field(
        default=DocumentProcessingStatus.PENDING,
        description="Document status after the operation completes",
    )
    message: str = Field(..., description="Human-readable summary of the operation")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or follow-up actions for the operation",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata describing the operation impact",
    )


class CollectionOperationDetail(BaseModel):
    """Per-document detail for collection-level management operations."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(
        ..., description="Document identifier affected by the operation"
    )
    status: DocumentProcessingStatus = Field(
        ..., description="Resulting status recorded for the document"
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional message describing individual document handling",
    )


class CollectionOperationResult(BaseModel):
    """Response payload for collection-level management operations."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ..., description="Operation status: success|partial_success|error"
    )
    collection: str = Field(
        ..., description="Collection identifier affected by the operation"
    )
    message: str = Field(
        ..., description="Human-readable summary of the collection operation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while processing the collection",
    )
    affected_documents: List[CollectionOperationDetail] = Field(
        default_factory=list,
        description="Subset of documents impacted by the collection operation",
    )
    deleted_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregated deletion counts per table when applicable",
    )


# ------------------------- Web Crawler schemas -------------------------


class WebCrawlConfig(BaseModel):
    """Configuration for website crawling and ingestion.

    This model defines how a website should be crawled, including URL filtering,
    depth limits, concurrency control, and content extraction rules.
    """

    model_config = ConfigDict(frozen=True)

    # Basic configuration
    start_url: str = Field(..., description="Starting URL for crawling")
    max_pages: int = Field(
        default=100,
        ge=1,
        description="Maximum number of pages to crawl",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        description="Maximum crawl depth from start URL",
    )

    # URL filtering configuration
    url_patterns: Optional[List[str]] = Field(
        default=None,
        description="URL match patterns (regex) - only matching URLs will be crawled",
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="URL exclusion patterns (regex) - matching URLs will be skipped",
    )
    same_domain_only: bool = Field(
        default=True,
        description="Whether to only crawl URLs from the same domain",
    )

    # Content extraction configuration
    content_selector: Optional[str] = Field(
        default=None,
        description="CSS selector for extracting main content area (e.g., 'main article')",
    )
    remove_selectors: Optional[List[str]] = Field(
        default=None,
        description="CSS selectors for elements to remove (e.g., ['nav', 'footer'])",
    )

    # Concurrency and rate limiting
    concurrent_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of concurrent requests",
    )
    request_delay: float = Field(
        default=1.0,
        ge=0,
        description="Delay between requests in seconds",
    )

    # Other configuration
    user_agent: Optional[str] = Field(
        default="Mozilla/5.0 (xagent WebCrawler/1.0)",
        description="User-Agent string for HTTP requests",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to respect robots.txt rules",
    )


class CrawlResult(BaseModel):
    """Result of crawling a single page."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., description="The URL that was crawled")
    title: Optional[str] = Field(None, description="Page title")
    content_markdown: str = Field(..., description="Page content as markdown")
    status: str = Field(..., description="Crawl status: success|failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    depth: int = Field(..., ge=0, description="Crawl depth from start URL")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        description="When the page was crawled",
    )
    content_length: int = Field(
        ..., ge=0, description="Length of content in characters"
    )
    links_found: int = Field(
        default=0, ge=0, description="Number of links found on this page"
    )


class WebIngestionResult(BaseModel):
    """Result of website crawling and knowledge base ingestion.

    This model provides comprehensive statistics about the crawling and
    ingestion process, including success/failure counts and timing.
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        ...,
        description="Overall status: success|error|partial",
    )
    collection: str = Field(..., description="Target collection name")

    # Crawl statistics
    total_urls_found: int = Field(
        ..., ge=0, description="Total number of unique URLs discovered"
    )
    pages_crawled: int = Field(
        ..., ge=0, description="Number of successfully crawled pages"
    )
    pages_failed: int = Field(
        ..., ge=0, description="Number of pages that failed to crawl"
    )

    # Ingestion statistics
    documents_created: int = Field(
        ..., ge=0, description="Number of documents created in collection"
    )
    chunks_created: int = Field(..., ge=0, description="Total number of chunks created")
    embeddings_created: int = Field(
        ..., ge=0, description="Total number of embeddings generated"
    )

    # Details
    crawled_urls: List[str] = Field(
        default_factory=list,
        description="List of successfully crawled URLs",
    )
    failed_urls: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of failed URLs to error messages",
    )

    message: str = Field(..., description="Human-readable summary message")
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    elapsed_time_ms: int = Field(
        ..., ge=0, description="Total elapsed time in milliseconds"
    )
