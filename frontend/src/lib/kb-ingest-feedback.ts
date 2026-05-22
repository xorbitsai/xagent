export interface KnowledgeBaseErrorToastCopy {
  genericTitle: string
  embeddingTitle: string
  embeddingDescription: string
  rollbackTitle: string
  rollbackDescription: string
}

export interface KnowledgeBaseErrorToastContent {
  title: string
  description?: string
}

export interface KnowledgeBaseIngestionResultLike {
  collection?: string
  document_count?: number
  chunks_count?: number
  status: string
  message: string
  failed_step?: string
  file_name?: string
  doc_id?: string
  chunk_count?: number
  embedding_count?: number
  vector_count?: number
  documents_processed?: number
  chunks_created?: number
  parses_completed?: number
  embeddings_created?: number
  error?: string
}

export interface NormalizeKnowledgeBaseIngestionResultOptions {
  collection: string
  fileName?: string
}

export function normalizeKnowledgeBaseIngestionResult(
  result: KnowledgeBaseIngestionResultLike,
  options: NormalizeKnowledgeBaseIngestionResultOptions
): Required<
  Pick<
    KnowledgeBaseIngestionResultLike,
    | "collection"
    | "document_count"
    | "chunks_count"
    | "status"
    | "message"
    | "parses_completed"
    | "file_name"
    | "vector_count"
  >
> &
  Pick<
    KnowledgeBaseIngestionResultLike,
    | "failed_step"
    | "doc_id"
    | "embedding_count"
    | "embeddings_created"
    | "error"
  > {
  const fileName = result.file_name ?? options.fileName
  const documentCount = result.document_count
    ?? result.documents_processed
    ?? (result.doc_id ? 1 : 0)
  const chunkCount = result.chunks_count
    ?? result.chunks_created
    ?? result.chunk_count
    ?? 0
  const parseCount = result.parses_completed ?? (result.doc_id ? 1 : 0)
  const vectorCount = result.vector_count
    ?? result.embeddings_created
    ?? result.embedding_count
    ?? 0

  return {
    collection: result.collection ?? options.collection,
    document_count: documentCount,
    chunks_count: chunkCount,
    status: result.status,
    message: result.message,
    failed_step: result.failed_step,
    file_name: fileName ?? options.collection,
    parses_completed: parseCount,
    doc_id: result.doc_id,
    embedding_count: result.embedding_count,
    vector_count: vectorCount,
    embeddings_created: result.embeddings_created,
    error: result.error ?? (result.status === "success" ? undefined : result.message),
  }
}

function normalizeMessage(message: string): string {
  return message.replace(/\s+/g, " ").trim()
}

function truncateDescription(message: string, maxLength = 180): string {
  const normalized = normalizeMessage(message)
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, maxLength - 1).trimEnd()}…`
}

function extractOriginalIngestionError(message: string): string | null {
  const marker = "Original ingestion error:"
  const startIndex = message.indexOf(marker)
  if (startIndex === -1) {
    return null
  }
  const rawOriginal = message.slice(startIndex + marker.length).trim()
  const backupIndex = rawOriginal.indexOf("; backup restore also failed:")
  const original = backupIndex >= 0
    ? rawOriginal.slice(0, backupIndex).trim()
    : rawOriginal
  return original || null
}

function isEmbeddingConfigurationError(message: string): boolean {
  const normalized = message.toLowerCase()
  return (
    normalized.includes("no embedding model available") ||
    (
      normalized.includes("not found in hub") &&
      normalized.includes("environment configuration available for embedding")
    ) ||
    (
      normalized.includes("no environment configuration") &&
      normalized.includes("embedding")
    )
  )
}

function isRollbackFailure(message: string): boolean {
  return (
    message.startsWith("Failed to fully roll back ingest") ||
    message.startsWith("Failed to fully roll back cloud ingest")
  )
}

export function getKnowledgeBaseErrorToastContent(
  message: string,
  copy: KnowledgeBaseErrorToastCopy
): KnowledgeBaseErrorToastContent {
  const normalized = normalizeMessage(message)
  const originalError = extractOriginalIngestionError(normalized)
  const rootCause = originalError ?? normalized

  if (isEmbeddingConfigurationError(rootCause)) {
    return {
      title: copy.embeddingTitle,
      description: isRollbackFailure(normalized)
        ? `${copy.embeddingDescription} ${copy.rollbackDescription}`
        : copy.embeddingDescription,
    }
  }

  if (isRollbackFailure(normalized)) {
    return {
      title: copy.rollbackTitle,
      description: truncateDescription(rootCause),
    }
  }

  return {
    title: copy.genericTitle,
    description: truncateDescription(normalized),
  }
}

export function buildKnowledgeBaseErrorResult(
  collection: string,
  message: string,
  failedStep?: string,
  fileName?: string
): KnowledgeBaseIngestionResultLike {
  return {
    collection,
    document_count: 0,
    chunks_count: 0,
    status: "error",
    message,
    failed_step: failedStep,
    file_name: fileName,
  }
}
