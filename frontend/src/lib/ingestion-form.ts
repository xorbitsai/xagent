import { parseSeparatorsInput } from "@/lib/separators"

/** Minimal ingestion config shape used when appending to FormData for ingest/ingest-web APIs. */
export interface IngestionConfigForm {
  parse_method: string
  chunk_strategy: string
  chunk_size: number
  chunk_overlap: number
  separators?: string
  embedding_model_id: string
  embedding_batch_size: number
  max_retries: number
  retry_delay: number
}

/**
 * Appends ingestion config fields (including optional separators when chunk_strategy is recursive)
 * to the given FormData. Use for both /api/kb/ingest and /api/kb/ingest-web requests.
 */
export function appendIngestionConfigToFormData(
  formData: FormData,
  config: IngestionConfigForm
): void {
  formData.append("parse_method", config.parse_method)
  formData.append("chunk_strategy", config.chunk_strategy)
  formData.append("chunk_size", config.chunk_size.toString())
  formData.append("chunk_overlap", config.chunk_overlap.toString())
  if (
    config.chunk_strategy === "recursive" &&
    config.separators != null &&
    config.separators.trim() !== ""
  ) {
    const parsed = parseSeparatorsInput(config.separators)
    if (parsed.length > 0) {
      formData.append("separators", JSON.stringify(parsed))
    }
  }
  formData.append("embedding_model_id", config.embedding_model_id)
  formData.append(
    "embedding_batch_size",
    config.embedding_batch_size.toString()
  )
  formData.append("max_retries", config.max_retries.toString())
  formData.append("retry_delay", config.retry_delay.toString())
}
