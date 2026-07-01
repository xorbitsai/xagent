export interface CollectionDocumentInfo {
  filename: string
  file_id?: string
  doc_id?: string
}

/** Public per-document ingestion lifecycle statuses (mirrors backend enum). */
export type DocumentIngestionStatus =
  | "pending"
  | "running"
  | "chunked"
  | "partially_embedded"
  | "success"
  | "failed"
  | "cancelled"

/** Frontend-only status for a file that has been submitted but has no backend row yet. */
export type DocumentDisplayStatus = DocumentIngestionStatus | "uploading"

export interface KnowledgeBaseDocumentStatus {
  filename: string
  file_id?: string
  doc_id?: string
  status: DocumentDisplayStatus
  message?: string
  updated_at?: string
  can_delete: boolean
}

/** Statuses that will not change without another upload — polling can stop. */
const TERMINAL_DOCUMENT_STATUSES: ReadonlySet<DocumentDisplayStatus> = new Set([
  "success",
  "failed",
  "cancelled",
])

export function isTerminalDocumentStatus(status: DocumentDisplayStatus): boolean {
  return TERMINAL_DOCUMENT_STATUSES.has(status)
}

function normalizeDocumentStatus(value: unknown): DocumentIngestionStatus {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : ""
  switch (normalized) {
    case "pending":
    case "running":
    case "chunked":
    case "partially_embedded":
    case "success":
    case "failed":
    case "cancelled":
      return normalized
    default:
      return "success"
  }
}

/**
 * Fetch per-document ingestion status for a collection.
 *
 * Returns `null` on auth/not-found terminal errors (401/403/404) so callers can
 * stop polling; throws on other transient failures so callers can retain the
 * last rows and retry.
 */
export async function fetchCollectionDocumentStatuses(
  apiUrl: string,
  collectionName: string,
  requester: (url: string) => Promise<Response>
): Promise<KnowledgeBaseDocumentStatus[] | null> {
  const response = await requester(
    `${apiUrl}/api/kb/collections/${encodeURIComponent(collectionName)}/documents`
  )
  if (response.status === 401 || response.status === 403 || response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error(`Failed to load document status (HTTP ${response.status})`)
  }
  const data = await response.json().catch(() => null)
  const rows = Array.isArray(data?.documents) ? data.documents : []
  return rows
    .filter((row: unknown): row is Record<string, unknown> => !!row && typeof row === "object")
    .map((row: Record<string, unknown>) => ({
      filename: typeof row.filename === "string" ? row.filename : "",
      file_id: normalizeOptionalIdentifier(row.file_id),
      doc_id: normalizeOptionalIdentifier(row.doc_id),
      status: normalizeDocumentStatus(row.status),
      message: typeof row.message === "string" && row.message ? row.message : undefined,
      updated_at: typeof row.updated_at === "string" && row.updated_at ? row.updated_at : undefined,
      can_delete: row.can_delete !== false,
    }))
    .filter((row: KnowledgeBaseDocumentStatus) => row.filename.length > 0)
}

/**
 * Merge backend status rows with optimistic `uploading` rows for files that
 * have been submitted but do not yet have a backend record. Backend rows win
 * when a filename is represented in both.
 */
export function mergeDocumentStatuses(
  backendRows: KnowledgeBaseDocumentStatus[],
  optimisticFilenames: string[]
): KnowledgeBaseDocumentStatus[] {
  const representedFilenames = new Set(
    backendRows.map((row) => row.filename.split("/").pop() || row.filename)
  )
  const optimisticRows: KnowledgeBaseDocumentStatus[] = []
  const seen = new Set<string>()
  for (const rawName of optimisticFilenames) {
    const filename = rawName.trim()
    if (!filename || seen.has(filename) || representedFilenames.has(filename)) {
      continue
    }
    seen.add(filename)
    optimisticRows.push({ filename, status: "uploading", can_delete: false })
  }
  return [...optimisticRows, ...backendRows]
}

export interface CollectionDocumentSource {
  document_names?: string[]
  document_metadata?: CollectionDocumentInfo[]
}

export interface CollectionTranslator {
  (key: string, vars?: Record<string, string | number>): string
}

function normalizeOptionalIdentifier(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined
  }

  const normalizedValue = value.trim()
  return normalizedValue || undefined
}

function getDocumentIdentityKey(document: CollectionDocumentInfo): string {
  return JSON.stringify([
    document.filename,
    normalizeOptionalIdentifier(document.file_id) ?? null,
    normalizeOptionalIdentifier(document.doc_id) ?? null,
  ])
}

export function getCollectionDocuments(collectionInfo: CollectionDocumentSource | null): CollectionDocumentInfo[] {
  if (!collectionInfo) {
    return []
  }

  const representedFilenames = new Set<string>()
  const seenDocumentKeys = new Set<string>()
  const documents: CollectionDocumentInfo[] = []

  if (Array.isArray(collectionInfo.document_metadata) && collectionInfo.document_metadata.length > 0) {
    for (const document of collectionInfo.document_metadata) {
      if (typeof document.filename !== "string") {
        continue
      }
      const normalizedFilename = document.filename.trim()
      if (!normalizedFilename) {
        continue
      }

      const normalizedDocument = {
        ...document,
        filename: normalizedFilename,
        file_id: normalizeOptionalIdentifier(document.file_id),
        doc_id: normalizeOptionalIdentifier(document.doc_id),
      }
      const documentKey = getDocumentIdentityKey(normalizedDocument)
      if (seenDocumentKeys.has(documentKey)) {
        continue
      }

      seenDocumentKeys.add(documentKey)
      representedFilenames.add(normalizedFilename)
      documents.push(normalizedDocument)
    }
  }

  if (!Array.isArray(collectionInfo.document_names)) {
    return documents
  }

  for (const filename of collectionInfo.document_names) {
    if (typeof filename !== "string") {
      continue
    }
    const normalizedFilename = filename.trim()
    if (!normalizedFilename || representedFilenames.has(normalizedFilename)) {
      continue
    }

    representedFilenames.add(normalizedFilename)
    documents.push({ filename: normalizedFilename })
  }

  return documents
}

export function buildDeleteDocumentUrl(apiUrl: string, collectionName: string, document: CollectionDocumentInfo): string {
  const baseUrl = `${apiUrl}/api/kb/collections/${encodeURIComponent(collectionName)}/documents/${encodeURIComponent(document.filename)}`
  const query = new URLSearchParams()

  if (document.file_id) {
    query.set("file_id", document.file_id)
  } else if (document.doc_id) {
    query.set("doc_id", document.doc_id)
  }

  const queryString = query.toString()
  return queryString ? `${baseUrl}?${queryString}` : baseUrl
}

export function getDeleteErrorMessage(result: unknown, fallbackMessage: string): string {
  if (!result || typeof result !== "object") {
    return fallbackMessage
  }

  const response = result as { detail?: unknown; message?: unknown; errors?: unknown }
  if (typeof response.detail === "string" && response.detail) {
    return response.detail
  }
  if (typeof response.message === "string" && response.message) {
    return response.message
  }
  if (Array.isArray(response.errors)) {
    const firstError = response.errors.find((error): error is string => typeof error === "string" && error.length > 0)
    if (firstError) {
      return firstError
    }
  }

  return fallbackMessage
}
