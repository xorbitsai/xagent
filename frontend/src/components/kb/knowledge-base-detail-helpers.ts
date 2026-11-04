export interface CollectionDocumentInfo {
  filename: string
  file_id?: string
  doc_id?: string
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
