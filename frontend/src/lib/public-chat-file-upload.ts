export interface PublicChatUploadedFile {
  file_id: string
  name?: string
  size?: number
  type?: string
}

export interface PublicChatUploadFailure {
  name: string
  error: string
}

interface UploadPublicChatFilesOptions {
  url: string
  accessToken: string
  files: File[]
  taskType: string
  taskId?: number | string | null
  fallbackError: string
  signal?: AbortSignal
  onFailures?: (failures: PublicChatUploadFailure[]) => void
}

interface PublicChatUploadResponse {
  success?: boolean
  files?: unknown
  detail?: unknown
  message?: unknown
}

interface IndexedFile {
  file: File
  index: number
}

interface NormalizedOutcome {
  index: number
  uploaded?: PublicChatUploadedFile
  failure?: PublicChatUploadFailure
}

/**
 * Keep public multipart requests comfortably below nginx's fixed 500M body
 * ceiling. The 450 MiB payload budget leaves at least 50 MiB for multipart
 * boundaries and headers while retaining the ordinary one-request path.
 */
const PUBLIC_UPLOAD_MULTIPART_BYTE_BUDGET = 450 * 1024 * 1024
const PUBLIC_UPLOAD_MAX_FILES_PER_REQUEST = 20

function uploadErrorMessage(
  data: PublicChatUploadResponse | null,
  fallbackError: string,
): string {
  return typeof data?.detail === "string"
    ? data.detail
    : typeof data?.message === "string"
      ? data.message
      : fallbackError
}

function partitionTaskBoundFiles(files: IndexedFile[]): IndexedFile[][] {
  const chunks: IndexedFile[][] = []
  let current: IndexedFile[] = []
  let currentBytes = 0

  files.forEach(indexedFile => {
    const exceedsCurrentChunk = current.length > 0 && (
      current.length >= PUBLIC_UPLOAD_MAX_FILES_PER_REQUEST
      || currentBytes + indexedFile.file.size > PUBLIC_UPLOAD_MULTIPART_BYTE_BUDGET
    )
    if (exceedsCurrentChunk) {
      chunks.push(current)
      current = []
      currentBytes = 0
    }
    current.push(indexedFile)
    currentBytes += indexedFile.file.size
  })

  if (current.length > 0) chunks.push(current)
  return chunks
}

function normalizeChunkOutcomes(
  rawFiles: unknown,
  chunk: IndexedFile[],
  fallbackError: string,
): NormalizedOutcome[] {
  if (!Array.isArray(rawFiles) || rawFiles.length !== chunk.length) {
    throw new Error(fallbackError)
  }

  const records = rawFiles.map(item => {
    if (typeof item !== "object" || item === null) throw new Error(fallbackError)
    return item as Record<string, unknown>
  })
  const hasExplicitCorrelation = records.some(record => "source_index" in record)
  const usedIndexes = new Set<number>()

  return records.map((record, responseIndex) => {
    const sourceIndex = hasExplicitCorrelation ? record.source_index : responseIndex
    if (
      typeof sourceIndex !== "number"
      || !Number.isInteger(sourceIndex)
      || sourceIndex < 0
      || sourceIndex >= chunk.length
      || usedIndexes.has(sourceIndex)
    ) {
      throw new Error(fallbackError)
    }
    usedIndexes.add(sourceIndex)
    const source = chunk[sourceIndex]

    if (record.success === false) {
      if (typeof record.error !== "string" || !record.error) {
        throw new Error(fallbackError)
      }
      return {
        index: source.index,
        failure: {
          name: typeof record.filename === "string" && record.filename
            ? record.filename
            : source.file.name,
          error: record.error,
        },
      }
    }

    if (typeof record.file_id !== "string" || !record.file_id) {
      throw new Error(fallbackError)
    }
    return {
      index: source.index,
      uploaded: {
        file_id: record.file_id,
        name: typeof record.filename === "string" ? record.filename : source.file.name,
        size: typeof record.file_size === "number" ? record.file_size : source.file.size,
        type: typeof record.mime_type === "string" ? record.mime_type : source.file.type,
      },
    }
  })
}

/**
 * Upload public-chat attachments and retain every successful per-file outcome.
 * Task-bound sets are sent sequentially in bounded multipart chunks; task-less
 * workforce openings intentionally stay as one request so their existing
 * ten-file guard remains authoritative.
 */
export async function uploadPublicChatFiles({
  url,
  accessToken,
  files,
  taskType,
  taskId,
  fallbackError,
  signal,
  onFailures,
}: UploadPublicChatFilesOptions): Promise<PublicChatUploadedFile[]> {
  if (files.length === 0) return []

  const indexedFiles = files.map((file, index) => ({ file, index }))
  const locallyRejected = taskId == null
    ? []
    : indexedFiles.filter(({ file }) => file.size > PUBLIC_UPLOAD_MULTIPART_BYTE_BUDGET)
  const eligibleFiles = taskId == null
    ? indexedFiles
    : indexedFiles.filter(({ file }) => file.size <= PUBLIC_UPLOAD_MULTIPART_BYTE_BUDGET)
  const chunks = taskId == null ? [eligibleFiles] : partitionTaskBoundFiles(eligibleFiles)
  const outcomes: NormalizedOutcome[] = locallyRejected.map(({ file, index }) => ({
    index,
    failure: {
      name: file.name,
      error: "File exceeds the public upload request limit",
    },
  }))

  for (const chunk of chunks) {
    signal?.throwIfAborted()
    const formData = new FormData()
    chunk.forEach(({ file }) => formData.append("files", file))
    formData.append("task_type", taskType)
    if (taskId != null) formData.append("task_id", taskId.toString())

    const response = await fetch(url, {
      method: "POST",
      headers: { "Authorization": `Bearer ${accessToken}` },
      body: formData,
      signal,
    })
    const data = await response.json().catch(() => null) as PublicChatUploadResponse | null
    if (!response.ok || data?.success !== true) {
      throw new Error(uploadErrorMessage(data, fallbackError))
    }
    outcomes.push(...normalizeChunkOutcomes(data.files, chunk, fallbackError))
  }

  const ordered = outcomes.sort((left, right) => left.index - right.index)
  const failures = ordered.flatMap(outcome => outcome.failure ? [outcome.failure] : [])
  if (failures.length > 0) onFailures?.(failures)

  const uploaded = ordered.flatMap(outcome => outcome.uploaded ? [outcome.uploaded] : [])
  if (uploaded.length === 0 && failures.length > 0) {
    throw new Error(`${failures[0].name}: ${failures[0].error}`)
  }
  return uploaded
}
