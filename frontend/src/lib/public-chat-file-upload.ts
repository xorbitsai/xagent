import { classifyUploadError, isJsonRecord, parseApiResponse } from "@/lib/api-wrapper"
import type { ClientErrorCode } from "@/lib/client-errors"
import { normalizeUploadFileIds } from "@/lib/upload-file-ids"

export interface PublicChatUploadedFile {
  file_id: string
  name?: string
  size?: number
  type?: string
}

interface UploadPublicChatFileOptions {
  url: string
  accessToken: string
  file: File
  taskType: string
  taskId?: number | string | null
  fallbackError: string
  formatError?: (code: ClientErrorCode) => string
}

export class PublicChatUploadError extends Error {
  readonly errorCode: ClientErrorCode

  constructor(message: string, errorCode: ClientErrorCode) {
    super(message)
    this.name = "PublicChatUploadError"
    this.errorCode = errorCode
  }
}

export async function uploadPublicChatFile({
  url,
  accessToken,
  file,
  taskType,
  taskId,
  fallbackError,
  formatError,
}: UploadPublicChatFileOptions): Promise<PublicChatUploadedFile> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("task_type", taskType)
  if (taskId != null) {
    formData.append("task_id", taskId.toString())
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Authorization": `Bearer ${accessToken}` },
    body: formData,
  })
  const parsed = await parseApiResponse(response)
  const data = isJsonRecord(parsed.data) ? parsed.data : null
  const fileId = normalizeUploadFileIds([data?.file_id], 1)?.[0] ?? null

  if (!response.ok || data?.success !== true || !fileId) {
    const classified = classifyUploadError(response, parsed)
    const message = formatError?.(classified.errorCode)
      || (classified.errorCode === "upload_failed" ? fallbackError : classified.message)
    throw new PublicChatUploadError(message, classified.errorCode)
  }

  return {
    file_id: fileId,
    name: file.name,
    size: file.size,
    type: file.type,
  }
}
