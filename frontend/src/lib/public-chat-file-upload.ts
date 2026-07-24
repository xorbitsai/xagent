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
}

interface PublicChatUploadResponse {
  success?: boolean
  file_id?: unknown
  detail?: unknown
  message?: unknown
}

export async function uploadPublicChatFile({
  url,
  accessToken,
  file,
  taskType,
  taskId,
  fallbackError,
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
  const data = await response.json().catch(() => null) as PublicChatUploadResponse | null
  const fileId = typeof data?.file_id === "string" ? data.file_id : null

  if (!response.ok || data?.success !== true || !fileId) {
    const backendMessage = typeof data?.detail === "string"
      ? data.detail
      : typeof data?.message === "string"
        ? data.message
        : null
    throw new Error(backendMessage || fallbackError)
  }

  return {
    file_id: fileId,
    name: file.name,
    size: file.size,
    type: file.type,
  }
}
