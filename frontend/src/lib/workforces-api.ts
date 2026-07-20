"use client"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import type {
  WorkforceAgentOption,
  WorkforceAgentExecution,
  WorkforceArchiveResponse,
  WorkforceCanvasResponse,
  WorkforceCreatePayload,
  WorkforceDetail,
  WorkforceListResponse,
  WorkforcePromptCreatePayload,
  WorkforceRunHistoryItem,
  WorkforceRunHistoryResponse,
  WorkforceRunPayload,
  WorkforceRunResponse,
  WorkforceUpdatePayload,
  WorkforceWorker,
  WorkforceWorkerPayload,
  WorkforceWorkerUpdatePayload,
} from "@/types/workforce"

function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg)
        }
        return null
      })
      .filter(Boolean)
    if (messages.length > 0) {
      return messages.join("; ")
    }
  }
  return fallback
}

async function parseApiError(response: Response, fallback: string): Promise<Error> {
  try {
    const data = await response.json()
    return new Error(formatApiDetail(data?.detail, fallback))
  } catch {
    return new Error(fallback)
  }
}

function jsonHeaders(): HeadersInit {
  return {
    "Content-Type": "application/json",
  }
}

export async function listWorkforces(params?: {
  page?: number
  size?: number
  search?: string
  status?: string
}): Promise<WorkforceListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.size) searchParams.set("size", String(params.size))
  if (params?.search) searchParams.set("search", params.search)
  if (params?.status) searchParams.set("status", params.status)

  const suffix = searchParams.toString() ? `?${searchParams.toString()}` : ""
  const response = await apiRequest(`${getApiUrl()}/api/workforces${suffix}`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforces")
  }
  return response.json()
}

export async function getWorkforce(workforceId: number | string): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce")
  }
  return response.json()
}

export async function createWorkforce(payload: WorkforceCreatePayload): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to create workforce")
  }
  return response.json()
}

export async function createWorkforceFromPrompt(
  payload: WorkforcePromptCreatePayload,
): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/from-prompt`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to create workforce")
  }
  return response.json()
}

export async function updateWorkforce(
  workforceId: number | string,
  payload: WorkforceUpdatePayload,
): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}`, {
    method: "PATCH",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update workforce")
  }
  return response.json()
}

export async function archiveWorkforce(
  workforceId: number | string,
): Promise<WorkforceArchiveResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}`, {
    method: "DELETE",
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to archive workforce")
  }
  return response.json()
}

export async function publishWorkforce(
  workforceId: number | string,
): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/publish`, {
    method: "POST",
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to publish workforce")
  }
  return response.json()
}

export async function unpublishWorkforce(
  workforceId: number | string,
): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/unpublish`, {
    method: "POST",
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to unpublish workforce")
  }
  return response.json()
}

export async function addWorkforceAgent(
  workforceId: number | string,
  payload: WorkforceWorkerPayload,
): Promise<WorkforceWorker> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/agents`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to add workforce worker")
  }
  return response.json()
}

export async function updateWorkforceAgent(
  workforceId: number | string,
  memberId: number | string,
  payload: WorkforceWorkerUpdatePayload,
): Promise<WorkforceWorker> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/agents/${memberId}`,
    {
      method: "PATCH",
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update workforce worker")
  }
  return response.json()
}

export async function removeWorkforceAgent(
  workforceId: number | string,
  memberId: number | string,
): Promise<void> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/agents/${memberId}`,
    {
      method: "DELETE",
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to remove workforce worker")
  }
}

export async function runWorkforce(
  workforceId: number | string,
  payload: WorkforceRunPayload | string,
): Promise<WorkforceRunResponse> {
  const body = typeof payload === "string" ? { message: payload } : payload
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/runs`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to run workforce")
  }
  return response.json()
}

export async function getWorkforceAgentExecution(
  workforceId: number | string,
  taskId: number | string,
  workerTaskId: string,
): Promise<WorkforceAgentExecution> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/runs/${taskId}/agent-executions/${encodeURIComponent(workerTaskId)}`,
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load Agent execution")
  }
  return response.json()
}

export async function listWorkforceRuns(
  workforceId: number | string,
  params?: {
    page?: number
    size?: number
  },
): Promise<WorkforceRunHistoryResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.size) searchParams.set("size", String(params.size))

  const suffix = searchParams.toString() ? `?${searchParams.toString()}` : ""
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/runs${suffix}`,
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce runs")
  }
  return response.json()
}

export async function getWorkforceRun(
  workforceId: number | string,
  runId: number | string,
): Promise<WorkforceRunHistoryItem> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/runs/${runId}`,
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce run")
  }
  return response.json()
}

export async function getWorkforceCanvas(
  workforceId: number | string,
): Promise<WorkforceCanvasResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/canvas`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce canvas")
  }
  return response.json()
}

export async function listAgentOptions(): Promise<WorkforceAgentOption[]> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/agent-options`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load agents")
  }
  return response.json()
}
