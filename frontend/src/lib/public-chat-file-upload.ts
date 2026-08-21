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
  signal?: AbortSignal
}

interface UploadDeferredPublicChatFilesOptions extends Omit<UploadPublicChatFileOptions, "file"> {
  /**
   * Stable identity for the credential/session generation that owns this
   * upload. Object identity scopes cached ids without retaining raw tokens.
   */
  uploadContext: object
}

type ScheduledUpload = {
  run: (signal: AbortSignal) => Promise<PublicChatUploadedFile>
  resolve: (file: PublicChatUploadedFile) => void
  reject: (reason: unknown) => void
  signal?: AbortSignal
  abortQueued: () => void
}

interface CachedUploadState {
  currentAttempt?: symbol
  uploaded?: PublicChatUploadedFile
}

const MAX_ACTIVE_DEFERRED_UPLOADS = 3
const deferredUploadQueue: ScheduledUpload[] = []
const deferredUploadCache = new WeakMap<
  File,
  WeakMap<object, Map<string, CachedUploadState>>
>()
let activeDeferredUploads = 0

const abortReason = (signal: AbortSignal): unknown =>
  signal.reason ?? new DOMException("The operation was aborted.", "AbortError")

/** Reject an async transport or parsing operation as soon as its signal fires. */
function raceWithAbort<T>(operation: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return operation
  if (signal.aborted) return Promise.reject(abortReason(signal))

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortReason(signal))
    signal.addEventListener("abort", onAbort, { once: true })
    operation.then(
      (value) => {
        signal.removeEventListener("abort", onAbort)
        resolve(value)
      },
      (error) => {
        signal.removeEventListener("abort", onAbort)
        reject(error)
      },
    )
  })
}

/**
 * Drain the module-wide browser queue in admission order. Every settled or
 * owner-aborted job releases its global slot for the next queued upload.
 */
function drainDeferredUploadQueue() {
  while (
    activeDeferredUploads < MAX_ACTIVE_DEFERRED_UPLOADS
    && deferredUploadQueue.length > 0
  ) {
    const scheduled = deferredUploadQueue.shift()
    if (!scheduled) return

    scheduled.signal?.removeEventListener("abort", scheduled.abortQueued)
    if (scheduled.signal?.aborted) {
      scheduled.reject(abortReason(scheduled.signal))
      continue
    }

    activeDeferredUploads += 1
    const controller = new AbortController()
    const abortActive = () => controller.abort(abortReason(scheduled.signal as AbortSignal))
    scheduled.signal?.addEventListener("abort", abortActive, { once: true })

    void scheduled.run(controller.signal)
      .then(scheduled.resolve, scheduled.reject)
      .finally(() => {
        scheduled.signal?.removeEventListener("abort", abortActive)
        activeDeferredUploads -= 1
        drainDeferredUploadQueue()
      })
  }
}

function scheduleDeferredUpload(
  run: (signal: AbortSignal) => Promise<PublicChatUploadedFile>,
  signal: AbortSignal | undefined,
): Promise<PublicChatUploadedFile> {
  if (signal?.aborted) return Promise.reject(abortReason(signal))

  const scheduled = new Promise<PublicChatUploadedFile>((resolve, reject) => {
    const job: ScheduledUpload = {
      run,
      resolve,
      reject,
      signal,
      abortQueued: () => {
        const index = deferredUploadQueue.indexOf(job)
        if (index < 0) return
        deferredUploadQueue.splice(index, 1)
        signal?.removeEventListener("abort", job.abortQueued)
        reject(abortReason(signal as AbortSignal))
      },
    }
    signal?.addEventListener("abort", job.abortQueued, { once: true })
    deferredUploadQueue.push(job)
  })
  drainDeferredUploadQueue()
  return scheduled
}

interface PublicChatUploadResponse {
  success?: boolean
  file_id?: unknown
  detail?: unknown
  message?: unknown
}

/** Upload one public-chat file without changing the existing request contract. */
export async function uploadPublicChatFile({
  url,
  accessToken,
  file,
  taskType,
  taskId,
  fallbackError,
  signal,
}: UploadPublicChatFileOptions): Promise<PublicChatUploadedFile> {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("task_type", taskType)
  if (taskId != null) {
    formData.append("task_id", taskId.toString())
  }

  const response = await raceWithAbort(fetch(url, {
    method: "POST",
    headers: { "Authorization": `Bearer ${accessToken}` },
    body: formData,
    signal,
  }), signal)
  const data = await raceWithAbort(
    response.json().catch(() => null) as Promise<PublicChatUploadResponse | null>,
    signal,
  )
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

const uploadScopeKey = ({
  url,
  taskType,
  taskId,
}: UploadDeferredPublicChatFilesOptions): string =>
  `${url}\u0000${taskType}\u0000${taskId == null ? "taskless" : `${typeof taskId}:${taskId}`}`

function getCachedUploadState(
  file: File,
  uploadContext: object,
  scopeKey: string,
): CachedUploadState {
  let contexts = deferredUploadCache.get(file)
  if (!contexts) {
    contexts = new WeakMap()
    deferredUploadCache.set(file, contexts)
  }
  let scopes = contexts.get(uploadContext)
  if (!scopes) {
    scopes = new Map()
    contexts.set(uploadContext, scopes)
  }
  let state = scopes.get(scopeKey)
  if (!state) {
    state = {}
    scopes.set(scopeKey, state)
  }
  return state
}

/**
 * Upload a deferred selection with bounded FIFO admission.
 *
 * Successful ids are retained only for the same caller-owned credential,
 * endpoint, task binding, and task type. Each invocation supersedes older
 * in-flight attempts in that scope, preventing late results from publishing
 * over the current attempt while preserving successful partial retries.
 */
export async function uploadDeferredPublicChatFiles(
  files: File[],
  options: UploadDeferredPublicChatFilesOptions,
): Promise<PublicChatUploadedFile[]> {
  const attempt = Symbol("deferred-public-upload-attempt")
  const scopeKey = uploadScopeKey(options)
  const uploads = files.map((file) => {
    const state = getCachedUploadState(file, options.uploadContext, scopeKey)
    if (state.uploaded) return Promise.resolve(state.uploaded)

    state.currentAttempt = attempt
    return scheduleDeferredUpload(async (signal) => {
      try {
        const uploaded = await uploadPublicChatFile({ ...options, file, signal })
        if (state.currentAttempt === attempt) {
          state.uploaded = uploaded
          state.currentAttempt = undefined
        }
        return uploaded
      } catch (error) {
        if (state.currentAttempt === attempt) state.currentAttempt = undefined
        throw error
      }
    }, options.signal)
  })
  const settled = await Promise.allSettled(uploads)
  const failed = settled.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  )

  if (failed) throw failed.reason
  return settled.map(
    (result) => (result as PromiseFulfilledResult<PublicChatUploadedFile>).value,
  )
}
