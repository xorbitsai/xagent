import { afterEach, describe, expect, it, vi } from "vitest"

import {
  uploadDeferredPublicChatFiles,
  uploadPublicChatFile,
} from "./public-chat-file-upload"

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("uploadPublicChatFile", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("rejects backend HTTP failures instead of silently accepting them", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "File is too large" }), {
        status: 413,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })).rejects.toThrow("File is too large")

    const [, request] = fetchMock.mock.calls[0]
    expect(new Headers(request?.headers).get("Authorization")).toBe(
      "Bearer guest-token",
    )
    const body = request?.body as FormData
    expect(body.get("file")).toBe(file)
    expect(body.get("task_type")).toBe("task")
    expect(body.get("task_id")).toBe("42")
  })

  it("returns normalized file metadata for successful uploads", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ success: true, file_id: "file-1" }),
    )
    const file = new File(["trip"], "trip.txt", { type: "text/plain" })

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      file,
      taskType: "task",
      fallbackError: "Upload failed",
    })).resolves.toEqual({
      file_id: "file-1",
      name: "trip.txt",
      size: 4,
      type: "text/plain",
    })
  })
})

describe("uploadDeferredPublicChatFiles", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  const uploadContext = {}
  const options = {
    url: "http://api.local/api/share/files/upload",
    accessToken: "guest-token",
    taskType: "task",
    fallbackError: "Upload failed",
    uploadContext,
  }

  it("admits uploads FIFO with at most three active requests through drainage", async () => {
    const requests = Array.from({ length: 7 }, () => deferred<Response>())
    const admitted: string[] = []
    let active = 0
    let maxActive = 0
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      const file = (init?.body as FormData).get("file") as File
      const request = requests[admitted.length]
      admitted.push(file.name)
      active += 1
      maxActive = Math.max(maxActive, active)
      try {
        return await request.promise
      } finally {
        active -= 1
      }
    })
    const files = Array.from(
      { length: 7 },
      (_, index) => new File([`${index}`], `file-${index}.txt`),
    )

    const upload = uploadDeferredPublicChatFiles(files, options)

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    expect(admitted).toEqual(["file-0.txt", "file-1.txt", "file-2.txt"])
    for (let index = 0; index < requests.length; index += 1) {
      requests[index].resolve(jsonResponse({
        success: true,
        file_id: `id-${index}`,
      }))
      if (index < requests.length - 3) {
        await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(index + 4))
      }
      expect(active).toBeLessThanOrEqual(3)
    }

    await expect(upload).resolves.toEqual(files.map((file, index) => ({
      file_id: `id-${index}`,
      name: file.name,
      size: file.size,
      type: file.type,
    })))
    expect(admitted).toEqual(files.map((file) => file.name))
    expect(maxActive).toBe(3)
  })

  it("settles every scheduled upload and preserves successful ids after partial failure", async () => {
    const requests = Array.from({ length: 4 }, () => deferred<Response>())
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      () => requests[fetchMock.mock.calls.length - 1].promise,
    )
    const files = Array.from(
      { length: 4 },
      (_, index) => new File([`${index}`], `file-${index}.txt`),
    )
    let settled = false

    const upload = uploadDeferredPublicChatFiles(files, options).finally(() => {
      settled = true
    })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    requests[0].resolve(jsonResponse({ success: true, file_id: "id-0" }))
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4))
    requests[1].resolve(jsonResponse({ detail: "storage unavailable" }, 503))
    requests[2].resolve(jsonResponse({ success: true, file_id: "id-2" }))
    await Promise.resolve()
    expect(settled).toBe(false)
    requests[3].resolve(jsonResponse({ success: true, file_id: "id-3" }))

    await expect(upload).rejects.toThrow("storage unavailable")
    // Cache state is context-scoped rather than published as a bare File id.
    expect(files.every(file => !("file_id" in file))).toBe(true)
  })

  it("skips files that succeeded when the same selection is retried", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "id-success" }))
      .mockResolvedValueOnce(jsonResponse({ detail: "retry me" }, 503))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "id-retry" }))
    const successful = new File(["ok"], "successful.txt")
    const retry = new File(["retry"], "retry.txt")

    await expect(
      uploadDeferredPublicChatFiles([successful, retry], options),
    ).rejects.toThrow("retry me")
    await expect(
      uploadDeferredPublicChatFiles([successful, retry], options),
    ).resolves.toEqual([
      expect.objectContaining({ file_id: "id-success", name: "successful.txt" }),
      expect.objectContaining({ file_id: "id-retry", name: "retry.txt" }),
    ])

    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(fetchMock.mock.calls.map(([, init]) =>
      ((init?.body as FormData).get("file") as File).name)).toEqual([
      "successful.txt",
      "retry.txt",
      "retry.txt",
    ])
  })

  it("does not reuse ids across task or taskless upload scopes", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "task-a" }))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "task-b" }))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "taskless" }))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "task-bound" }))
    const taskFile = new File(["same"], "task.txt")
    const bindingFile = new File(["same"], "binding.txt")

    await expect(uploadDeferredPublicChatFiles([taskFile], {
      ...options,
      taskId: 1,
    })).resolves.toEqual([expect.objectContaining({ file_id: "task-a" })])
    await expect(uploadDeferredPublicChatFiles([taskFile], {
      ...options,
      taskId: 2,
    })).resolves.toEqual([expect.objectContaining({ file_id: "task-b" })])
    await expect(uploadDeferredPublicChatFiles([bindingFile], options))
      .resolves.toEqual([expect.objectContaining({ file_id: "taskless" })])
    await expect(uploadDeferredPublicChatFiles([bindingFile], {
      ...options,
      taskId: 3,
    })).resolves.toEqual([expect.objectContaining({ file_id: "task-bound" })])

    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

  it("posts the same taskless File again for a new conversation context", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "conversation-a" }))
      .mockResolvedValueOnce(jsonResponse({ success: true, file_id: "conversation-b" }))
    const file = new File(["same"], "same.txt")
    const conversationA = {}
    const conversationB = {}

    await expect(uploadDeferredPublicChatFiles([file], {
      ...options,
      uploadContext: conversationA,
    })).resolves.toEqual([expect.objectContaining({ file_id: "conversation-a" })])
    await expect(uploadDeferredPublicChatFiles([file], {
      ...options,
      uploadContext: conversationA,
    })).resolves.toEqual([expect.objectContaining({ file_id: "conversation-a" })])
    await expect(uploadDeferredPublicChatFiles([file], {
      ...options,
      uploadContext: conversationB,
    })).resolves.toEqual([expect.objectContaining({ file_id: "conversation-b" })])

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("does not let a late obsolete attempt publish over the current attempt", async () => {
    const first = deferred<Response>()
    const second = deferred<Response>()
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
    const file = new File(["same"], "late.txt")

    const staleUpload = uploadDeferredPublicChatFiles([file], options)
    const currentUpload = uploadDeferredPublicChatFiles([file], options)
    second.resolve(jsonResponse({ success: true, file_id: "current" }))
    await expect(currentUpload).resolves.toEqual([
      expect.objectContaining({ file_id: "current" }),
    ])
    first.resolve(jsonResponse({ success: true, file_id: "stale" }))
    await expect(staleUpload).resolves.toEqual([
      expect.objectContaining({ file_id: "stale" }),
    ])

    await expect(uploadDeferredPublicChatFiles([file], options)).resolves.toEqual([
      expect.objectContaining({ file_id: "current" }),
    ])
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("removes an aborted queued upload without fetching it", async () => {
    const active = Array.from({ length: 3 }, () => deferred<Response>())
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      () => active[fetchMock.mock.calls.length - 1].promise,
    )
    const occupied = uploadDeferredPublicChatFiles(
      Array.from({ length: 3 }, (_, index) => new File(["x"], `active-${index}.txt`)),
      options,
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    const controller = new AbortController()
    const queued = uploadDeferredPublicChatFiles(
      [new File(["queued"], "queued.txt")],
      { ...options, signal: controller.signal },
    )

    controller.abort()
    await expect(queued).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock).toHaveBeenCalledTimes(3)

    active.forEach((request, index) => request.resolve(
      jsonResponse({ success: true, file_id: `active-${index}` }),
    ))
    await occupied
  })

  it("aborts an active upload and drains its slot", async () => {
    const controller = new AbortController()
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((_url, init) => (
      new Promise<Response>((resolve, reject) => {
        const file = (init?.body as FormData).get("file") as File
        if (file.name === "replacement.txt") {
          resolve(jsonResponse({ success: true, file_id: "replacement" }))
          return
        }
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true })
      })
    ))
    const occupied = uploadDeferredPublicChatFiles(
      [
        new File(["x"], "cancel.txt"),
        new File(["x"], "stall-1.txt"),
        new File(["x"], "stall-2.txt"),
      ],
      { ...options, signal: controller.signal },
    )
    const replacement = uploadDeferredPublicChatFiles(
      [new File(["x"], "replacement.txt")],
      options,
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))

    controller.abort()
    await expect(occupied).rejects.toMatchObject({ name: "AbortError" })
    await expect(replacement).resolves.toEqual([
      expect.objectContaining({ file_id: "replacement" }),
    ])
    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

})
