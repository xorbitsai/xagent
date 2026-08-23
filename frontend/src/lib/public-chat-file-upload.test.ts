import { afterEach, describe, expect, it, vi } from "vitest"

import { uploadPublicChatFiles } from "./public-chat-file-upload"

describe("uploadPublicChatFiles", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("sends all files in one authenticated multipart request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "File is too large" }), {
        status: 413,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const first = new File(["first"], "first.txt", { type: "text/plain" })
    const second = new File(["second"], "second.txt", { type: "text/plain" })

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [first, second],
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })).rejects.toThrow("File is too large")

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [, request] = fetchMock.mock.calls[0]
    expect(new Headers(request?.headers).get("Authorization")).toBe(
      "Bearer guest-token",
    )
    const body = request?.body as FormData
    expect(body.getAll("files")).toEqual([first, second])
    expect(body.get("file")).toBeNull()
    expect(body.get("task_type")).toBe("task")
    expect(body.get("task_id")).toBe("42")
  })

  it("forwards the caller-owned abort signal", async () => {
    const controller = new AbortController()
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [{ file_id: "file-1" }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [new File(["first"], "first.txt")],
      taskType: "task",
      fallbackError: "Upload failed",
      signal: controller.signal,
    })

    expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal)
  })

  it("normalizes successful batch metadata", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [
          {
            file_id: "file-1",
            filename: "first.txt",
            file_size: 5,
            mime_type: "text/plain",
          },
          {
            file_id: "file-2",
            filename: "second.txt",
            file_size: 6,
            mime_type: "text/plain",
          },
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const first = new File(["first"], "first.txt", { type: "text/plain" })
    const second = new File(["second"], "second.txt", { type: "text/plain" })

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [first, second],
      taskType: "task",
      fallbackError: "Upload failed",
    })).resolves.toEqual([
      { file_id: "file-1", name: "first.txt", size: 5, type: "text/plain" },
      { file_id: "file-2", name: "second.txt", size: 6, type: "text/plain" },
    ])
  })

  it("preserves successful siblings without filtered-array positional drift", async () => {
    const onFailures = vi.fn()
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [
          {
            success: false,
            source_index: 0,
            filename: "oversized.txt",
            error: "File size exceeds maximum limit of 100MB",
          },
          {
            success: true,
            source_index: 1,
            file_id: "file-1",
            filename: "second.txt",
            file_size: 6,
            mime_type: "text/plain",
          },
        ],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [new File(["large"], "oversized.txt"), new File(["second"], "second.txt")],
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
      onFailures,
    })).resolves.toEqual([
      { file_id: "file-1", name: "second.txt", size: 6, type: "text/plain" },
    ])
    expect(onFailures).toHaveBeenCalledWith([
      {
        name: "oversized.txt",
        error: "File size exceeds maximum limit of 100MB",
      },
    ])
  })

  it.each([
    { file_id: 123 },
    { file_id: null },
    { file_id: "" },
  ])("rejects a malformed file_id: $file_id", async malformed => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [{ success: true, source_index: 0, ...malformed }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [new File(["first"], "first.txt")],
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })).rejects.toThrow("Upload failed")
  })

  it("partitions task-bound files by the conservative multipart byte budget", async () => {
    const files = ["first", "second", "third"].map(name => {
      const file = new File([name], `${name}.txt`)
      Object.defineProperty(file, "size", { value: 200 * 1024 * 1024 })
      return file
    })
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (_url, request) => {
        const chunk = (request?.body as FormData).getAll("files") as File[]
        return new Response(JSON.stringify({
          success: true,
          files: chunk.map((file, source_index) => ({
            success: true,
            source_index,
            file_id: `${file.name}-id`,
          })),
        }), { status: 200, headers: { "Content-Type": "application/json" } })
      },
    )

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })).resolves.toEqual(files.map(file => ({
      file_id: `${file.name}-id`,
      name: file.name,
      size: file.size,
      type: file.type,
    })))

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect((fetchMock.mock.calls[0][1]?.body as FormData).getAll("files")).toEqual(files.slice(0, 2))
    expect((fetchMock.mock.calls[1][1]?.body as FormData).getAll("files")).toEqual(files.slice(2))
  })

  it("does not send a task-bound file that exceeds the multipart budget", async () => {
    const oversized = new File(["large"], "oversized.txt")
    Object.defineProperty(oversized, "size", { value: 451 * 1024 * 1024 })
    const valid = new File(["valid"], "valid.txt")
    const onFailures = vi.fn()
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [{ success: true, source_index: 0, file_id: "valid-id" }],
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    )

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [oversized, valid],
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
      onFailures,
    })).resolves.toEqual([
      { file_id: "valid-id", name: "valid.txt", size: 5, type: "" },
    ])

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect((fetchMock.mock.calls[0][1]?.body as FormData).getAll("files")).toEqual([valid])
    expect(onFailures).toHaveBeenCalledWith([{
      name: "oversized.txt",
      error: "File exceeds the public upload request limit",
    }])
  })

  it("bounds task-bound chunks by file count", async () => {
    const files = Array.from({ length: 21 }, (_, index) => new File(["x"], `${index}.txt`))
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (_url, request) => {
        const chunk = (request?.body as FormData).getAll("files") as File[]
        return new Response(JSON.stringify({
          success: true,
          files: chunk.map((file, source_index) => ({
            success: true,
            source_index,
            file_id: `${file.name}-id`,
          })),
        }), { status: 200, headers: { "Content-Type": "application/json" } })
      },
    )

    await uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect((fetchMock.mock.calls[0][1]?.body as FormData).getAll("files")).toHaveLength(20)
    expect((fetchMock.mock.calls[1][1]?.body as FormData).getAll("files")).toHaveLength(1)
  })

  it("stops before the next chunk when the caller aborts", async () => {
    const controller = new AbortController()
    const files = ["first", "second"].map(name => {
      const file = new File([name], `${name}.txt`)
      Object.defineProperty(file, "size", { value: 300 * 1024 * 1024 })
      return file
    })
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      controller.abort()
      return new Response(JSON.stringify({
        success: true,
        files: [{ success: true, source_index: 0, file_id: "first-id" }],
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    })

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files,
      taskType: "task",
      taskId: 42,
      fallbackError: "Upload failed",
      signal: controller.signal,
    })).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("rejects an incomplete batch response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        success: true,
        files: [{ file_id: "file-1" }],
        message: "Successfully uploaded 2 files",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [new File(["first"], "first.txt"), new File(["second"], "second.txt")],
      taskType: "task",
      fallbackError: "Upload failed",
    })).rejects.toThrow("Upload failed")
  })

  it("does not issue an empty upload request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")

    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [],
      taskType: "task",
      fallbackError: "Upload failed",
    })).resolves.toEqual([])
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
