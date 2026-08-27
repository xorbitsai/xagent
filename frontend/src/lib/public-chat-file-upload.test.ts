import { afterEach, describe, expect, it, vi } from "vitest"

import {
  uploadPublicChatFile,
  uploadPublicChatFiles,
} from "./public-chat-file-upload"

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
      new Response(JSON.stringify({ success: true, file_id: "file-1" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
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

  it("uploads files one at a time in source order", async () => {
    let resolveFirst!: (response: Response) => void
    const firstResponse = new Promise<Response>((resolve) => {
      resolveFirst = resolve
    })
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockReturnValueOnce(firstResponse)
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, file_id: "file-2" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
    const first = new File(["one"], "one.txt", { type: "text/plain" })
    const second = new File(["two"], "two.txt", { type: "text/plain" })
    const uploaded = uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [first, second],
      taskType: "task",
      fallbackError: "Upload failed",
    })

    expect(fetchMock).toHaveBeenCalledOnce()

    resolveFirst(new Response(JSON.stringify({ success: true, file_id: "file-1" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    await expect(uploaded).resolves.toEqual([
      expect.objectContaining({ file_id: "file-1" }),
      expect.objectContaining({ file_id: "file-2" }),
    ])
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("does not start a later file after an upload failure", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "File is too large" }), {
          status: 413,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ success: true, file_id: "file-2" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
    await expect(uploadPublicChatFiles({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      files: [
        new File(["oversize"], "large.txt", { type: "text/plain" }),
        new File(["next"], "next.txt", { type: "text/plain" }),
      ],
      taskType: "task",
      fallbackError: "Upload failed",
    })).rejects.toThrow("File is too large")

    expect(fetchMock).toHaveBeenCalledOnce()
  })
})
