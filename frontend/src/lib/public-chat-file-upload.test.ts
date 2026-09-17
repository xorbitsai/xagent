import { afterEach, describe, expect, it, vi } from "vitest"

import { uploadPublicChatFile } from "./public-chat-file-upload"

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
    })).rejects.toMatchObject({
      errorCode: "upload_too_large",
      message: "File is too large. Please reduce the upload size and try again.",
    })

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

  it("rejects a successful response with a whitespace-only file identifier", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ success: true, file_id: "   " }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(uploadPublicChatFile({
      url: "http://api.local/api/share/files/upload",
      accessToken: "guest-token",
      file: new File(["trip"], "trip.txt", { type: "text/plain" }),
      taskType: "task",
      fallbackError: "Upload failed",
    })).rejects.toMatchObject({
      errorCode: "upload_failed",
      message: "Upload failed",
    })
  })
})
