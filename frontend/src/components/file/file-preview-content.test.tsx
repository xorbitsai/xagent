/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const dispatchMock = vi.hoisted(() => vi.fn())
const fileAccessRequestMock = vi.hoisted(() => vi.fn())
const fileViewerProps = vi.hoisted(() => ({ current: null as { filesDisabled?: boolean } | null }))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      filePreview: {
        fileId: "guest-file",
        fileName: "guest.txt",
        content: "",
        error: null,
        isLoading: false,
        viewMode: "preview",
      },
    },
    dispatch: dispatchMock,
    filesDisabled: true,
    getFilePreviewUrl: () => "/api/files/public/preview/guest-file?token=guest-token",
  }),
}))

vi.mock("@/contexts/file-access-context", () => ({
  useFileAccess: () => ({ request: fileAccessRequestMock }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/file/file-viewer", () => ({
  FileViewer: (props: { filesDisabled?: boolean }) => {
    fileViewerProps.current = props
    return null
  },
}))

import { FilePreviewContent } from "./file-preview-content"

describe("FilePreviewContent", () => {
  afterEach(() => {
    cleanup()
    dispatchMock.mockReset()
    fileAccessRequestMock.mockReset()
    fileViewerProps.current = null
  })

  it("loads preview bytes through the provider-scoped request owner", async () => {
    fileAccessRequestMock.mockResolvedValue(new Response("guest content", {
      headers: { "Content-Type": "text/plain" },
    }))

    render(<FilePreviewContent open />)

    await waitFor(() => expect(fileAccessRequestMock).toHaveBeenCalledWith(
      "/api/files/public/preview/guest-file?token=guest-token",
      expect.objectContaining({ cache: "no-cache" }),
    ))
    expect(dispatchMock).toHaveBeenCalledWith(expect.objectContaining({
      type: "SET_FILE_PREVIEW_CONTENT",
      payload: expect.objectContaining({ content: "guest content" }),
    }))
  })

  it("passes the App file capability to the Markdown preview owner", () => {
    render(<FilePreviewContent open={false} />)

    expect(fileViewerProps.current?.filesDisabled).toBe(true)
  })

  it("reports a non-success response through the preview state owner", async () => {
    fileAccessRequestMock.mockResolvedValue(new Response("", { status: 403 }))

    render(<FilePreviewContent open />)

    await waitFor(() => expect(dispatchMock).toHaveBeenCalledWith({
      type: "SET_FILE_PREVIEW_CONTENT",
      payload: {
        content: "",
        mimeType: undefined,
        error: "files.previewDialog.errors.loadFailed",
      },
    }))
  })

  it.each([
    [new TypeError("Failed to fetch"), "files.previewDialog.errors.cors"],
    [new Error("socket reset"), "files.previewDialog.errors.networkErrorWithMsg"],
  ])("classifies provider request failures without bypassing the policy", async (error, expected) => {
    fileAccessRequestMock.mockRejectedValue(error)

    render(<FilePreviewContent open />)

    await waitFor(() => expect(dispatchMock).toHaveBeenCalledWith(expect.objectContaining({
      type: "SET_FILE_PREVIEW_CONTENT",
      payload: expect.objectContaining({ error: expected }),
    })))
  })
})
