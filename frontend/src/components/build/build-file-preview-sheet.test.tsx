/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    state: {
      filePreview: {
        isOpen: true,
        fileId: "artifact-file-id",
        fileName: "artifact.html",
        viewMode: "preview",
      },
    },
    closeFilePreview: closeFilePreviewMock,
    dispatch: dispatchMock,
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: ({ open }: { open: boolean }) => (
    <div data-testid="file-preview-content">{open ? "open" : "closed"}</div>
  ),
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: () => <div data-testid="file-preview-actions" />,
}))

import { BuildFilePreviewSheet } from "./build-file-preview-sheet"

describe("BuildFilePreviewSheet", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders the shared file preview when generated artifacts are opened", () => {
    render(<BuildFilePreviewSheet />)

    expect(screen.getByText("artifact.html")).toBeInTheDocument()
    expect(screen.getByTestId("file-preview-content")).toHaveTextContent("open")
    expect(screen.getByTestId("file-preview-actions")).toBeInTheDocument()
  })
})
