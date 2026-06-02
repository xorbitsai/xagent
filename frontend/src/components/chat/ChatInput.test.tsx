import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const openFilePreviewMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const resetMentionMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://upload.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    openFilePreview: openFilePreviewMock,
  }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: routerPushMock,
  }),
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: resetMentionMock,
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

import { ChatInput } from "./ChatInput"

const emptyJsonResponse = () =>
  new Response(JSON.stringify([]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })

describe("ChatInput", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation(() => Promise.resolve(emptyJsonResponse()))
    openFilePreviewMock.mockReset()
    routerPushMock.mockReset()
    resetMentionMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("requires a model when submitting generic chat", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(screen.getByText("chatPage.input.noModelAlert")).toBeInTheDocument()
    })
    expect(onSend).not.toHaveBeenCalled()
  })

  it("allows selected agent submissions without a local model", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
        selectedAgents={[{ id: 42, name: "Shared Agent" }]}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "hello",
        expect.objectContaining({ model: "" })
      )
    })
    expect(resetMentionMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("chatPage.input.noModelAlert")).not.toBeInTheDocument()
  })

  it("does not show pause for uppercase terminal task status", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="next request"
        isLoading
        onInputChange={vi.fn()}
        onPause={vi.fn()}
        onSend={vi.fn()}
        taskStatus="FAILED"
      />
    )

    expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).not.toBeDisabled()
  })
})
