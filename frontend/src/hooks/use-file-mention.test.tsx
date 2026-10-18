import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const createFileChipHTMLMock = vi.hoisted(() => vi.fn(() => "<span>chip</span>"))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/components/chat/FileChip", () => ({
  createFileChipHTML: createFileChipHTMLMock,
}))

import { useFileMention } from "./use-file-mention"

function HookHarness({ filesDisabled = false, inputKey = "Enter", preventDefault = () => {} }: {
  filesDisabled?: boolean; inputKey?: string; preventDefault?: () => void
}) {
  const editorRef = React.useRef<HTMLDivElement>(null)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const [keyHandled, setKeyHandled] = React.useState<boolean | null>(null)
  const mention = useFileMention(
    editorRef,
    containerRef,
    () => { },
    (key) => key,
    filesDisabled,
  )

  return (
    <div ref={containerRef} data-testid="container">
      <div ref={editorRef} data-testid="editor" contentEditable suppressContentEditableWarning />
      <button onClick={() => mention.checkTrigger()}>check-trigger</button>
      <button onClick={() => mention.setShowFilePicker(true)}>force-picker</button>
      <button onClick={() => mention.insertFile({
        file_id: "file-1",
        filename: "report.txt",
        file_size: 12,
        modified_time: 1,
      })}>
        insert-file
      </button>
      <button onClick={() => setKeyHandled(mention.handleKeyDown({
        key: inputKey,
        preventDefault,
      } as unknown as React.KeyboardEvent))}>
        handle-key
      </button>
      <div data-testid="picker-visible">{String(mention.showFilePicker)}</div>
      <div data-testid="key-handled">{String(keyHandled)}</div>
      <div data-testid="selected-index">{mention.selectedFileIndex}</div>
      <div data-testid="loading">{String(mention.isLoadingFiles)}</div>
      <div data-testid="file-count">{mention.filteredFiles.length}</div>
    </div>
  )
}

describe("useFileMention", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    createFileChipHTMLMock.mockClear()
    vi.useFakeTimers()
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value: vi.fn(() => ({
        x: 0,
        y: 300,
        width: 0,
        height: 20,
        top: 300,
        right: 0,
        bottom: 320,
        left: 0,
        toJSON: () => ({}),
      })),
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(document, "execCommand")
    vi.restoreAllMocks()
    vi.useRealTimers()
    cleanup()
  })

  it.each(["ArrowUp", "ArrowDown", "Escape", "Tab", "x"])("handles the open picker's %s key according to its keyboard contract", async key => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => ({ files: [] }) })
    const preventDefault = vi.fn()
    render(<HookHarness inputKey={key} preventDefault={preventDefault} />)
    fireEvent.click(screen.getByText("force-picker"))
    fireEvent.click(screen.getByText("handle-key"))
    expect(screen.getByTestId("key-handled")).toHaveTextContent(String(key !== "x"))
    expect(screen.getByTestId("picker-visible")).toHaveTextContent(String(key !== "Escape"))
    expect(createFileChipHTMLMock).not.toHaveBeenCalled()
    expect(preventDefault).toHaveBeenCalledTimes(key === "x" ? 0 : 1)
  })

  it.each(["ArrowUp", "ArrowDown"])("moves the selected file with %s and clamps at the list boundary", async key => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => ({ files: [1, 2, 3].map(id => ({
      file_id: String(id), filename: `file-${id}`, file_size: 1, modified_time: 1,
    })) }) })
    const preventDefault = vi.fn()
    const { rerender } = render(<HookHarness inputKey="ArrowDown" preventDefault={preventDefault} />)
    fireEvent.click(screen.getByText("force-picker"))
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    fireEvent.click(screen.getByText("handle-key"))
    expect(screen.getByTestId("selected-index")).toHaveTextContent("1")
    rerender(<HookHarness inputKey={key} preventDefault={preventDefault} />)
    fireEvent.click(screen.getByText("handle-key"))
    fireEvent.click(screen.getByText("handle-key"))
    expect(screen.getByTestId("selected-index")).toHaveTextContent(key === "ArrowUp" ? "0" : "2")
    expect(preventDefault).toHaveBeenCalledTimes(3)
  })

  it("uses server-side search for mention queries", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({
        files: [
          {
            file_id: "file-1",
            filename: "report.txt",
            file_size: 12,
            modified_time: Math.floor(Date.now() / 1000),
          },
        ],
      }),
    })

    render(<HookHarness />)

    const editor = screen.getByTestId("editor")
    editor.textContent = "@report"
    const textNode = editor.firstChild
    expect(textNode).not.toBeNull()

    const range = document.createRange()
    range.setStart(textNode!, 7)
    range.collapse(true)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    fireEvent.click(screen.getByText("check-trigger"))

    await act(async () => {
      vi.advanceTimersByTime(150)
      await Promise.resolve()
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/files/list?page=1&size=20&search=report"
    )
    // Closing on a lost selection must clear the query, not just hide the UI.
    selection?.removeAllRanges()
    fireEvent.click(screen.getByText("check-trigger"))
    expect(screen.getByTestId("picker-visible")).toHaveTextContent("false")
    fireEvent.click(screen.getByText("force-picker"))
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    expect(apiRequestMock).toHaveBeenLastCalledWith("http://api.local/api/files/list?page=1&size=20")
  })

  it("keeps mention fetch, picker, insertion, and key handling inert when files are disabled", async () => {
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn(),
    })
    render(<HookHarness filesDisabled />)

    const editor = screen.getByTestId("editor")
    editor.textContent = "@report"
    const textNode = editor.firstChild
    expect(textNode).not.toBeNull()

    const range = document.createRange()
    range.setStart(textNode!, 7)
    range.collapse(true)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    fireEvent.click(screen.getByText("check-trigger"))
    fireEvent.click(screen.getByText("force-picker"))

    await act(async () => {
      vi.advanceTimersByTime(500)
      await Promise.resolve()
    })

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.getByTestId("picker-visible")).toHaveTextContent("false")

    fireEvent.click(screen.getByText("insert-file"))
    fireEvent.click(screen.getByText("handle-key"))

    expect(screen.getByTestId("key-handled")).toHaveTextContent("false")
    expect(createFileChipHTMLMock).not.toHaveBeenCalled()
  })

  it("cancels a live mention surface when the file capability changes to disabled", async () => {
    const { rerender } = render(<HookHarness />)
    fireEvent.click(screen.getByText("force-picker"))
    expect(screen.getByTestId("picker-visible")).toHaveTextContent("true")

    rerender(<HookHarness filesDisabled />)
    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByTestId("picker-visible")).toHaveTextContent("false")
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("reports an authenticated listing error and clears loading", async () => {
    let rejectListing!: (error: Error) => void
    apiRequestMock.mockImplementation(() => new Promise((_resolve, reject) => { rejectListing = reject }))
    vi.spyOn(console, "error").mockImplementation(() => {})
    render(<HookHarness />)
    fireEvent.click(screen.getByText("force-picker"))
    await act(async () => { await vi.advanceTimersByTimeAsync(150) })
    expect(screen.getByTestId("loading")).toHaveTextContent("true")
    await act(async () => { rejectListing(new Error("listing failed")) })
    expect(toastErrorMock).toHaveBeenCalledWith("files.previewDialog.errors.loadFailed")
    expect(screen.getByTestId("loading")).toHaveTextContent("false")
    expect(screen.getByTestId("file-count")).toHaveTextContent("0")
  })
})
