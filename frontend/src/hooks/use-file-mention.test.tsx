import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

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
  createFileChipHTML: () => "<span>chip</span>",
}))

import { useFileMention } from "./use-file-mention"

function HookHarness() {
  const editorRef = React.useRef<HTMLDivElement>(null)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const mention = useFileMention(editorRef, containerRef, () => { }, (key) => key)

  return (
    <div ref={containerRef} data-testid="container">
      <div ref={editorRef} data-testid="editor" contentEditable suppressContentEditableWarning />
      <button onClick={() => mention.checkTrigger()}>check-trigger</button>
    </div>
  )
}

describe("useFileMention", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
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
    vi.restoreAllMocks()
    vi.useRealTimers()
    cleanup()
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
  })
})
