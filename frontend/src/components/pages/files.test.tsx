import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://api.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) => {
      if (vars?.name) return `${key}:${vars.name}`
      if (vars?.count) return `${key}:${vars.count}`
      return key
    },
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button {...props}>{children}</button>
  ),
}))

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))

vi.mock("@/components/ui/search-input", () => ({
  SearchInput: ({ value, onChange, containerClassName: _containerClassName, ...props }: {
    value: string;
    onChange: (value: string) => void;
    containerClassName?: string;
  }) => (
    <input {...props} value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: ({ isOpen, onConfirm }: { isOpen: boolean; onConfirm: () => void }) => (
    isOpen ? <button onClick={onConfirm}>confirm-delete</button> : null
  ),
}))

vi.mock("@/components/file/standalone-file-preview-dialog", () => ({
  StandaloneFilePreviewDialog: () => null,
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Upload: Icon,
    FileText: Icon,
    Image: Icon,
    Video: Icon,
    Archive: Icon,
    Download: Icon,
    Trash2: Icon,
    FileCode: Icon,
    FileJson: Icon,
    FileSpreadsheet: Icon,
    Folder: Icon,
    LayoutGrid: Icon,
    Eye: Icon,
    MessageSquare: Icon,
    ChevronLeft: Icon,
    ChevronRight: Icon,
  }
})

import { FilesPage } from "./files"

describe("FilesPage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("shows backend delete error details when durable storage delete is unavailable", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url.startsWith("http://api.local/api/files/list")) {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [{
              file_id: "file-1",
              filename: "report.txt",
              file_size: 12,
              modified_time: Math.floor(Date.now() / 1000),
            }],
            total_count: 1,
            page: 1,
            pages: 1,
          }),
        })
      }

      if (url === "http://api.local/api/files/tasks") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            tasks: [],
          }),
        })
      }

      if (url === "http://api.local/api/files/file-1" && options?.method === "DELETE") {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: "Startup file storage sync failed" }), {
            status: 503,
            headers: { "Content-Type": "application/json" },
          })
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<FilesPage />)

    await screen.findByText("report.txt")

    fireEvent.click(screen.getByTitle("files.actions.delete"))
    fireEvent.click(screen.getByText("confirm-delete"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "Startup file storage sync failed",
        expect.anything(),
      )
    })
  })

  it("requests paginated files and loads the next page", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/files/list?page=1&size=20") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [
              {
                file_id: "file-1",
                filename: "first.txt",
                file_size: 12,
                modified_time: Math.floor(Date.now() / 1000),
              },
            ],
            total_count: 21,
            page: 1,
            pages: 2,
          }),
        })
      }

      if (url === "http://api.local/api/files/list?page=2&size=20") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [
              {
                file_id: "file-2",
                filename: "second.txt",
                file_size: 24,
                modified_time: Math.floor(Date.now() / 1000),
              },
            ],
            total_count: 21,
            page: 2,
            pages: 2,
          }),
        })
      }

      if (url === "http://api.local/api/files/tasks") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            tasks: [
              {
                task_id: 101,
                title: "Demo task",
                file_count: 1,
              },
            ],
          }),
        })
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<FilesPage />)

    await screen.findByText("first.txt")
    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/files/list?page=1&size=20")
    expect(screen.getByText("Demo task")).toBeInTheDocument()

    fireEvent.click(screen.getByText("files.pagination.next"))

    await screen.findByText("second.txt")
    expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/files/list?page=2&size=20")
  })

  it("debounces search and resets to page 1 with a single request", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/files/list?page=1&size=20") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [
              {
                file_id: "file-1",
                filename: "first.txt",
                file_size: 12,
                modified_time: Math.floor(Date.now() / 1000),
              },
            ],
            total_count: 21,
            page: 1,
            pages: 2,
          }),
        })
      }

      if (url === "http://api.local/api/files/list?page=2&size=20") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [
              {
                file_id: "file-2",
                filename: "second.txt",
                file_size: 24,
                modified_time: Math.floor(Date.now() / 1000),
              },
            ],
            total_count: 21,
            page: 2,
            pages: 2,
          }),
        })
      }

      if (url === "http://api.local/api/files/list?page=1&size=20&search=second") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            files: [
              {
                file_id: "file-2",
                filename: "second.txt",
                file_size: 24,
                modified_time: Math.floor(Date.now() / 1000),
              },
            ],
            total_count: 1,
            page: 1,
            pages: 1,
          }),
        })
      }

      if (url === "http://api.local/api/files/tasks") {
        return Promise.resolve({
          ok: true,
          json: vi.fn().mockResolvedValue({
            tasks: [],
          }),
        })
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<FilesPage />)

    await screen.findByText("first.txt")
    fireEvent.click(screen.getByText("files.pagination.next"))
    await screen.findByText("second.txt")

    vi.useFakeTimers()
    apiRequestMock.mockClear()
    fireEvent.change(screen.getByPlaceholderText("files.search.placeholder"), {
      target: { value: "second" },
    })

    expect(apiRequestMock).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(300)
      await Promise.resolve()
    })

    expect(apiRequestMock).toHaveBeenCalledTimes(1)
    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/files/list?page=1&size=20&search=second"
    )
    vi.useRealTimers()
  })
})
