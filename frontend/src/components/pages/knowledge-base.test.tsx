import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string | number>) => {
      if (vars?.name) {
        return `${key}:${vars.name}`
      }

      return key
    },
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
    warning: toastWarningMock,
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Plus: Icon,
    FileText: Icon,
    FolderOpen: Icon,
    HardDrive: Icon,
    Trash2: Icon,
  }
})

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/search-input", () => ({
  SearchInput: ({ value, onChange, containerClassName: _containerClassName, ...props }: { value: string; onChange: (value: string) => void; containerClassName?: string }) => (
    <input {...props} value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}))

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
}))

vi.mock("@/components/ui/sheet", () => ({
  Sheet: ({ open, children }: { open: boolean; children: React.ReactNode }) => (open ? <div>{children}</div> : null),
  SheetContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/kb/knowledge-base-detail", () => ({
  KnowledgeBaseDetailContent: ({ collectionName }: { collectionName: string }) => <div>{collectionName}</div>,
}))

vi.mock("@/components/kb/knowledge-base-creation-dialog", () => ({
  KnowledgeBaseCreationDialog: () => null,
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: ({ isOpen, onConfirm }: { isOpen: boolean; onConfirm: () => void }) => (
    isOpen ? <button onClick={onConfirm}>confirm-delete</button> : null
  ),
}))

import { KnowledgeBasePage } from "./knowledge-base"

function createJsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: vi.fn().mockResolvedValue(body),
  }
}

describe("KnowledgeBasePage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastWarningMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("shows the collection delete action in the detail sheet flow", async () => {
    let collectionFetchCount = 0

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === "http://api.local/api/kb/collections" && !options) {
        collectionFetchCount += 1

        if (collectionFetchCount === 1) {
          return Promise.resolve(createJsonResponse({
            collections: [{
              name: "demo",
              documents: 1,
              parses: 0,
              chunks: 2,
              embeddings: 3,
              document_names: ["report.pdf"],
            }],
          }))
        }

        return Promise.resolve(createJsonResponse({ collections: [] }))
      }

      if (url === "http://api.local/api/kb/collections/demo" && options?.method === "DELETE") {
        return Promise.resolve(createJsonResponse({ status: "success" }))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByText("demo"))

    expect(screen.getByRole("button", { name: "common.delete" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "common.delete" }))
    fireEvent.click(screen.getByText("confirm-delete"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/kb/collections/demo",
        { method: "DELETE" }
      )
    })

    await waitFor(() => {
      expect(collectionFetchCount).toBe(2)
      expect(screen.getByText("kb.empty.noKB")).toBeInTheDocument()
    })
  })

  it("warns but still refreshes when collection delete is only partially successful", async () => {
    let collectionFetchCount = 0

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === "http://api.local/api/kb/collections" && !options) {
        collectionFetchCount += 1

        if (collectionFetchCount === 1) {
          return Promise.resolve(createJsonResponse({
            collections: [{
              name: "demo",
              documents: 1,
              parses: 0,
              chunks: 2,
              embeddings: 3,
              document_names: ["report.pdf"],
            }],
          }))
        }

        return Promise.resolve(createJsonResponse({ collections: [] }))
      }

      if (url === "http://api.local/api/kb/collections/demo" && options?.method === "DELETE") {
        return Promise.resolve(createJsonResponse({
          status: "partial_success",
          message: "cleanup warning",
          warnings: ["cleanup warning"],
        }))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByText("demo"))
    fireEvent.click(screen.getByRole("button", { name: "common.delete" }))
    fireEvent.click(screen.getByText("confirm-delete"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/kb/collections/demo",
        { method: "DELETE" }
      )
    })

    await waitFor(() => {
      expect(collectionFetchCount).toBe(2)
      expect(screen.getByText("kb.empty.noKB")).toBeInTheDocument()
      expect(toastWarningMock).toHaveBeenCalledWith("cleanup warning")
    })
  })
})
