import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
  parseApiResponse: async (response: { json: () => Promise<unknown> }) => ({
    data: await response.json(),
    text: null,
    isHtml: false,
  }),
  getUploadErrorMessage: (
    _response: unknown,
    parsed: { data?: { message?: string } | null },
    messages: { generic: string }
  ) => parsed?.data?.message || messages.generic,
  isJsonRecord: (value: unknown) => typeof value === "object" && value !== null && !Array.isArray(value),
  UPLOAD_ERROR_MESSAGES: {},
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
    warning: vi.fn(),
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    ArrowLeft: Icon,
    HardDrive: Icon,
    Search: Icon,
    Upload: Icon,
    Plus: Icon,
    Trash2: Icon,
    FileIcon: Icon,
    CheckCircle: Icon,
    XCircle: Icon,
    AlertCircle: Icon,
    Globe: Icon,
    Loader2: Icon,
  }
})

vi.mock("@radix-ui/react-tabs", () => ({
  Trigger: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => <label {...props}>{children}</label>,
}))

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement> & { children: React.ReactNode }) => <div {...props}>{children}</div>,
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({
    children,
    open = true,
  }: {
    children: React.ReactNode
    open?: boolean
  }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => <div />,
}))

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: () => null,
}))

vi.mock("./knowledge-base-document-list", () => ({
  KnowledgeBaseDocumentList: () => <div data-testid="kb-document-list" />,
}))

import { KnowledgeBaseDetailContent } from "./knowledge-base-detail"

function createJsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: vi.fn().mockResolvedValue(body),
  }
}

function createSucceededJob(result: Record<string, unknown>) {
  return {
    id: "job-1",
    user_id: 1,
    job_type: "kb.ingest.web",
    queue: "kb",
    status: "succeeded",
    progress: { message: "Completed", completed: 1, total: 1 },
    result,
    error_message: null,
    celery_task_id: "task-1",
    attempts: 1,
    max_attempts: 3,
  }
}

describe("KnowledgeBaseDetailContent web ingest", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/kb/collections") {
        return Promise.resolve(
          createJsonResponse({
            collections: [
              {
                name: "demo",
                documents: 0,
                chunks: 0,
                embeddings: 0,
                parses: 0,
                document_names: [],
              },
            ],
          })
        )
      }
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url === "http://api.local/api/kb/ingest-web/jobs") {
        return Promise.resolve(
          createJsonResponse(
            createSucceededJob({
              status: "partial",
              collection: "demo",
              total_urls_found: 1,
              pages_crawled: 1,
              pages_failed: 1,
              documents_created: 0,
              chunks_created: 0,
              embeddings_created: 0,
              crawled_urls: [],
              failed_urls: {
                "https://example.com/docs": "embedding missing",
              },
              message: "Web import partially failed",
              warnings: [],
              elapsed_time_ms: 0,
            })
          )
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps the add-source dialog open for partial web failures and surfaces the error", async () => {
    render(<KnowledgeBaseDetailContent collectionName="demo" />)

    await waitFor(() => {
      expect(screen.getByText("kb.detail.files.addSource")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText("kb.detail.files.addSource"))
    fireEvent.click(screen.getByText("kb.dialog.tabs.web"))
    fireEvent.change(screen.getByLabelText("kb.dialog.webImport.basic.startUrl *"), {
      target: { value: "https://example.com/docs" },
    })
    fireEvent.click(screen.getByText("kb.index.startImport"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.detail.errors.webImportFailed",
        expect.objectContaining({
          description: "Web import partially failed",
        })
      )
    })

    expect(await screen.findByText("kb.dialog.webImport.status.failed")).toBeInTheDocument()
    expect(await screen.findByText("Web import partially failed")).toBeInTheDocument()
    expect(screen.getByDisplayValue("https://example.com/docs")).toBeInTheDocument()

    const collectionCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/collections")
    expect(collectionCalls).toHaveLength(1)
  })
})
