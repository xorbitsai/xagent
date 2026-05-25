import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())

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
  cn: (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(" "),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
    warning: vi.fn(),
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Upload: Icon,
    Globe: Icon,
    Settings: Icon,
    CheckCircle: Icon,
    Clock: Icon,
    XCircle: Icon,
    AlertCircle: Icon,
    FileText: Icon,
    Cloud: Icon,
    Database: Icon,
    ChevronDown: Icon,
    ChevronUp: Icon,
    ArrowRight: Icon,
    ArrowLeft: Icon,
  }
})

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => <label {...props}>{children}</label>,
}))

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
}))

vi.mock("@/components/ui/card", () => ({
  Card: ({
    children,
    ...props
  }: React.HTMLAttributes<HTMLDivElement> & { children: React.ReactNode }) => <div {...props}>{children}</div>,
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/textarea", () => ({
  Textarea: (props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => <textarea {...props} />,
}))

vi.mock("@/components/ui/progress", () => ({
  Progress: ({ value }: { value: number }) => <div data-testid="progress">{value}</div>,
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))

vi.mock("@/components/ui/select", () => ({
  Select: () => <div />,
}))

vi.mock("@/components/ui/stepper", () => ({
  Stepper: () => <div />,
}))

vi.mock("./cloud-connect-dialog", () => ({
  CloudConnectDialog: ({
    open,
    provider,
    onConfirm,
  }: {
    open: boolean
    provider: { id: string } | null
    onConfirm: (files: Array<{ id: string; name: string; size?: string }>) => void
  }) => (
    open && provider ? (
      <button
        data-testid="mock-cloud-confirm"
        onClick={() => onConfirm([{ id: `${provider.id}-file-1`, name: "alpha.pdf", size: "1 KB" }])}
      >
        mock cloud confirm
      </button>
    ) : null
  ),
}))

import { KnowledgeBaseCreationDialog } from "./knowledge-base-creation-dialog"

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
    job_type: "kb.ingest.document",
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

function installApiMocks() {
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
    if (url === "http://api.local/api/models/?category=embedding") {
      return Promise.resolve(createJsonResponse([]))
    }
    if (url === "http://api.local/api/models/user-default") {
      return Promise.resolve(createJsonResponse({}))
    }
    if (url === "http://api.local/api/jobs/capabilities") {
      return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
    }
    if (url === "http://api.local/api/kb/ingest/jobs") {
      return Promise.resolve(
        createJsonResponse(
          createSucceededJob({
            status: "success",
            collection: (options?.body as FormData).get("collection"),
            document_count: 1,
            chunks_count: 1,
            message: "ok",
          })
        )
      )
    }

    throw new Error(`Unhandled apiRequest: ${url}`)
  })
}

describe("KnowledgeBaseCreationDialog multi-file naming", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it("requires an explicit collection name for multiple file uploads", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(screen.getByText("common.next"))

    const fileInput = container.querySelector("#file-upload") as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(["a"], "alpha.txt", { type: "text/plain" }),
          new File(["b"], "beta.txt", { type: "text/plain" }),
        ],
      },
    })

    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("kb.errors.multiFileNameRequired")
    })

    const ingestCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest/jobs")
    expect(ingestCalls).toHaveLength(0)
  })

  it("uses the same explicit collection name for each uploaded file", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })

    fireEvent.click(screen.getByText("common.next"))

    const fileInput = container.querySelector("#file-upload") as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(["a"], "alpha.txt", { type: "text/plain" }),
          new File(["b"], "beta.txt", { type: "text/plain" }),
        ],
      },
    })

    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      const ingestCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest/jobs")
      expect(ingestCalls).toHaveLength(2)
      for (const [, options] of ingestCalls) {
        expect((options?.body as FormData).get("collection")).toBe("team-docs")
      }
    })

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs", "team-docs"])
    })
  })

  it("uses the sync ingest endpoint when background jobs are unavailable", async () => {
    const onSuccess = vi.fn()
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "sync" }))
      }
      if (url === "http://api.local/api/kb/ingest") {
        return Promise.resolve(
          createJsonResponse({
            status: "success",
            collection: (options?.body as FormData).get("collection"),
            document_count: 1,
            chunks_count: 1,
            message: "ok",
          })
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.click(screen.getByText("common.next"))

    const fileInput = container.querySelector("#file-upload") as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [new File(["a"], "alpha.txt", { type: "text/plain" })],
      },
    })

    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      const syncCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest")
      const jobCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest/jobs")
      expect(syncCalls).toHaveLength(1)
      expect(jobCalls).toHaveLength(0)
    })

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["alpha"])
    })
  })

  it("keeps the dialog open for cloud partial failures and surfaces the failure message", async () => {
    const onOpenChange = vi.fn()
    const onSuccess = vi.fn()
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=embedding") {
        return Promise.resolve(createJsonResponse([]))
      }
      if (url === "http://api.local/api/models/user-default") {
        return Promise.resolve(createJsonResponse({}))
      }
      if (url === "http://api.local/api/jobs/capabilities") {
        return Promise.resolve(createJsonResponse({ kb_ingest_mode: "celery" }))
      }
      if (url === "http://api.local/api/kb/ingest-cloud") {
        return Promise.resolve(
          createJsonResponse([
            {
              status: "partial",
              message: "Cloud import partially failed",
              doc_id: "doc-1",
              chunk_count: 2,
              embedding_count: 0,
              completed_steps: [{ name: "register_document" }],
              failed_step: "compute_embeddings",
            },
          ])
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    try {
      render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
      )

      fireEvent.click(screen.getByText("common.next"))
      fireEvent.click(screen.getByText("kb.dialog.tabs.cloud"))
      fireEvent.click(screen.getByText("kb.dialog.cloudConnect.googleDrive"))
      fireEvent.click(await screen.findByTestId("mock-cloud-confirm"))

      await waitFor(() => {
        expect(screen.getByText("alpha.pdf")).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText("common.next"))
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith(
          "kb.errors.cloudIngestFailed",
          expect.objectContaining({
            description: "Cloud import partially failed",
          })
        )
      })

      expect(toastSuccessMock).not.toHaveBeenCalled()
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect(onSuccess).not.toHaveBeenCalled()
      expect(await screen.findByText("Cloud import partially failed")).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("keeps the dialog open for web partial failures and surfaces the failure message", async () => {
    const onOpenChange = vi.fn()
    const onSuccess = vi.fn()
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    apiRequestMock.mockImplementation((url: string) => {
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
              collection: "web_collection",
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

    try {
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
      )

      fireEvent.click(screen.getByText("common.next"))
      fireEvent.click(screen.getByText("kb.dialog.tabs.web"))
      fireEvent.change(container.querySelector("#start_url") as HTMLInputElement, {
        target: { value: "https://example.com/docs" },
      })
      fireEvent.click(screen.getByText("common.next"))
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith(
          "kb.errors.webIngestFailed",
          expect.objectContaining({
            description: "Web import partially failed",
          })
        )
      })

      expect(toastSuccessMock).not.toHaveBeenCalled()
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect(onSuccess).not.toHaveBeenCalled()
      expect(await screen.findByText("kb.dialog.webImport.status.failed")).toBeInTheDocument()
      expect(await screen.findByText("Web import partially failed")).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })
})
