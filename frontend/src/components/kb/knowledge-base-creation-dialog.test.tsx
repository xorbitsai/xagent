import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())
const inTeamMock = vi.hoisted(() => ({ value: false }))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ inTeam: inTeamMock.value }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
  parseApiResponse: async (response: { json: () => Promise<unknown> }) => ({
    data: await response.json(),
    text: null,
    isHtml: false,
  }),
  // Mirrors api-wrapper.ts: detail wins over message. Getting this backwards
  // silently drops the backend sentence and makes assertions about it vacuous.
  getUploadErrorMessage: (
    _response: unknown,
    parsed: { data?: { detail?: string; message?: string } | null },
    messages: { generic: string }
  ) => parsed?.data?.detail || parsed?.data?.message || messages.generic,
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

// The component imports toast from this wrapper, not from `sonner` directly:
// mocking the raw package would leave the wrapper's injected options in the
// asserted arguments and force a meaningless matcher for them.
vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
    warning: toastWarningMock,
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
    User: Icon,
    Users: Icon,
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
    onConfirm: (files: Array<{ id: string; name: string; size?: string; resourceKey?: string }>) => void
  }) => (
    open && provider ? (
      <>
        <button
          data-testid="mock-cloud-confirm"
          onClick={() => onConfirm([{
            id: `${provider.id}-file-1`,
            name: "alpha.pdf",
            size: "1 KB",
            resourceKey: "resource-secret",
          }])}
        >
          mock cloud confirm
        </button>
        <button
          data-testid="mock-cloud-confirm-no-key"
          onClick={() => onConfirm([{
            id: `${provider.id}-file-2`,
            name: "beta.pdf",
            size: "1 KB",
          }])}
        >
          mock cloud confirm without resource key
        </button>
      </>
    ) : null
  ),
}))

import { KnowledgeBaseCreationDialog } from "./knowledge-base-creation-dialog"

function createJsonResponse(body: unknown, ok = true, status?: number) {
  return {
    ok,
    status: status ?? (ok ? 200 : 500),
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
    if (url.endsWith("/reserve-team") || url.endsWith("/release-team-claim")) {
      return Promise.resolve(createJsonResponse(null, true, 204))
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

const IMPORT_TABS = ["file", "web", "cloud"] as const
type ImportTab = (typeof IMPORT_TABS)[number]

/** Walk the wizard to step 3 (where the create button lives) for one import tab. */
async function goToStep3(
  container: HTMLElement,
  tab: ImportTab,
  fileCount = 1,
  cloudFileHasResourceKey = true,
) {
  fireEvent.click(screen.getByText("common.next"))

  if (tab === "file") {
    fireEvent.change(container.querySelector("#file-upload") as HTMLInputElement, {
      target: {
        files: Array.from(
          { length: fileCount },
          (_, index) => new File(["a"], `file ${index}!.txt`, { type: "text/plain" })
        ),
      },
    })
  } else if (tab === "web") {
    fireEvent.click(screen.getByText("kb.dialog.tabs.web"))
    fireEvent.change(container.querySelector("#start_url") as HTMLInputElement, {
      target: { value: "https://example.com/docs" },
    })
  } else {
    fireEvent.click(screen.getByText("kb.dialog.tabs.cloud"))
    fireEvent.click(screen.getByText("kb.dialog.cloudConnect.googleDrive"))
    fireEvent.click(await screen.findByTestId(
      cloudFileHasResourceKey ? "mock-cloud-confirm" : "mock-cloud-confirm-no-key"
    ))
    await waitFor(() => {
      expect(
        screen.getByText(cloudFileHasResourceKey ? "alpha.pdf" : "beta.pdf")
      ).toBeInTheDocument()
    })
  }

  fireEvent.click(screen.getByText("common.next"))
}

/** The shared guard must warn, park the user on step 1, and ingest nothing. */
async function expectNameRejected(container: HTMLElement) {
  await waitFor(() => {
    expect(toastErrorMock).toHaveBeenCalledWith("kb.errors.nameRequired")
  })

  // Step 1 is the only step rendering the name field, so its presence proves
  // the user is where the problem can actually be fixed.
  const nameInput = container.querySelector("#collection_name")
  expect(nameInput).not.toBeNull()

  // A toast alone leaves a screen reader with nothing at the field itself.
  expect(nameInput?.getAttribute("aria-required")).toBe("true")
  expect(nameInput?.getAttribute("aria-invalid")).toBe("true")
  expect(screen.getByText("kb.errors.nameRequired")).toBeInTheDocument()
  expect(nameInput?.getAttribute("aria-describedby")).toBe("collection_name_error")
  expect(container.querySelector("label[for=collection_name]")?.textContent).toContain("*")

  expect(
    apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/kb/ingest"))
  ).toHaveLength(0)
}

describe("KnowledgeBaseCreationDialog collection naming", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    toastWarningMock.mockReset()
    inTeamMock.value = false
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it.each([
    ["an empty", ""],
    ["a whitespace-only", "   "],
  ])("refuses to leave step 1 with %s collection name", async (_label, value) => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    if (value) {
      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value },
      })
    }

    fireEvent.click(screen.getByText("common.next"))

    await expectNameRejected(container)
    // Step 2 owns the file picker: never rendering it proves we did not advance.
    expect(container.querySelector("#file-upload")).toBeNull()
  })

  it("clears the name error once the user starts typing", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(screen.getByText("common.next"))
    await expectNameRejected(container)

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "t" },
    })

    expect(container.querySelector("#collection_name")?.getAttribute("aria-invalid")).toBe("false")
    expect(screen.queryByText("kb.errors.nameRequired")).toBeNull()
  })

  it("clears the name error when the dialog is closed and reopened", async () => {
    const { container, rerender } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.click(screen.getByText("common.next"))
    await expectNameRejected(container)

    // Escape, the overlay and the close button all bypass the cancel handler,
    // so only the parent's `open` flag flips.
    rerender(<KnowledgeBaseCreationDialog open={false} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)
    rerender(<KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />)

    await waitFor(() => {
      expect(container.querySelector("#collection_name")?.getAttribute("aria-invalid")).toBe("false")
    })
    expect(screen.queryByText("kb.errors.nameRequired")).toBeNull()
  })

  it("previews the name the user typed", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })

    await goToStep3(container, "file")

    expect(screen.getByText("team-docs")).toBeInTheDocument()
    // The step-1 gate is what keeps the preview from ever needing a stand-in
    // name; the "KB <date>" fallback it used to render is gone.
    expect(container.textContent).not.toMatch(/KB \d/)
  })

  it("uses the same explicit collection name for each uploaded file", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })

    await goToStep3(container, "file", 2)
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

  it("advises picking another name when the entered one is taken", async () => {
    // The reported #1139 path: this is the only screen where the user typed the
    // name, so it is the only one allowed to tell them to change it.
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
      if (url === "http://api.local/api/kb/ingest/jobs") {
        return Promise.resolve(
          createJsonResponse(
            {
              detail:
                "Knowledge base name unavailable: test.",
            },
            false,
            409
          )
        )
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "test" },
    })

    fireEvent.click(screen.getByText("common.next"))
    fireEvent.change(container.querySelector("#file-upload") as HTMLInputElement, {
      target: { files: [new File(["a"], "alpha.txt", { type: "text/plain" })] },
    })
    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "kb.errors.nameUnavailable",
        expect.objectContaining({ description: "kb.errors.nameUnavailableHint" })
      )
    })

    // The backend sentence is replaced by the localized copy, not appended.
    expect(JSON.stringify(toastErrorMock.mock.calls)).not.toContain(
      "Knowledge base name unavailable"
    )
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

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "alpha" },
    })

    await goToStep3(container, "file")
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
      const { container } = render(
        <KnowledgeBaseCreationDialog open={true} onOpenChange={onOpenChange} onSuccess={onSuccess} />
      )

      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value: "cloud-docs" },
      })

      await goToStep3(container, "cloud")
      fireEvent.click(screen.getByText("kb.dialog.createButton"))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith(
          "kb.errors.cloudIngestFailed",
          expect.objectContaining({
            description: "Cloud import partially failed",
          })
        )
      })

      const cloudCall = apiRequestMock.mock.calls.find(
        ([url]) => url === "http://api.local/api/kb/ingest-cloud"
      )
      expect(JSON.parse(cloudCall?.[1]?.body as string).files).toEqual([
        {
          provider: "google-drive",
          fileId: "google-drive-file-1",
          fileName: "alpha.pdf",
          resourceKey: "resource-secret",
        },
      ])

      expect(toastSuccessMock).not.toHaveBeenCalled()
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect(onSuccess).not.toHaveBeenCalled()
      expect(await screen.findByText("Cloud import partially failed")).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("omits an absent cloud resource key from the ingest request", async () => {
    const baseApiRequest = apiRequestMock.getMockImplementation()
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url === "http://api.local/api/kb/ingest-cloud") {
        return Promise.resolve(createJsonResponse([{
          status: "success",
          message: "ok",
          doc_id: "doc-1",
          chunk_count: 1,
          embedding_count: 1,
        }]))
      }
      if (!baseApiRequest) {
        throw new Error(`Unhandled apiRequest: ${url}`)
      }
      return baseApiRequest(url, options)
    })

    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "cloud-docs" },
    })

    await goToStep3(container, "cloud", 1, false)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/kb/ingest-cloud",
        expect.any(Object),
      )
    })
    const cloudCall = apiRequestMock.mock.calls.find(
      ([url]) => url === "http://api.local/api/kb/ingest-cloud"
    )
    expect(JSON.parse(cloudCall?.[1]?.body as string).files).toEqual([
      {
        provider: "google-drive",
        fileId: "google-drive-file-2",
        fileName: "beta.pdf",
      },
    ])
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

      fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
        target: { value: "web_collection" },
      })

      await goToStep3(container, "web")
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

const RESERVE_URL = "http://api.local/api/knowledge-bases/team-docs/reserve-team"
const RELEASE_URL = "http://api.local/api/knowledge-bases/team-docs/release-team-claim"

function callsTo(url: string) {
  return apiRequestMock.mock.calls.filter(([called]) => called === url)
}

/** Override one route, leaving every other route on the installed mock. */
function mockRoute(
  match: (url: string) => boolean,
  respond: (url: string, options?: RequestInit) => unknown
) {
  const base = apiRequestMock.getMockImplementation()!
  apiRequestMock.mockImplementation((url: string, options?: RequestInit) =>
    match(url) ? respond(url, options) : base(url, options)
  )
}

/** Name the knowledge base and pick Team, which only renders when `inTeam`. */
function nameAndChooseTeam(container: HTMLElement) {
  fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
    target: { value: "team-docs" },
  })
  fireEvent.click(container.querySelector("#kb-ownership-team") as HTMLElement)
}

describe("KnowledgeBaseCreationDialog ownership", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    toastWarningMock.mockReset()
    inTeamMock.value = true
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it("hides the selector and never touches the team endpoints outside a team", async () => {
    // The open-source single-node build has no team and no /api/knowledge-bases
    // routes: the selector must not render and no request may be made.
    inTeamMock.value = false
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })
    expect(container.querySelector("#kb-ownership-team")).toBeNull()
    expect(container.querySelector("#kb-ownership-personal")).toBeNull()

    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).includes("/api/knowledge-bases/"))
    ).toHaveLength(0)
  })

  it("defaults to personal inside a team and reserves nothing", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    expect(container.querySelector("#kb-ownership-personal")?.className).toContain("border-primary")

    fireEvent.change(container.querySelector("#collection_name") as HTMLInputElement, {
      target: { value: "team-docs" },
    })
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(1)
    })
    expect(callsTo(RESERVE_URL)).toHaveLength(0)
  })

  it("moves the ownership choice with the keyboard, not just the mouse", () => {
    // These are Cards standing in for radios, so the keyboard handling a real
    // radio group gets for free is written out — pin the essentials.
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const personal = container.querySelector("#kb-ownership-personal") as HTMLElement
    const team = container.querySelector("#kb-ownership-team") as HTMLElement
    expect(personal.getAttribute("role")).toBe("radio")
    expect(personal.getAttribute("tabindex")).toBe("0")
    expect(team.getAttribute("tabindex")).toBe("-1")

    fireEvent.keyDown(personal, { key: "ArrowRight" })
    expect(team.getAttribute("aria-checked")).toBe("true")
    // The tab stop and focus follow the selection, or the arrow key would
    // strand the user on a card that is no longer tabbable.
    expect(team.getAttribute("tabindex")).toBe("0")
    expect(document.activeElement).toBe(team)

    fireEvent.keyDown(team, { key: "ArrowRight" })
    expect(personal.getAttribute("aria-checked")).toBe("true")

    // Space and Enter select the card under focus, not the current selection.
    fireEvent.keyDown(team, { key: " " })
    expect(team.getAttribute("aria-checked")).toBe("true")
    fireEvent.keyDown(personal, { key: "Enter" })
    expect(personal.getAttribute("aria-checked")).toBe("true")

    fireEvent.keyDown(personal, { key: "End" })
    expect(team.getAttribute("aria-checked")).toBe("true")
    expect(document.activeElement).toBe(team)
    fireEvent.keyDown(team, { key: "Home" })
    expect(personal.getAttribute("aria-checked")).toBe("true")
    expect(document.activeElement).toBe(personal)
  })

  it("reserves the name exactly once before a multi-file team ingest", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file", 2)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs", "team-docs"])
    })
    expect(callsTo(RESERVE_URL)).toHaveLength(1)
    // The claim lands before the first byte does.
    const reserveIndex = apiRequestMock.mock.calls.findIndex(([url]) => url === RESERVE_URL)
    const ingestIndex = apiRequestMock.mock.calls.findIndex(
      ([url]) => url === "http://api.local/api/kb/ingest/jobs"
    )
    expect(reserveIndex).toBeGreaterThan(-1)
    expect(reserveIndex).toBeLessThan(ingestIndex)
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it.each([
    ["web", "http://api.local/api/kb/ingest-web/jobs"],
    ["cloud", "http://api.local/api/kb/ingest-cloud"],
  ] as const)("reserves once before the %s ingest too", async (tab, ingestUrl) => {
    mockRoute(
      (url) => url === ingestUrl,
      () =>
        createJsonResponse(
          tab === "web"
            ? createSucceededJob({
                status: "success",
                collection: "team-docs",
                total_urls_found: 1,
                pages_crawled: 1,
                pages_failed: 0,
                documents_created: 1,
                chunks_created: 1,
                embeddings_created: 1,
                crawled_urls: [],
                failed_urls: {},
                message: "ok",
                warnings: [],
                elapsed_time_ms: 0,
              })
            : [{ status: "success", collection: "team-docs", message: "ok", document_count: 1, chunks_count: 1 }]
        )
    )
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, tab)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(ingestUrl)).toHaveLength(1)
    })
    expect(callsTo(RESERVE_URL)).toHaveLength(1)
  })

  it("rejects the name and skips the ingest when the reserve answers 409", async () => {
    mockRoute(
      (url) => url === RESERVE_URL,
      () => createJsonResponse({ detail: "name taken" }, false, 409)
    )
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("kb.errors.nameUnavailable", expect.anything())
    })
    expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(0)
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
    // Back on step 1, at the field where the name can actually be changed.
    expect(screen.getByText("kb.errors.nameUnavailableHint")).toBeInTheDocument()
    expect(container.querySelector("#collection_name")).not.toBeNull()
  })

  it("releases the claim once when the ingest fails", async () => {
    mockRoute(
      (url) => url === "http://api.local/api/kb/ingest/jobs",
      () => createJsonResponse({ detail: "ingest exploded" }, false, 500)
    )
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(RELEASE_URL)).toHaveLength(1)
    })
    expect(toastErrorMock).toHaveBeenCalled()
    // A clean release says nothing: the ingest error is what the user needs.
    expect(toastWarningMock).not.toHaveBeenCalled()
  })

  it("keeps the claim when part of a multi-file upload already landed", async () => {
    // File 1 put data behind the claim, so the failure of file 2 must not
    // release the name out from under a live team collection.
    let ingestCalls = 0
    mockRoute(
      (url) => url === "http://api.local/api/kb/ingest/jobs",
      () => {
        ingestCalls += 1
        return ingestCalls === 1
          ? createJsonResponse(
              createSucceededJob({
                status: "success",
                collection: "team-docs",
                document_count: 1,
                chunks_count: 1,
                message: "ok",
              })
            )
          : createJsonResponse({ detail: "ingest exploded" }, false, 500)
      }
    )
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file", 2)
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs"])
    })
    expect(toastErrorMock).toHaveBeenCalled()
    expect(callsTo(RELEASE_URL)).toHaveLength(0)
  })

  it.each([
    ["a network error", () => Promise.reject(new Error("connection reset"))],
    ["a 5xx", () => Promise.resolve(createJsonResponse({ detail: "boom" }, false, 503))],
  ])("releases an ambiguous claim when the reserve dies with %s", async (_label, respond) => {
    // A thrown or 5xx reserve may still have committed server-side, so the
    // ingest must not start and the possible claim is given back.
    mockRoute((url) => url === RESERVE_URL, respond)
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(callsTo(RELEASE_URL)).toHaveLength(1)
    })
    expect(toastErrorMock).toHaveBeenCalled()
    expect(callsTo("http://api.local/api/kb/ingest/jobs")).toHaveLength(0)
  })

  it("warns without clobbering the ingest error when the release also fails", async () => {
    mockRoute(
      (url) => url === "http://api.local/api/kb/ingest/jobs",
      () => createJsonResponse({ detail: "ingest exploded" }, false, 500)
    )
    mockRoute(
      (url) => url === RELEASE_URL,
      () => createJsonResponse({ detail: "storage offline" }, false, 500)
    )
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )
    nameAndChooseTeam(container)
    await goToStep3(container, "file")
    fireEvent.click(screen.getByText("kb.dialog.createButton"))

    await waitFor(() => {
      expect(toastWarningMock).toHaveBeenCalledWith("kb.ownership.releaseFailed")
    })
    expect(toastErrorMock).toHaveBeenCalled()
  })
})
