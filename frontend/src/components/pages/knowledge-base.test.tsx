import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const authMock = vi.hoisted(() => ({
  user: { id: 1, is_admin: false },
  inTeam: true,
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

// Partial mock: only the API base URL is stubbed. Replacing the whole module
// strips `cn`, which every rendered UI primitive calls.
vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en",
    t: (key: string, vars?: Record<string, string | number>) => {
      if (vars?.name) {
        return `${key}:${vars.name}`
      }
      if (vars?.owners) {
        return `${key}:${vars.owners}`
      }

      return key
    },
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authMock,
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
    warning: toastWarningMock,
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Plus: Icon,
    FileText: Icon,
    FolderOpen: Icon,
    Globe: Icon,
    HardDrive: Icon,
    Database: Icon,
    Plug: Icon,
    Search: Icon,
    Settings2: Icon,
    Trash2: Icon,
    UploadCloud: Icon,
    Users: Icon,
    X: Icon,
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

function createStatusResponse(status: number, body?: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body ?? {}),
  }
}

const ONE_COLLECTION = {
  collections: [{
    name: "demo",
    documents: 1,
    parses: 0,
    chunks: 2,
    embeddings: 3,
    document_names: ["report.pdf"],
    ownership: "personal",
  }],
}

describe("KnowledgeBasePage", () => {
  beforeEach(() => {
    authMock.user.is_admin = false
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastWarningMock.mockReset()
    toastSuccessMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("shows owner email instead of an opaque username to administrators", async () => {
    authMock.user.is_admin = true
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/kb/collections") {
        return Promise.resolve(createJsonResponse({
          collections: [{
            ...ONE_COLLECTION.collections[0],
            owners: [2],
          }],
        }))
      }
      if (url.startsWith("http://api.local/api/admin/users?")) {
        return Promise.resolve(createJsonResponse({
          users: [{
            id: 2,
            username: "acct_0123456789abcdef0123456789abcdef",
            email: "owner@example.com",
          }],
          pages: 1,
        }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    expect(await screen.findByText("kb.card.ownerLabel:owner@example.com")).toBeInTheDocument()
    expect(screen.queryByText(/acct_0123456789abcdef/)).not.toBeInTheDocument()
  })

  it("formats an incomplete owner fallback with the account ID convention", async () => {
    authMock.user.is_admin = true
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/kb/collections") {
        return Promise.resolve(createJsonResponse({
          collections: [{
            ...ONE_COLLECTION.collections[0],
            owners: [2],
          }],
        }))
      }
      if (url.startsWith("http://api.local/api/admin/users?")) {
        return Promise.resolve(createJsonResponse({
          users: [{ id: 2, username: "", email: null }],
          pages: 1,
        }))
      }
      return Promise.resolve(createStatusResponse(404, { detail: "not found" }))
    })

    render(<KnowledgeBasePage />)

    expect(await screen.findByText("kb.card.ownerLabel:#2")).toBeInTheDocument()
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
      expect(screen.getByText("kb.emptyState.title")).toBeInTheDocument()
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
      expect(screen.getByText("kb.emptyState.title")).toBeInTheDocument()
      expect(toastWarningMock).toHaveBeenCalledWith("cleanup warning")
    })
  })

  it("waits for the promotion job to finish before reporting success", async () => {
    let jobPollCount = 0
    let collectionFetchCount = 0

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === "http://api.local/api/kb/collections" && !options) {
        collectionFetchCount += 1
        return Promise.resolve(createJsonResponse(ONE_COLLECTION))
      }

      if (url === "http://api.local/api/knowledge-bases/demo/promote-team" && options?.method === "POST") {
        return Promise.resolve(createStatusResponse(202, {
          id: "job-1",
          user_id: 1,
          job_type: "kb.team.transfer",
          queue: "kb",
          status: "running",
          attempts: 1,
          max_attempts: 3,
        }))
      }

      if (url === "http://api.local/api/jobs/job-1") {
        jobPollCount += 1
        return Promise.resolve(createJsonResponse({
          id: "job-1",
          user_id: 1,
          job_type: "kb.team.transfer",
          queue: "kb",
          status: jobPollCount === 1 ? "running" : "succeeded",
          attempts: 1,
          max_attempts: 3,
        }))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByTitle("kb.ownership.makeTeam"))

    // First poll still reports running: the transfer is not done, so the user
    // must not be told it succeeded yet.
    await waitFor(() => {
      expect(jobPollCount).toBe(1)
    }, { timeout: 3000 })
    expect(toastSuccessMock).not.toHaveBeenCalled()

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledWith("kb.ownership.teamSuccess")
    }, { timeout: 4000 })
    expect(toastErrorMock).not.toHaveBeenCalled()

    // Let the post-success refresh finish inside this test, so its request does
    // not land on the next test's mock.
    await waitFor(() => {
      expect(collectionFetchCount).toBeGreaterThanOrEqual(2)
    })
  })

  it("surfaces a failed promotion job as an error", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === "http://api.local/api/kb/collections" && !options) {
        return Promise.resolve(createJsonResponse(ONE_COLLECTION))
      }

      if (url === "http://api.local/api/knowledge-bases/demo/promote-team" && options?.method === "POST") {
        return Promise.resolve(createStatusResponse(202, {
          id: "job-2",
          user_id: 1,
          job_type: "kb.team.transfer",
          queue: "kb",
          status: "enqueued",
          attempts: 1,
          max_attempts: 3,
        }))
      }

      if (url === "http://api.local/api/jobs/job-2") {
        return Promise.resolve(createJsonResponse({
          id: "job-2",
          user_id: 1,
          job_type: "kb.team.transfer",
          queue: "kb",
          status: "failed",
          error_message: "team storage already contains this knowledge base",
          attempts: 3,
          max_attempts: 3,
        }))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByTitle("kb.ownership.makeTeam"))

    await waitFor(() => {
      expect(toastErrorMock.mock.calls.map((call) => call[0])).toContain(
        "team storage already contains this knowledge base",
      )
    }, { timeout: 3000 })
    expect(toastSuccessMock).not.toHaveBeenCalled()
  })

  it("surfaces an error when the 202 job body is malformed", async () => {
    const seenUrls: string[] = []
    let collectionFetchCount = 0

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      seenUrls.push(url)

      if (url === "http://api.local/api/kb/collections" && !options) {
        collectionFetchCount += 1
        return Promise.resolve(createJsonResponse(ONE_COLLECTION))
      }

      if (url === "http://api.local/api/knowledge-bases/demo/promote-team" && options?.method === "POST") {
        // Accepted, but the body is not a job descriptor: the transfer cannot be
        // tracked, so its completion is unknown and must not read as success.
        return Promise.resolve(createStatusResponse(202, { not_a_valid_job: true }))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByTitle("kb.ownership.makeTeam"))

    await waitFor(() => {
      expect(toastErrorMock.mock.calls.map((call) => call[0])).toContain("kb.ownership.failed")
    })
    expect(toastSuccessMock).not.toHaveBeenCalled()
    expect(seenUrls.some((url) => url.includes("/api/jobs/"))).toBe(false)
    // No success means no post-success refresh: only the initial mount fetch.
    expect(collectionFetchCount).toBe(1)
  })

  it("surfaces an error when the 202 job body cannot be parsed", async () => {
    let collectionFetchCount = 0

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === "http://api.local/api/kb/collections" && !options) {
        collectionFetchCount += 1
        return Promise.resolve(createJsonResponse(ONE_COLLECTION))
      }

      if (url === "http://api.local/api/knowledge-bases/demo/promote-team" && options?.method === "POST") {
        return Promise.resolve({
          ok: true,
          status: 202,
          json: vi.fn().mockRejectedValue(new SyntaxError("Unexpected end of JSON input")),
        })
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByTitle("kb.ownership.makeTeam"))

    await waitFor(() => {
      expect(toastErrorMock.mock.calls.map((call) => call[0])).toContain("kb.ownership.failed")
    })
    expect(toastSuccessMock).not.toHaveBeenCalled()
    expect(collectionFetchCount).toBe(1)
  })

  it("keeps the synchronous 204 contract when the backend has no job queue", async () => {
    const seenUrls: string[] = []

    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      seenUrls.push(url)

      if (url === "http://api.local/api/kb/collections" && !options) {
        return Promise.resolve(createJsonResponse(ONE_COLLECTION))
      }

      if (url === "http://api.local/api/knowledge-bases/demo/promote-team" && options?.method === "POST") {
        return Promise.resolve(createStatusResponse(204))
      }

      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    render(<KnowledgeBasePage />)

    await screen.findByText("demo")

    fireEvent.click(screen.getByTitle("kb.ownership.makeTeam"))

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledWith("kb.ownership.teamSuccess")
    })
    expect(seenUrls.some((url) => url.includes("/api/jobs/"))).toBe(false)
  })
})
