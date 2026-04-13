import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
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
    Upload: Icon,
    Globe: Icon,
    Settings: Icon,
    CheckCircle: Icon,
    Clock: Icon,
    XCircle: Icon,
    AlertCircle: Icon,
    FileText: Icon,
    Cloud: Icon,
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
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
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

vi.mock("./cloud-connect-dialog", () => ({
  CloudConnectDialog: () => null,
}))

import { KnowledgeBaseCreationDialog } from "./knowledge-base-creation-dialog"

function createJsonResponse(body: unknown, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    json: vi.fn().mockResolvedValue(body),
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
}

describe("KnowledgeBaseCreationDialog multi-file naming", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    installApiMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it("requires an explicit collection name for multiple file uploads", async () => {
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={vi.fn()} />
    )

    const fileInput = container.querySelector("#file-upload") as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(["a"], "alpha.txt", { type: "text/plain" }),
          new File(["b"], "beta.txt", { type: "text/plain" }),
        ],
      },
    })

    const submitButton = screen.getByText("kb.index.startImport")
    expect(submitButton).toBeDisabled()
    expect(screen.getByText("kb.dialog.basicInfo.multiFileRequiredHint")).toBeInTheDocument()

    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledTimes(2)
    })
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("uses the same explicit collection name for each uploaded file", async () => {
    const onSuccess = vi.fn()
    const { container } = render(
      <KnowledgeBaseCreationDialog open={true} onOpenChange={vi.fn()} onSuccess={onSuccess} />
    )

    fireEvent.change(screen.getByLabelText("kb.dialog.basicInfo.nameLabel"), {
      target: { value: "team-docs" },
    })

    const fileInput = container.querySelector("#file-upload") as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [
          new File(["a"], "alpha.txt", { type: "text/plain" }),
          new File(["b"], "beta.txt", { type: "text/plain" }),
        ],
      },
    })

    fireEvent.click(screen.getByText("kb.index.startImport"))

    await waitFor(() => {
      const ingestCalls = apiRequestMock.mock.calls.filter(([url]) => url === "http://api.local/api/kb/ingest")
      expect(ingestCalls).toHaveLength(2)
      for (const [, options] of ingestCalls) {
        expect((options?.body as FormData).get("collection")).toBe("team-docs")
      }
    })

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(["team-docs", "team-docs"])
    })
  })
})
