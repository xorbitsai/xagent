import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Check: Icon,
    ChevronRight: Icon,
    Folder: Icon,
    File: Icon,
    Loader2: Icon,
    Search: Icon,
    RefreshCw: Icon,
    Trash2: Icon,
  }
})

vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button {...props}>{children}</button>
  ),
}))

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => (
    <label {...props}>{children}</label>
  ),
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("@/components/ui/select", () => ({
  Select: ({
    options,
    onValueChange,
  }: {
    options: Array<{ value: string; label: string }>
    onValueChange: (value: string) => void
  }) => (
    <div>
      {options.map(option => (
        <button
          key={option.value}
          data-testid={`select-${option.value}`}
          onClick={() => onValueChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  ),
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: () => null,
}))

import { CloudConnectDialog } from "./cloud-connect-dialog"

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue(body),
  }
}

describe("CloudConnectDialog", () => {
  beforeEach(() => {
    vi.stubGlobal("React", React)
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    translateMock.mockClear()
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/cloud/accounts?provider=google-drive") {
        return Promise.resolve(jsonResponse([
          { id: 1, provider: "google-drive", email: "user@example.com", created_at: "now" },
        ]))
      }
      if (url === "http://api.local/api/cloud/google-drive/files?folder_id=root&account_id=1") {
        return Promise.resolve(jsonResponse(
          Array.from({ length: 6 }, (_, index) => ({
            id: `file-${index + 1}`,
            name: `File ${index + 1}.pdf`,
            type: "file",
            ...(index === 0 ? { resourceKey: "resource-key-1" } : {}),
          }))
        ))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it("preserves a Drive resource key through file selection", async () => {
    const onConfirm = vi.fn()
    const onOpenChange = vi.fn()

    render(
      <CloudConnectDialog
        open={true}
        onOpenChange={onOpenChange}
        provider={{
          id: "google-drive",
          name: "Google Drive",
          hasDrives: false,
          authPath: "google",
          logo: "drive",
        }}
        onConfirm={onConfirm}
      />
    )

    fireEvent.click(await screen.findByTestId("select-user@example.com"))
    fireEvent.click(await screen.findByText("File 1.pdf"))
    fireEvent.click(screen.getByText("kb.dialog.cloudConnect.select.confirm"))

    expect(onConfirm).toHaveBeenCalledWith([
      expect.objectContaining({
        id: "file-1",
        resourceKey: "resource-key-1",
      }),
    ])
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("prevents selecting more than five files", async () => {
    render(
      <CloudConnectDialog
        open={true}
        onOpenChange={vi.fn()}
        provider={{
          id: "google-drive",
          name: "Google Drive",
          hasDrives: false,
          authPath: "google",
          logo: "drive",
        }}
        onConfirm={vi.fn()}
      />
    )

    fireEvent.click(await screen.findByTestId("select-user@example.com"))
    await screen.findByText("File 6.pdf")

    for (let index = 1; index <= 6; index += 1) {
      fireEvent.click(screen.getByText(`File ${index}.pdf`))
    }

    expect(translateMock).toHaveBeenCalledWith(
      "kb.dialog.cloudConnect.selectedFiles.limitReached",
      { count: 5 },
    )
    expect(toastErrorMock).toHaveBeenCalledWith(
      "kb.dialog.cloudConnect.selectedFiles.limitReached"
    )
    await waitFor(() => {
      expect(screen.getAllByText(/^File [1-5]\.pdf$/)).toHaveLength(10)
    })
    expect(screen.getAllByText("File 6.pdf")).toHaveLength(1)
  })
})
