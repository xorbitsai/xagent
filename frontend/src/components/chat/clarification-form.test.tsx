/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const appContextMock = vi.hoisted(() => ({
  dispatch: vi.fn(),
  filesDisabled: false,
  providerAvailable: true,
  sendMessage: vi.fn(),
}))
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => {
    if (!appContextMock.providerAvailable) {
      throw new Error("App provider is unavailable")
    }
    return appContextMock
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

import { ClarificationForm } from "./clarification-form"

describe("ClarificationForm Session file capability", () => {
  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("removes direct file upload UI and drops staged files after files are disabled", async () => {
    const onSend = vi.fn()
    const interactions = [
      {
        type: "file_upload" as const,
        field: "evidence",
        label: "Evidence",
      },
      {
        type: "text_input" as const,
        field: "note",
        label: "Note",
        placeholder: "Add a note",
      },
    ]
    const { container, rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["secret"], "secret.txt", { type: "text/plain" })],
      },
    })
    expect(screen.getByText("secret.txt")).toBeInTheDocument()

    appContextMock.filesDisabled = true
    rerender(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(screen.queryByText("secret.txt")).not.toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText("Add a note"), {
      target: { value: "Continue without a file" },
    })
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "Note: Continue without a file",
        [],
        {},
      )
    })
  })

  it("removes action-card upload choices and drops their staged files", async () => {
    const onSend = vi.fn()
    const interactions = [
      {
        type: "action_cards" as const,
        field: "source",
        label: "Source",
        options: [
          {
            label: "Upload a file",
            value: "upload",
            action_type: "upload",
          },
          {
            label: "Skip upload",
            value: "skip_upload",
            action_type: "skip",
          },
        ],
      },
    ]
    const { container, rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    fireEvent.click(screen.getByText("Upload a file"))
    const fileInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["secret"], "secret.csv", { type: "text/csv" })],
      },
    })
    expect(screen.getByText("secret.csv")).toBeInTheDocument()

    appContextMock.filesDisabled = true
    rerender(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )

    expect(screen.queryByText("Upload a file")).not.toBeInTheDocument()
    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(screen.queryByText("secret.csv")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("Skip upload"))
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith("Source: Skip upload", [], {})
    })
  })

  it("preserves file submission for legacy contexts where files are enabled", async () => {
    const onSend = vi.fn()
    const file = new File(["report"], "report.txt", { type: "text/plain" })
    const { container } = render(
      <ClarificationForm
        interactions={[
          {
            type: "file_upload",
            field: "evidence",
            label: "Evidence",
          },
        ]}
        onSend={onSend}
      />,
    )

    fireEvent.change(
      container.querySelector<HTMLInputElement>('input[type="file"]')!,
      { target: { files: [file] } },
    )
    fireEvent.click(
      screen.getByRole("button", {
        name: "chatPage.clarification.submit",
      }),
    )

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "chatPage.clarification.uploadedFiles",
        [file],
        {},
      )
    })
  })

  it("fails closed for file uploads when no app provider or override is available", () => {
    appContextMock.providerAvailable = false
    const { container } = render(
      <ClarificationForm
        interactions={[{ type: "file_upload", field: "evidence", label: "Evidence" }]}
        onSend={vi.fn()}
      />,
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
  })

  it("allows builder callers to explicitly enable file uploads without an app provider", () => {
    appContextMock.providerAvailable = false
    const { container } = render(
      <ClarificationForm
        filesDisabled={false}
        interactions={[{ type: "file_upload", field: "evidence", label: "Evidence" }]}
        onSend={vi.fn()}
      />,
    )

    expect(container.querySelector('input[type="file"]')).not.toBeNull()
  })
})
