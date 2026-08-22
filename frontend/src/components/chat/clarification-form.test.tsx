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
import type { McpApp } from "@/contexts/mcp-apps-context"

const appContextMock = vi.hoisted(() => ({
  dispatch: vi.fn(),
  filesDisabled: false,
  providerAvailable: true,
  sendMessage: vi.fn(),
}))
const toastErrorMock = vi.hoisted(() => vi.fn())
const mcpAppsMock = vi.hoisted(() => ({
  apps: [] as McpApp[],
  refresh: vi.fn(),
}))

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
    t: (key: string, vars?: Record<string, string | number>) =>
      vars ? `${key}:${JSON.stringify(vars)}` : key,
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

// Only exercised by connect_apps interactions (below); every other test in
// this file never mounts ConnectAppsField, so these mocks are inert for them.
vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => mcpAppsMock,
}))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "test-token" }),
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

describe("ClarificationForm connect_apps interaction", () => {
  const CONNECT_APPS_INTERACTION = {
    type: "connect_apps" as const,
    field: "connect_apps",
    label: "Connect your apps",
    apps: ["Gmail"],
  }

  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
    mcpAppsMock.apps = [
      {
        id: "gmail",
        name: "Gmail",
        description: "",
        icon: "",
        users: "",
        transport: "builtin",
        provider: "google",
        category: "Communication",
        is_connected: false,
      },
    ]
    mcpAppsMock.refresh.mockReset().mockResolvedValue(undefined)
  })

  afterEach(() => {
    cleanup()
  })

  it("renders open by default even when active=false, unlike every other interaction type", () => {
    // Simulates the AI Team Marketplace Hire flow: the interaction is seeded
    // onto a task that never enters waiting_for_user, so `active` is false
    // from the very first render - a plain question field would stay
    // collapsed/disabled forever, but connect_apps must not.
    render(
      <ClarificationForm
        interactions={[CONNECT_APPS_INTERACTION]}
        active={false}
        onSend={vi.fn()}
      />,
    )

    expect(screen.getByText("Gmail")).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: 'chatPage.clarification.connectApps.continueWith:{"provider":"Gmail"}',
      }),
    ).toBeInTheDocument()
  })

  it("shows the live-translated connectApps title in the header instead of the generic 'Ask User' title, ignoring the persisted label", () => {
    // CONNECT_APPS_INTERACTION.label ("Connect your apps") stands in for the
    // DB-persisted, hire-time-translated string (see hire-agent.ts's
    // buildConnectAppsInteraction) - the header must not use it, or a locale
    // switch after hiring would leave it frozen in the original language.
    render(
      <ClarificationForm interactions={[CONNECT_APPS_INTERACTION]} onSend={vi.fn()} />,
    )

    expect(screen.getByText("chatPage.clarification.connectApps.title")).toBeInTheDocument()
    expect(screen.queryByText("chatPage.clarification.title")).not.toBeInTheDocument()
    expect(screen.queryByText(CONNECT_APPS_INTERACTION.label)).not.toBeInTheDocument()
  })

  it("does not render the generic Submit button - connecting happens per-provider, not via a form submit", () => {
    render(
      <ClarificationForm interactions={[CONNECT_APPS_INTERACTION]} onSend={vi.fn()} />,
    )

    expect(
      screen.queryByRole("button", { name: "chatPage.clarification.submit" }),
    ).not.toBeInTheDocument()
  })

  it("sends a skip acknowledgement message when 'I'll do this later' is clicked", async () => {
    const onSend = vi.fn()
    render(
      <ClarificationForm interactions={[CONNECT_APPS_INTERACTION]} onSend={onSend} />,
    )

    fireEvent.click(screen.getByText("chatPage.clarification.connectApps.skip"))

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "chatPage.clarification.connectApps.skip",
        [],
        {},
      )
    })
  })

  it("renders the real connect_apps widget instead of an 'unsupported type' error when mixed into a list with another interaction type", () => {
    // Not producible by any seeder today (see LIVE_WIDGET_TYPES's comment in
    // clarification-form.tsx), but nothing rules it out - isConnectAppsOnly
    // is false here since the list isn't every() connect_apps, so this must
    // go through renderField's normal per-field switch instead of the
    // dedicated isConnectAppsOnly branch.
    render(
      <ClarificationForm
        interactions={[
          CONNECT_APPS_INTERACTION,
          { type: "text_input", field: "note", label: "Note" },
        ]}
        onSend={vi.fn()}
      />,
    )

    expect(screen.getByText("Gmail")).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: 'chatPage.clarification.connectApps.continueWith:{"provider":"Gmail"}',
      }),
    ).toBeInTheDocument()
    expect(
      screen.queryByText('chatPage.clarification.unsupportedType:{"type":"connect_apps"}'),
    ).not.toBeInTheDocument()
  })
})
