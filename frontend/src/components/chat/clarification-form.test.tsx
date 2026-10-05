/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { McpApp } from "@/contexts/mcp-apps-context"
import { resolveTranslation } from "@/i18n/translations"

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

// `translate` is a mutable box so a test can swap the active locale's `t` and
// rerender, the way I18nProvider does - it changes its context value without
// remounting consumers. `identity` is kept beside it as the single definition
// the file-wide reset below restores.
const i18nMock = vi.hoisted(() => {
  const identity = (key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}:${JSON.stringify(vars)}` : key
  return { identity, translate: identity }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      i18nMock.translate(key, vars),
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
import { createClarificationSendFailure } from "./clarification-delivery"
import { suggestedClarificationValue } from "./clarification-guidance"
import type { Interaction } from "@/contexts/app-context-chat"

// Every describe in this file gets the identity translate back, so a locale
// swapped by one test cannot leak into a suite added below it.
beforeEach(() => {
  i18nMock.translate = i18nMock.identity
})

describe("ClarificationForm guided answers", () => {
  beforeEach(() => {
    // MultiSelect uses Next's automatic JSX runtime; Vitest uses classic JSX.
    vi.stubGlobal("React", React)
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    toastErrorMock.mockReset()
  })
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  const select: Interaction = {
    type: "select_one", field: "cadence", label: "Cadence", default_value: "weekly",
    options: [
      { value: "daily", label: "Daily" },
      { value: "weekly", label: "Weekly", description: "One summary each week" },
    ],
  }
  const submit = () => fireEvent.click(screen.getByRole("button", { name: "chatPage.clarification.submit" }))
  const next = () => fireEvent.click(screen.getByRole("button", { name: "chatPage.clarification.next" }))
  const previous = () => fireEvent.click(screen.getByRole("button", { name: "chatPage.clarification.previous" }))
  const defer = (field: string) => fireEvent.click(screen.getByRole("button", {
    name: name => ["notSure", "answerInstead"].some(action => name === `chatPage.clarification.${action}: ${field}`),
  }))
  const textFields = (count: number): Interaction[] => Array.from({ length: count }, (_, i) => ({
    type: "text_input", field: `q${i}`, label: `Question ${i}`, placeholder: `Answer ${i}`,
  }))

  it("keeps a wire suggestion through normalization, but submits only an explicit choice", async () => {
    const { normalizeInteractions } = await vi.importActual<typeof import("@/contexts/app-context-chat")>("@/contexts/app-context-chat")
    const onSend = vi.fn()
    render(<ClarificationForm requestId="request-1" interactions={normalizeInteractions([select])} onSend={onSend} />)
    expect(screen.getByText("chatPage.clarification.recommended")).toBeInTheDocument()
    expect(screen.getByText("One summary each week")).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Weekly" })).not.toBeChecked()
    submit()
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("radio", { name: "Daily" }))
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: Daily", [], { request_id: "request-1" }))
  })

  it("keeps large option sets as a dropdown and labels only the suggested choice", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[{ ...select, options: [...select.options!, ...[1, 2, 3, 4].map(i => ({ value: String(i), label: String(i) }))] }]} onSend={onSend} />)
    expect(screen.queryByRole("radio")).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("chatPage.clarification.selectOption"))
    fireEvent.click(screen.getByText("Weekly (chatPage.clarification.recommended)"))
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: Weekly", [], {}))
  })

  it("still shows radio choices at the five-option boundary", () => {
    render(<ClarificationForm interactions={[{ ...select, options: [...select.options!, ...[1, 2, 3].map(i => ({ value: String(i), label: String(i) }))] }]} onSend={vi.fn()} />)
    expect(screen.getAllByRole("radio")).toHaveLength(5)
    expect(screen.queryByText("chatPage.clarification.selectOption")).not.toBeInTheDocument()
  })

  it("disables radio choices while the chosen answer is being sent", () => {
    const onSend = vi.fn(() => new Promise<void>(() => {}))
    render(<ClarificationForm interactions={[select]} onSend={onSend} />)
    fireEvent.click(screen.getByRole("radio", { name: "Weekly" }))
    submit()
    for (const radio of screen.getAllByRole("radio")) expect(radio).toBeDisabled()
    fireEvent.click(screen.getByRole("radio", { name: "Daily" }))
    expect(screen.getByRole("radio", { name: "Weekly" })).toBeChecked()
    expect(onSend).toHaveBeenCalledOnce()
    expect(onSend).toHaveBeenCalledWith("Cadence: Weekly", [], {})
  })

  it("lets users adopt a numeric zero suggestion without treating it as empty", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[{ type: "number_input", field: "threshold", label: "Threshold", default_value: 0 }]} onSend={onSend} />)
    expect(screen.getByRole("spinbutton")).toHaveValue(null)
    fireEvent.click(screen.getByRole("button", { name: 'chatPage.clarification.useSuggestion:{"value":"0"}' }))
    expect(screen.getByRole("spinbutton")).toHaveValue(0)
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Threshold: 0", [], {}))
  })

  it("does not apply consent defaults or offer to skip approval controls", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[{ type: "confirm", field: "send", label: "Send to team", default_value: true }]} onSend={onSend} />)
    expect(screen.getByRole("switch")).not.toBeChecked()
    expect(screen.queryByText("chatPage.clarification.notSure")).not.toBeInTheDocument()
    expect(screen.queryByText("chatPage.clarification.recommended")).not.toBeInTheDocument()
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Send to team: chatPage.clarification.no", [], {}))
  })

  it("adopts an existing multi-select suggestion only after an explicit click", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[{ ...select, type: "select_multiple" }]} onSend={onSend} />)
    expect(screen.getByText("chatPage.clarification.selectOptions")).toBeInTheDocument()
    submit()
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: 'chatPage.clarification.useSuggestion:{"value":"Weekly"}' }))
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: Weekly", [], {}))
  })

  it("adds a multi-select suggestion without replacing existing choices and then hides it", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[{ ...select, type: "select_multiple" }]} onSend={onSend} />)
    fireEvent.click(screen.getByText("chatPage.clarification.selectOptions"))
    fireEvent.click(screen.getByText("Daily"))
    const suggestionButton = screen.getByRole("button", { name: 'chatPage.clarification.useSuggestion:{"value":"Weekly"}' })
    fireEvent.click(suggestionButton)
    expect(suggestionButton).not.toBeInTheDocument()
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: Daily, Weekly", [], {}))
  })

  it("does not open a disabled multi-select while a response is being sent", () => {
    const onSend = vi.fn(() => new Promise<void>(() => {}))
    render(<ClarificationForm interactions={[{ ...select, type: "select_multiple" }]} onSend={onSend} />)
    fireEvent.click(screen.getByRole("button", { name: 'chatPage.clarification.useSuggestion:{"value":"Weekly"}' }))
    submit()
    fireEvent.click(screen.getByText("Weekly"))
    expect(screen.queryByText("Daily")).not.toBeInTheDocument()
    expect(onSend).toHaveBeenCalledOnce()
  })

  it("reports a skipped suggestion as missing, never as the suggested answer", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[select]} onSend={onSend} />)
    fireEvent.click(screen.getByRole("radio", { name: "Weekly" }))
    defer("Cadence")
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: chatPage.clarification.deferredAnswer", [], {}))
  })

  it("can undo deferral and restore a previously chosen answer", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[select]} onSend={onSend} />)
    fireEvent.click(screen.getByRole("radio", { name: "Weekly" }))
    defer("Cadence")
    defer("Cadence")
    expect(screen.getByRole("radio", { name: "Weekly" })).toBeChecked()
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Cadence: Weekly", [], {}))
  })

  it("never submits attachments for a deferred upload", async () => {
    const onSend = vi.fn()
    const { container } = render(<ClarificationForm interactions={[{ type: "file_upload", field: "sample", label: "Sample" }]} onSend={onSend} />)
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [new File(["data"], "sample.csv")] } })
    defer("Sample")
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Sample: chatPage.clarification.deferredAnswer", [], {}))
  })

  it("pages all seven questions without sending early or losing answers on back navigation", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={textFields(7)} onSend={onSend} />)
    expect(screen.getAllByRole("textbox")).toHaveLength(3)
    expect(screen.queryByRole("button", { name: "chatPage.clarification.submit" })).not.toBeInTheDocument()
    for (let i = 0; i < 3; i++) fireEvent.change(screen.getByPlaceholderText(`Answer ${i}`), { target: { value: `value ${i}` } })
    next()
    for (let i = 3; i < 6; i++) fireEvent.change(screen.getByPlaceholderText(`Answer ${i}`), { target: { value: `value ${i}` } })
    previous()
    expect(screen.getByPlaceholderText("Answer 0")).toHaveValue("value 0")
    next()
    next()
    fireEvent.change(screen.getByPlaceholderText("Answer 6"), { target: { value: "value 6" } })
    expect(onSend).not.toHaveBeenCalled()
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    expect(onSend).toHaveBeenCalledWith(Array.from({ length: 7 }, (_, i) => `Question ${i}: value ${i}`).join("\n"), [], {})
  })

  it("requires a fresh submit activation after the last Next button", async () => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={[
      ...textFields(3), { type: "confirm", field: "approve", label: "Approve" },
    ]} onSend={onSend} />)
    fireEvent.change(screen.getByPlaceholderText("Answer 0"), { target: { value: "kept" } })
    const nextButton = screen.getByRole("button", { name: "chatPage.clarification.next" })
    nextButton.focus()
    fireEvent.click(nextButton, { detail: 1 })
    expect(screen.getByRole("switch")).not.toBeChecked()
    // A queued activation of Next must not turn into a Submit click.
    fireEvent.click(nextButton, { detail: 2 })
    expect(onSend).not.toHaveBeenCalled()
    const submitButton = screen.getByRole("button", { name: "chatPage.clarification.submit" })
    expect(submitButton).not.toBe(nextButton)
    expect(submitButton).not.toHaveFocus()
    // Also ignore the second pointer click if it targets the new button.
    fireEvent.click(submitButton, { detail: 2 })
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(submitButton, { detail: 1 })
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    expect(onSend).toHaveBeenCalledWith("Question 0: kept\nApprove: chatPage.clarification.no", [], {})
  })

  it.each(["next", "previous"])("does not skip question groups when %s is double clicked", direction => {
    const onSend = vi.fn()
    render(<ClarificationForm interactions={textFields(7)} onSend={onSend} />)
    if (direction === "previous") {
      next()
      next()
    }
    const name = `chatPage.clarification.${direction}`
    const navigationButton = screen.getByRole("button", { name })
    navigationButton.focus()
    fireEvent.click(navigationButton, { detail: 1 })
    expect(screen.getByPlaceholderText("Answer 3")).toBeInTheDocument()
    fireEvent.click(navigationButton, { detail: 2 })
    const newButton = screen.getByRole("button", { name })
    expect(newButton).not.toBe(navigationButton)
    expect(newButton).not.toHaveFocus()
    fireEvent.click(newButton, { detail: 2 })
    expect(screen.getByPlaceholderText("Answer 3")).toBeInTheDocument()
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.click(newButton, { detail: 1 })
    expect(screen.getByPlaceholderText(direction === "next" ? "Answer 6" : "Answer 0")).toBeInTheDocument()
  })

  it("retains answers, deferrals, and files across pages after a rejected send", async () => {
    const onSend = vi.fn().mockRejectedValueOnce(createClarificationSendFailure("Not sent", "not_sent")).mockResolvedValueOnce(undefined)
    const file = new File(["data"], "sample.csv")
    const { container } = render(<ClarificationForm requestId="retry-request" interactions={[
      { type: "file_upload", field: "sample", label: "Sample" }, ...textFields(3),
    ]} onSend={onSend} />)
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } })
    fireEvent.change(screen.getByPlaceholderText("Answer 0"), { target: { value: "preserved" } })
    defer("Question 1")
    next()
    fireEvent.change(screen.getByPlaceholderText("Answer 2"), { target: { value: "last" } })
    submit()
    await screen.findByRole("alert")
    previous()
    expect(screen.getByText("sample.csv")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Answer 0")).toHaveValue("preserved")
    expect(screen.getByText("chatPage.clarification.deferredAnswer")).toBeInTheDocument()
    next()
    submit()
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(2))
    expect(onSend.mock.calls[1]).toEqual(onSend.mock.calls[0])
    expect(onSend.mock.calls[1]).toEqual([
      "Question 0: preserved\nQuestion 1: chatPage.clarification.deferredAnswer\nQuestion 2: last", [file], { request_id: "retry-request" },
    ])
  })

  it("starts a new request on page one without reusing deferrals or answers", () => {
    const interactions = textFields(4)
    const { rerender } = render(<ClarificationForm requestId="old" interactions={interactions} onSend={vi.fn()} />)
    defer("Question 0")
    next()
    rerender(<ClarificationForm requestId="new" interactions={interactions} onSend={vi.fn()} />)
    expect(screen.getByPlaceholderText("Answer 0")).toHaveValue("")
    expect(screen.queryByText("chatPage.clarification.deferredAnswer")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "chatPage.clarification.previous" })).not.toBeInTheDocument()
  })

  it("does not count disabled uploads as questions or expose hidden suggested actions", () => {
    render(<ClarificationForm filesDisabled interactions={[
      { type: "file_upload", field: "file", label: "File" }, ...textFields(2),
      { type: "action_cards", field: "source", label: "Source", default_value: "upload", options: [
        { value: "upload", label: "Upload", action_type: "upload" }, { value: "paste", label: "Paste", action_type: "none" },
      ] },
    ]} onSend={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "chatPage.clarification.next" })).not.toBeInTheDocument()
    expect(screen.queryByText("chatPage.clarification.recommended")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Paste" })).toHaveAttribute("aria-pressed", "false")
  })

  it("translates recommendation and deferral controls with a live locale change", () => {
    i18nMock.translate = (key, vars) => resolveTranslation("en", key as Parameters<typeof resolveTranslation>[1], vars)
    const { rerender } = render(<ClarificationForm interactions={[select]} onSend={vi.fn()} />)
    expect(screen.getByText("Recommended")).toBeInTheDocument()
    i18nMock.translate = (key, vars) => resolveTranslation("zh", key as Parameters<typeof resolveTranslation>[1], vars)
    rerender(<ClarificationForm interactions={[select]} onSend={vi.fn()} />)
    expect(screen.getByText("推荐")).toBeInTheDocument()
    expect(screen.getByText("不确定 / 暂不回答")).toBeInTheDocument()
  })

  it.each(["en", "zh"] as const)("keeps the visible defer action in its accessible name in %s", locale => {
    i18nMock.translate = (key, vars) => resolveTranslation(locale, key as Parameters<typeof resolveTranslation>[1], vars)
    render(<ClarificationForm interactions={[select]} onSend={vi.fn()} />)
    const deferText = resolveTranslation(locale, "chatPage.clarification.notSure")
    const resumeText = resolveTranslation(locale, "chatPage.clarification.answerInstead")
    const button = screen.getByRole("button", { name: `${deferText}: Cadence` })
    expect(button).toHaveTextContent(deferText)
    expect(button).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(button)
    expect(button).toHaveAccessibleName(`${resumeText}: Cadence`)
    expect(button).toHaveTextContent(resumeText)
    expect(button).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(button)
    expect(button).toHaveAccessibleName(`${deferText}: Cadence`)
    expect(button).toHaveAttribute("aria-pressed", "false")
  })

  it.each([
    { type: "number_input", default_value: Infinity },
    { type: "number_input", default_value: 0, min: 1 },
    { type: "number_input", default_value: 10, max: 5 },
    { type: "number_input", default_value: false },
    { type: "select_one", default_value: "missing", options: select.options },
    { type: "confirm", default_value: true },
    { type: "text_input", default_value: " " },
  ] as Partial<Interaction>[]) ("ignores unusable or unsafe suggestions: %s", input => {
    expect(suggestedClarificationValue({ field: "f", label: "F", ...input } as Interaction)).toBeUndefined()
  })
})

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

  it("binds a connect_apps skip to the rendered interaction request", async () => {
    render(
      <ClarificationForm
        interactions={[CONNECT_APPS_INTERACTION]}
        requestId="inputreq_0011223344556677889900aabbccddee"
      />,
    )

    fireEvent.click(screen.getByText("chatPage.clarification.connectApps.skip"))

    await waitFor(() => {
      expect(appContextMock.sendMessage).toHaveBeenCalledWith(
        "chatPage.clarification.connectApps.skip",
        {
          force: true,
          metadata: { request_id: "inputreq_0011223344556677889900aabbccddee" },
        },
        [],
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

  it("resolves the connect_apps field label live in the mixed-list branch too, not the persisted hire-time label", () => {
    // The singleton isConnectAppsOnly header was fixed to call t() live, but
    // that branch is skipped entirely for a mixed list (isConnectAppsOnly is
    // false) - the per-field label above renderField's switch is a second,
    // separate render path that has to make the same fix independently.
    render(
      <ClarificationForm
        interactions={[
          CONNECT_APPS_INTERACTION,
          { type: "text_input", field: "note", label: "Note" },
        ]}
        onSend={vi.fn()}
      />,
    )

    expect(
      screen.getByText("chatPage.clarification.connectApps.title"),
    ).toBeInTheDocument()
    expect(screen.queryByText(CONNECT_APPS_INTERACTION.label)).not.toBeInTheDocument()
    // An ordinary field's own persisted label is untouched by this.
    expect(screen.getByText("Note:")).toBeInTheDocument()
  })
})

describe("ClarificationForm delivery failures", () => {
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

  const deliveryError = (
    message: string,
    disposition: string,
    userFacing = false,
    errorCode: string | null = null,
  ) => Object.assign(new Error(message), { disposition, userFacing, errorCode })

  const submitAnswer = async (onSend: ReturnType<typeof vi.fn>) => {
    render(
      <ClarificationForm
        interactions={[{ type: "text_input" as const, field: "city", label: "City" }]}
        onSend={onSend}
      />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
  }

  it("localizes a coded backend rejection instead of trusting its prose", async () => {
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "checkpoint row includes storage-key=secret",
      "rejected",
      true,
      "task_checkpoint_unreadable",
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "clientErrors.taskCheckpointUnreadable",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent(
      "clientErrors.taskCheckpointUnreadable",
    )
    // The hint lives in the alert too, not only in the toast - without this
    // the inline hint could be deleted with every test still green.
    expect(alert).toHaveTextContent("chatPage.clarification.sendNotSent")
    expect(alert).not.toHaveTextContent("storage-key=secret")
  })

  it("keeps the form submittable after a failure that never reached the agent", async () => {
    const onSend = vi.fn().mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent", true),
    )

    await submitAnswer(onSend)

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(
      "Durable storage is temporarily unavailable",
      { description: "chatPage.clarification.sendNotSent" },
    ))
    const submit = screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })
    expect(submit).toBeEnabled()
    expect(screen.getByRole("textbox")).toHaveValue("Beijing")
  })

  it("warns before a resubmit when the delivery outcome is unknown", async () => {
    // No resubmit guard exists in this component (a retry mints a fresh
    // client message id today), so the copy must warn - not promise safety.
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "The task is busy applying an earlier answer.",
      "outcome_unknown",
      true,
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "The task is busy applying an earlier answer.",
        { description: "chatPage.clarification.sendOutcomeUnknown" },
      )
    })
    // Advisory only: the button stays enabled, exactly as it does today.
    expect(screen.getByRole("button", {
      name: "chatPage.clarification.submit",
    })).toBeEnabled()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "chatPage.clarification.sendOutcomeUnknown",
    )
  })

  it("keeps connection plumbing diagnostics away from the visitor", async () => {
    const onSend = vi.fn().mockRejectedValue(deliveryError(
      "Message not sent: the connection changed before delivery.",
      "not_sent",
    ))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "chatPage.clarification.sendError",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
    expect(await screen.findByRole("alert")).not.toHaveTextContent(
      "the connection changed before delivery",
    )
  })

  it("falls back to the generic string when the failure carries no reason", async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("   "))

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "chatPage.clarification.sendError",
        undefined,
      )
    })
  })

  it("clears the failure once the visitor edits an answer", async () => {
    const onSend = vi.fn().mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent", true),
    )

    await submitAnswer(onSend)

    await screen.findByRole("alert")
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Shanghai" } })
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
  })

  it("reads the reason off a failure that is not an Error instance", async () => {
    // `onSend` belongs to arbitrary builder callbacks (#1485), so a rejection
    // carrying the contract's fields need not be an `Error` subclass - the
    // disposition is probed structurally and the reason has to match.
    const onSend = vi.fn().mockRejectedValue({
      message: "A previous guidance message is still being applied.",
      disposition: "rejected",
      userFacing: true,
    })

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "A previous guidance message is still being applied.",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
  })

  it("never shows the builder's internal diagnostic to the visitor", async () => {
    // agent-builder-chat.tsx's onSendInteraction throws this exact failure as
    // a developer diagnostic (#1485) - it must never reach the visitor.
    const onSend = vi.fn().mockRejectedValue(
      createClarificationSendFailure("Failed to send interaction", "not_sent"),
    )

    await submitAnswer(onSend)

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "chatPage.clarification.sendError",
        { description: "chatPage.clarification.sendNotSent" },
      )
    })
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("chatPage.clarification.sendError")
    expect(alert).not.toHaveTextContent("Failed to send interaction")
  })

  it("re-renders the visible failure in the new locale", async () => {
    // I18nProvider swaps its context value on a locale change without
    // remounting consumers, so an alert holding pre-translated strings would
    // keep showing the previous language until it is cleared.
    const onSend = vi.fn().mockRejectedValue(
      deliveryError("Durable storage is temporarily unavailable", "not_sent", true),
    )
    const interactions = [{ type: "text_input" as const, field: "city", label: "City" }]
    const { rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("chatPage.clarification.sendNotSent")

    i18nMock.translate = (key: string) => `zh:${key}`
    rerender(<ClarificationForm interactions={interactions} onSend={onSend} />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "zh:chatPage.clarification.sendNotSent",
    )
    // The backend's own reason is not ours to translate - it passes through.
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Durable storage is temporarily unavailable",
    )
  })

  it("re-renders the generic fallback message in the new locale", async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("   "))
    const interactions = [{ type: "text_input" as const, field: "city", label: "City" }]
    const { rerender } = render(
      <ClarificationForm interactions={interactions} onSend={onSend} />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "chatPage.clarification.sendError",
    )

    i18nMock.translate = (key: string) => `zh:${key}`
    rerender(<ClarificationForm interactions={interactions} onSend={onSend} />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "zh:chatPage.clarification.sendError",
    )
  })

  it("uses hint keys that resolve in both locale trees", () => {
    // The component tests stub t() as identity, so they pin the key strings
    // only against themselves. This binds them to the real trees: a typo'd
    // key would fall back to itself instead of a translated sentence.
    for (const key of [
      "chatPage.clarification.sendNotSent",
      "chatPage.clarification.sendOutcomeUnknown",
    ] as const) {
      expect(resolveTranslation("en", key)).not.toBe(key)
      expect(resolveTranslation("zh", key)).not.toBe(key)
    }
  })

  it("clears a previous round's failure when the form is asked again", async () => {
    // The live turn render path keeps one component instance across
    // clarification rounds, so a stale round-1 alert would sit on top of
    // round 2's question.
    appContextMock.sendMessage.mockRejectedValue(deliveryError(
      "Durable storage is temporarily unavailable",
      "not_sent",
      true,
    ))
    const interactions = [{ type: "text_input" as const, field: "city", label: "City" }]
    const { rerender } = render(
      <ClarificationForm interactions={interactions} active />,
    )
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
    await screen.findByRole("alert")

    rerender(<ClarificationForm interactions={interactions} active={false} />)
    rerender(<ClarificationForm interactions={interactions} active />)

    expect(screen.queryByRole("alert")).toBeNull()
  })
})

describe("ClarificationForm interaction identity", () => {
  afterEach(() => {
    cleanup()
  })

  it("resets the reused form and ignores completion from the previous request", async () => {
    let resolveR1!: () => void
    const onSend = vi.fn(() => new Promise<void>((resolve) => { resolveR1 = resolve }))
    const form = (requestId: string, messageId?: string) => (
      <ClarificationForm
        interactions={[{ type: "text_input", field: "city", label: "City" }]}
        requestId={requestId}
        messageId={messageId}
        onSend={onSend}
      />
    )
    const { container, rerender } = render(form("inputreq_r1"))

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Sydney" } })
    expect(screen.getByRole("textbox")).toHaveValue("Sydney")

    rerender(form("inputreq_r1", "unrelated-rerender"))
    expect(screen.getByRole("textbox")).toHaveValue("Sydney")
    fireEvent.click(screen.getByRole("button", { name: "chatPage.clarification.submit" }))

    rerender(form("inputreq_r2"))

    expect(screen.getByRole("textbox")).toHaveValue("")
    expect(screen.getByRole("button", { name: "chatPage.clarification.submit" })).toBeEnabled()
    expect(container.querySelector('[aria-expanded="true"]')).not.toBeNull()
    expect(screen.queryByRole("alert")).toBeNull()

    await act(async () => resolveR1())

    expect(screen.getByRole("textbox")).toHaveValue("")
    expect(screen.getByRole("button", { name: "chatPage.clarification.submit" })).toBeEnabled()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("submits the request id bound to this rendered form", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined)
    render(
      <ClarificationForm
        interactions={[{ type: "text_input", field: "city", label: "City" }]}
        requestId="inputreq_0011223344556677889900aabbccddee"
        onSend={onSend}
      />,
    )

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Sydney" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(
      "City: Sydney",
      [],
      { request_id: "inputreq_0011223344556677889900aabbccddee" },
    ))
  })
})

describe("ClarificationForm blank option filtering", () => {
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

  // The option label is rendered into a <span> on both branches this suite
  // covers: select_one's dropdown option (ui/select.tsx, "font-medium
  // truncate") and action_cards' card (clarification-form.tsx, "font-medium
  // text-sm text-foreground"). action_cards renders each card as a <div
  // onClick>, not a <button> -- getAllByRole("button") returns an empty
  // array for it regardless of whether the blank-option filter is
  // trim-aware, so it cannot be used to detect a blank card here. No other
  // <span> in this component's default render (no files staged, no
  // description set on any option) is ever blank, so a <span> with blank
  // text content can only be a surviving blank option.
  const blankOptionSpans = (container: HTMLElement) =>
    Array.from(container.querySelectorAll("span")).filter(
      (el) => el.textContent !== "" && el.textContent?.trim() === "",
    )

  it("a select_one interaction whose only option is blank shows no options", () => {
    const interactions = [
      {
        type: "select_one" as const,
        field: "choice",
        label: "Choice",
        options: [{ label: "   ", value: "   " }],
      },
    ]
    render(<ClarificationForm interactions={interactions} onSend={vi.fn()} />)

    fireEvent.click(screen.getByText("chatPage.clarification.selectOption"))

    expect(screen.getByText("common.noOptions")).toBeInTheDocument()
  })

  it("a select_one interaction drops an option whose label alone is blank", () => {
    // label and value are independent halves of the same filter
    // (opt.value.trim() !== "" && opt.label.trim() !== ""); a blank value
    // makes this option non-blank in isolation, so a regression that only
    // reverted the label half back to a truthiness check would let this
    // option survive even though a case that leaves both halves blank at
    // once would still be dropped by the (unregressed) value half alone.
    const interactions = [
      {
        type: "select_one" as const,
        field: "choice",
        label: "Choice",
        options: [
          { label: "Import", value: "import" },
          { label: "   ", value: "blank-label-only" },
        ],
      },
    ]
    const { container } = render(
      <ClarificationForm interactions={interactions} onSend={vi.fn()} />,
    )

    expect(screen.getByText("Import")).toBeInTheDocument()
    expect(blankOptionSpans(container)).toHaveLength(0)
  })

  it("a select_one interaction drops an option whose value alone is blank", () => {
    // Mirrors the case above for the other half of the same filter: a
    // non-blank label makes this option non-blank in isolation, so a
    // regression that only reverted the value half back to a truthiness
    // check would let this option survive.
    const interactions = [
      {
        type: "select_one" as const,
        field: "choice",
        label: "Choice",
        options: [
          { label: "Import", value: "import" },
          { label: "Blank value only", value: "   " },
        ],
      },
    ]
    render(<ClarificationForm interactions={interactions} onSend={vi.fn()} />)

    expect(screen.getByRole("radio", { name: "Import" })).toBeInTheDocument()
    expect(screen.queryByText("Blank value only")).not.toBeInTheDocument()
  })

  it("an action_cards interaction keeps a good option and drops a blank one", () => {
    // Mirrors the shape the agent-builder skill instructs the model to use
    // for this interaction type: a mix of a real choice and, in the failure
    // case this fix targets, a blank one.
    const interactions = [
      {
        type: "action_cards" as const,
        field: "source",
        label: "Source",
        options: [
          { label: "Import", value: "import" },
          { label: "   ", value: "   " },
        ],
      },
    ]
    const { container } = render(
      <ClarificationForm interactions={interactions} onSend={vi.fn()} />,
    )

    // No dropdown to open: action_cards renders its cards directly inside
    // CollapsibleContent, which is open by default (active defaults to true).
    expect(screen.getByText("Import")).toBeInTheDocument()
    expect(blankOptionSpans(container)).toHaveLength(0)
  })

  it("renders options from a legacy message that still carries actions", () => {
    // The backend normalizer now strips a well-formed interaction down to
    // one options carrier and never emits actions, but that only applies to
    // new payloads. Rows persisted before that change, and anything from
    // Agent Builder's self-parsed chat response (which never reaches the
    // backend normalizer at all), can still carry only actions with no
    // options key -- this component's own fallback (rawOptions above) is
    // the sole thing still rendering those.
    const interactions = [
      {
        type: "action_cards" as const,
        field: "source",
        label: "Source",
        actions: [
          { label: "Import", value: "import" },
          { label: "   ", value: "   " },
        ],
      },
    ]
    const { container } = render(
      <ClarificationForm interactions={interactions} onSend={vi.fn()} />,
    )

    expect(screen.getByText("Import")).toBeInTheDocument()
    expect(blankOptionSpans(container)).toHaveLength(0)
  })
})
