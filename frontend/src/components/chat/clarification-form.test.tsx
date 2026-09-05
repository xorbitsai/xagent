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
  state: {
    commandOutcomes: {} as Record<string, unknown>,
    clarificationSubmissions: {} as Record<string, unknown>,
  },
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

// Every describe in this file gets the identity translate back, so a locale
// swapped by one test cannot leak into a suite added below it.
beforeEach(() => {
  i18nMock.translate = i18nMock.identity
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

    fireEvent.click(screen.getByText("chatPage.clarification.selectOption"))

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
    const { container } = render(
      <ClarificationForm interactions={interactions} onSend={vi.fn()} />,
    )

    fireEvent.click(screen.getByText("chatPage.clarification.selectOption"))

    expect(screen.getByText("Import")).toBeInTheDocument()
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

describe("ClarificationForm terminal command outcomes", () => {
  // Issue #1500: after a reply is durably accepted, its command can still
  // reach a terminal disposition before a turn is established. Whether the
  // form may invite a resend is decided by the structured outcome the
  // backend broadcasts for that exact command, never by task state alone.
  // The accountable submission lives in context keyed by request id, so the
  // gate survives the submitting component instance being replaced.
  beforeEach(() => {
    appContextMock.dispatch.mockReset()
    appContextMock.filesDisabled = false
    appContextMock.providerAvailable = true
    appContextMock.sendMessage.mockReset()
    appContextMock.sendMessage.mockResolvedValue(undefined)
    appContextMock.state = { commandOutcomes: {}, clarificationSubmissions: {} }
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    appContextMock.state = { commandOutcomes: {}, clarificationSubmissions: {} }
    cleanup()
  })

  const form = (active: boolean, requestId = "inputreq_r1") => (
    <ClarificationForm
      interactions={[{ type: "text_input" as const, field: "city", label: "City" }]}
      requestId={requestId}
      active={active}
    />
  )

  const submitButton = () =>
    screen.queryByRole("button", { name: "chatPage.clarification.submit" })

  // Drives one accepted submission and mirrors the RECORD dispatch into the
  // mock context state, the way the real reducer would.
  const submitAccepted = async (requestId = "inputreq_r1") => {
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )
    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
    // Accepted submissions collapse the form.
    await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull())
    const record = appContextMock.dispatch.mock.calls
      .map(([action]) => action)
      .find((action) => action?.type === "RECORD_CLARIFICATION_SUBMISSION")
    expect(record).toEqual({
      type: "RECORD_CLARIFICATION_SUBMISSION",
      payload: { requestId, commandId: expect.any(String), accepted: true },
    })
    // The recorded command id is the client message id the delivery used.
    const config = appContextMock.sendMessage.mock.calls[0][1] as {
      clientMessageId?: string
    }
    expect(record.payload.commandId).toBe(config.clientMessageId)
    appContextMock.state = {
      ...appContextMock.state,
      clarificationSubmissions: {
        [requestId]: { commandId: record.payload.commandId, accepted: true },
      },
    }
    return record.payload.commandId as string
  }

  const withOutcome = (commandId: string, resendSafe: boolean) => {
    appContextMock.state = {
      ...appContextMock.state,
      commandOutcomes: {
        [commandId]: {
          outcome: "failed",
          resendSafe,
          messageCode: "task_command_deferred",
        },
      },
    }
  }

  it("reactivates the form and preserves the draft when the outcome proves retry safe", async () => {
    const { rerender } = render(form(true))
    const commandId = await submitAccepted()

    rerender(form(false))
    withOutcome(commandId, true)
    rerender(form(true))

    await waitFor(() => expect(submitButton()).toBeEnabled())
    expect(screen.getByRole("textbox")).toHaveValue("Beijing")
    expect(
      screen.getByText("chatPage.clarification.replyNotApplied"),
    ).toBeInTheDocument()
    // The record is consumed so the resend is armed exactly once.
    expect(appContextMock.dispatch).toHaveBeenCalledWith({
      type: "CLEAR_CLARIFICATION_SUBMISSION",
      payload: { requestId: "inputreq_r1" },
    })
  })

  it("keeps the form locked and surfaces the ambiguity when the outcome is not proven safe", async () => {
    const { rerender } = render(form(true))
    const commandId = await submitAccepted()

    rerender(form(false))
    withOutcome(commandId, false)
    rerender(form(true))

    // The notice is visible, the draft is intact, and nothing invites a
    // duplicate submission of the accepted reply.
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("chatPage.clarification.replyOutcomeUnknown")
    expect(screen.getByRole("textbox")).toHaveValue("Beijing")
    expect(submitButton()).toBeDisabled()
  })

  it("does not reactivate while the accepted reply has no terminal outcome yet", async () => {
    const { rerender } = render(form(true))
    await submitAccepted()

    rerender(form(false))
    rerender(form(true))

    // No outcome means the command may still be in flight: the form stays
    // collapsed instead of inviting a duplicate.
    expect(screen.queryByRole("textbox")).toBeNull()
    expect(submitButton()).toBeNull()
  })

  it("is not reopened by a late terminal event after a turn is established", async () => {
    const { rerender } = render(form(true))
    const commandId = await submitAccepted()

    rerender(form(false))
    // The turn was established, the task is running, and only then does a
    // stale resend-safe terminal frame arrive: with the task not waiting,
    // nothing may reopen the submitted form.
    withOutcome(commandId, true)
    rerender(form(false))

    expect(screen.queryByRole("textbox")).toBeNull()
    expect(submitButton()).toBeNull()
  })

  it("gates a fresh component instance for a round another instance submitted", async () => {
    // The virtual waiting message and the persisted timeline message render
    // the same round in different component instances; replacing the
    // submitting instance must not drop the gate.
    render(form(true))
    const commandId = await submitAccepted()
    cleanup()

    withOutcome(commandId, false)
    render(form(true))

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("chatPage.clarification.replyOutcomeUnknown")
    expect(submitButton()).toBeDisabled()
  })

  it("locks a fresh component instance while the accepted reply is still in flight", async () => {
    // A freshly mounted instance starts with isSubmitted=false, so without
    // the in-flight lock it would offer Submit for a round whose durably
    // accepted reply has not reached a terminal outcome yet (surfaced by
    // review on #2126).
    render(form(true))
    await submitAccepted()
    cleanup()

    render(form(true))

    expect(submitButton()).toBeDisabled()
  })

  it("does not lock a fresh instance for an unconfirmed ack-timeout delivery", async () => {
    appContextMock.state = {
      commandOutcomes: {},
      clarificationSubmissions: {
        inputreq_r1: { commandId: "maybe-sent", accepted: false },
      },
    }
    render(form(true))

    // The reply may never have been accepted at all: the composer keeps its
    // advisory retry until a terminal outcome for the command arrives.
    await waitFor(() => expect(submitButton()).toBeEnabled())
  })

  it("records the submission when the delivery outcome is unknown", async () => {
    appContextMock.sendMessage.mockRejectedValue(Object.assign(
      new Error("ack timed out"),
      { disposition: "outcome_unknown", userFacing: true },
    ))
    render(form(true))

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )

    // The reply may still have been durably accepted, so its eventual
    // terminal outcome must gate this round like an acknowledged one.
    await waitFor(() => {
      expect(appContextMock.dispatch).toHaveBeenCalledWith({
        type: "RECORD_CLARIFICATION_SUBMISSION",
        payload: {
          requestId: "inputreq_r1",
          commandId: expect.any(String),
          // Unconfirmed: an ack timeout must not lock the form while no
          // terminal outcome exists.
          accepted: false,
        },
      })
    })
    // The existing advisory behavior is unchanged until an outcome arrives -
    // including on the gating effect's rerun after the entry is recorded.
    appContextMock.state = {
      ...appContextMock.state,
      clarificationSubmissions: {
        inputreq_r1: {
          commandId: (appContextMock.sendMessage.mock.calls[0][1] as {
            clientMessageId: string
          }).clientMessageId,
          accepted: false,
        },
      },
    }
    expect(submitButton()).toBeEnabled()
  })

  it("still reactivates a form that never submitted, ignoring unrelated outcomes", async () => {
    appContextMock.state = {
      clarificationSubmissions: {},
      commandOutcomes: {
        "someone-elses-command": {
          outcome: "failed",
          resendSafe: false,
          messageCode: "task_command_failed",
        },
      },
    }
    const { rerender } = render(form(false))
    rerender(form(true))

    await waitFor(() => expect(submitButton()).toBeEnabled())
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("resets the gate for a new clarification round", async () => {
    const { rerender } = render(form(true))
    const commandId = await submitAccepted()

    rerender(form(false))
    withOutcome(commandId, false)
    rerender(form(true))
    await screen.findByRole("alert")

    rerender(form(true, "inputreq_r2"))

    await waitFor(() => expect(submitButton()).toBeEnabled())
    expect(screen.getByRole("textbox")).toHaveValue("")
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("does not record a submission for a round without a request id", async () => {
    render(
      <ClarificationForm
        interactions={[{ type: "text_input" as const, field: "city", label: "City" }]}
        active
      />,
    )

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Beijing" } })
    fireEvent.click(
      screen.getByRole("button", { name: "chatPage.clarification.submit" }),
    )

    await waitFor(() => expect(appContextMock.sendMessage).toHaveBeenCalledTimes(1))
    // With no round identity to bind to, gating is skipped entirely rather
    // than risking a recorded reply gating a different question.
    expect(appContextMock.dispatch).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "RECORD_CLARIFICATION_SUBMISSION" }),
    )
  })

  it("uses outcome notice keys that resolve in both locale trees", () => {
    for (const key of [
      "chatPage.clarification.replyNotApplied",
      "chatPage.clarification.replyOutcomeUnknown",
    ] as const) {
      expect(resolveTranslation("en", key)).not.toBe(key)
      expect(resolveTranslation("zh", key)).not.toBe(key)
    }
  })
})
