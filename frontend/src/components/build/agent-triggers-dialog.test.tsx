/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => {
  return (key: string, vars?: Record<string, string | number>) => {
    if (vars?.count) return `${key}:${vars.count}`
    return key
  }
})

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock, locale: "en" }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

const toastMocks = vi.hoisted(() => ({
  error: vi.fn(),
  success: vi.fn(),
  info: vi.fn(),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: toastMocks,
}))

vi.mock("@/lib/clipboard", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}))

import { AgentTriggersDialog } from "./agent-triggers-dialog"
import type { AgentTrigger, StagedTrigger } from "@/lib/agent-triggers-api"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function makeTrigger(overrides: Partial<AgentTrigger> & { id: number }): AgentTrigger {
  return {
    user_id: 1,
    agent_id: 42,
    type: "webhook",
    name: "Trigger",
    enabled: true,
    config: {},
    prompt_template: null,
    webhook_token: null,
    webhook_secret: null,
    next_run_at: null,
    last_run_at: null,
    last_error: null,
    created_at: null,
    updated_at: null,
    ...overrides,
  }
}

const GMAIL_ACCOUNTS_URL = "http://api.local/api/cloud/accounts?provider=gmail"

describe("AgentTriggersDialog", () => {
  let gmailAccounts: Array<{ id: number; provider: string; email: string | null }>

  const baseTrigger9 = makeTrigger({
    id: 9,
    type: "gmail",
    name: "Support inbox",
    config: {
      watch_label: "INBOX",
      sender_filter: "boss@company.com",
      subject_keyword: "urgent",
      oauth_account_id: 7,
    },
    prompt_template: "Reply to {{payload}}",
  })

  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    gmailAccounts = [
      { id: 7, provider: "gmail", email: "gerard.santos@gmail.com" },
    ]
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse(gmailAccounts))
      }
      if (url === "http://api.local/api/agents/42/triggers" && init?.method === "POST") {
        // Eager creation (toggling a type on with no existing trigger of it)
        // POSTs here — echo a real trigger back, not the list shape used by GET.
        const body = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse(makeTrigger({ id: 20, ...body })))
      }
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(jsonResponse([baseTrigger9]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9/runs") {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9" && init?.method === "PATCH") {
        // Echo the base trigger merged with the PATCH body, like a real
        // backend would — a bare `[]` fallback here would make `updated`
        // shapeless for any code that reads fields off the response.
        const patch = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse({ ...baseTrigger9, ...patch }))
      }
      return Promise.resolve(jsonResponse([]))
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("renders an existing Gmail trigger with its filters", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        gmailConnection={{
          isConnected: true,
          connectedAccount: "gerard.santos@gmail.com",
        }}
      />,
    )

    expect(await screen.findByText("triggers.cards.gmail.title")).toBeInTheDocument()
    expect(screen.queryByText("triggers.cards.appWidget.title")).not.toBeInTheDocument()

    // The type opens on its manage list; the pencil opens the editor.
    fireEvent.click(screen.getByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))

    expect(await screen.findByLabelText("triggers.form.watchLabel")).toHaveValue("INBOX")
    expect(screen.getByText("triggers.form.watchLabelHelp")).toBeInTheDocument()
    // Sender/subject filters live behind the collapsed "Optional filters"
    // disclosure and must be expanded before they're visible.
    fireEvent.click(screen.getByText("triggers.gmail.optionalFilters"))
    expect(await screen.findByLabelText("triggers.form.senderFilter")).toHaveValue("boss@company.com")
    expect(screen.getByLabelText("triggers.form.subjectKeyword")).toHaveValue("urgent")
    // The bound account heads the editor (avatar row, reference design); the
    // connection banner only shows while Gmail is NOT connected.
    expect(screen.getAllByText("gerard.santos@gmail.com").length).toBeGreaterThan(0)
    expect(screen.queryByText("triggers.gmail.connected")).not.toBeInTheDocument()
  })

  it("prompts for Gmail connection when the connector is missing", async () => {
    const onConnectGmail = vi.fn()

    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{
          isConnected: false,
          connectedAccount: null,
        }}
        onConnectGmail={onConnectGmail}
      />,
    )

    expect(await screen.findByText("triggers.gmail.notConnected")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "triggers.gmail.connect" }))

    expect(onConnectGmail).toHaveBeenCalledTimes(1)
  })

  it("shows the bound Gmail account for an existing trigger", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        agentName="Inbox Agent"
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))

    // Bound account: shown as the editor's avatar header (reference design),
    // with no account picker rendered.
    expect(await screen.findByText("gerard.santos@gmail.com")).toBeInTheDocument()
    expect(screen.queryByText("triggers.form.gmailAccount")).not.toBeInTheDocument()
    expect(screen.queryByText("triggers.gmail.accountMissing")).not.toBeInTheDocument()
  })

  it("auto-selects the only connected account for a new Gmail trigger", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(
          jsonResponse([{ id: 3, provider: "gmail", email: "solo@gmail.com" }]),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // No Gmail trigger yet: the list shows the empty state whose CTA opens a
    // new-trigger draft, pre-bound to the only connected account.
    fireEvent.click(
      await screen.findByRole("button", { name: /triggers.cards.gmail.addTrigger/ }),
    )
    expect(await screen.findByText("solo@gmail.com")).toBeInTheDocument()
  })

  it("requires an explicit choice when several accounts are connected", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(
          jsonResponse([
            { id: 3, provider: "gmail", email: "first@gmail.com" },
            { id: 4, provider: "gmail", email: "second@gmail.com" },
          ]),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(
      await screen.findByRole("button", { name: /triggers.cards.gmail.addTrigger/ }),
    )
    expect(
      await screen.findByText("triggers.form.gmailAccountPlaceholder"),
    ).toBeInTheDocument()
    expect(screen.queryByText("first@gmail.com")).not.toBeInTheDocument()
    expect(screen.queryByText("second@gmail.com")).not.toBeInTheDocument()
  })

  it("shows the connect-Gmail empty state when no accounts are connected", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse([]))
    })
    const onConnectGmail = vi.fn()

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{ isConnected: false, connectedAccount: null }}
        onConnectGmail={onConnectGmail}
      />,
    )

    expect(await screen.findByText("triggers.cards.gmail.empty.title")).toBeInTheDocument()
    expect(screen.queryByText("triggers.form.gmailAccount")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /triggers.cards.gmail.empty.cta/ }))
    expect(onConnectGmail).toHaveBeenCalledTimes(1)
  })

  it("warns when the bound Gmail account is no longer connected", async () => {
    gmailAccounts = [{ id: 8, provider: "gmail", email: "other@gmail.com" }]

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))

    expect(await screen.findByText("triggers.gmail.accountMissing")).toBeInTheDocument()
  })

  it("persists the detail header switch immediately without pressing save", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    // Manage list: the trigger's card is visible; the first switch is the
    // type-level header switch (on, since the trigger is enabled).
    await screen.findByText("Support inbox")

    const [headerSwitch] = screen.getAllByRole("switch")
    expect(headerSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(headerSwitch)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/9",
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({ enabled: false }),
        }),
      )
    })
    expect(headerSwitch).toHaveAttribute("aria-checked", "false")
  })

  it("reconciles a card switch from the PATCH response, not just the requested value", async () => {
    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse(gmailAccounts))
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(jsonResponse([baseTrigger9]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9/runs") {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9" && init?.method === "PATCH") {
        // A backend that (hypothetically) overrides the requested value —
        // the derived switch must reflect the response, not the request.
        return Promise.resolve(jsonResponse({ ...baseTrigger9, enabled: true }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    await screen.findByText("Support inbox")

    // switches: [header master, card switch]
    const [, cardSwitch] = screen.getAllByRole("switch")
    expect(cardSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(cardSwitch)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/9",
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({ enabled: false }),
        }),
      )
    })
    // Reconciled: the response said `enabled: true`, so the card stays on.
    await waitFor(() => {
      expect(cardSwitch).toHaveAttribute("aria-checked", "true")
    })
  })

  it("reveals the one-time webhook secret on the list after saving a new webhook", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        return Promise.resolve(
          jsonResponse(
            makeTrigger({
              id: 11,
              name: "API / Webhook",
              webhook_token: "tok",
              webhook_secret: "wh_secret_once",
            }),
          ),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    // Toggling on with no webhook yet opens the draft editor; Save creates
    // the trigger and lands back on the list, where the freshly generated
    // secret is revealed once.
    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)

    await screen.findByLabelText("triggers.form.secret")
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    expect(await screen.findByText("wh_secret_once")).toBeInTheDocument()
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers",
        expect.objectContaining({ method: "POST" }),
      )
    })
  })

  it("calls onChanged exactly once for a Save (Done afterward does not refetch again)", async () => {
    // Mirrors the builder's wiring: onChanged is the sole resync signal:
    // onOpenChange(false) must not ALSO trigger a refetch, or every save
    // fires the same GET twice.
    const onChanged = vi.fn()
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        onChanged={onChanged}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    const watchInput = await screen.findByLabelText("triggers.form.watchLabel")
    fireEvent.change(watchInput, { target: { value: "Support" } })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSettings" }))

    await waitFor(() => {
      expect(onChanged).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(screen.getByRole("button", { name: "common.done" }))
    expect(onChanged).toHaveBeenCalledTimes(1)
  })

  it("opens a draft editor without any POST when the switch is turned on with no webhook yet", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // The webhook type has no triggers yet: toggling on goes straight into
    // the new-webhook editor. Nothing is created until Save, so the header
    // switch (derived from saved triggers) stays off.
    await screen.findByText("triggers.cards.webhook.title")
    const [webhookCardSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookCardSwitch)

    await screen.findByLabelText("triggers.form.secret")
    expect(screen.getByText("triggers.editor.webhookNew")).toBeInTheDocument()
    const [headerSwitch] = screen.getAllByRole("switch")
    expect(headerSwitch).toHaveAttribute("aria-checked", "false")
    const postCalls = apiRequestMock.mock.calls.filter(
      ([url, init]) => url === "http://api.local/api/agents/42/triggers" && init?.method === "POST",
    )
    expect(postCalls).toHaveLength(0)
  })

  it("creates once via POST on Save; editing the card afterwards updates via PATCH", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        const body = options.body ? JSON.parse(options.body) : {}
        return Promise.resolve(jsonResponse(makeTrigger({ id: 15, ...body })))
      }
      if (url === "http://api.local/api/agents/42/triggers/15/runs") {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers/15" && options?.method === "PATCH") {
        return Promise.resolve(jsonResponse(makeTrigger({ id: 15, name: "Renamed hook" })))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // Draft → Save: exactly one POST, with the switch-on default enabled.
    const [webhookCardSwitch] = await screen.findAllByRole("switch")
    fireEvent.click(webhookCardSwitch)
    await screen.findByLabelText("triggers.form.name")
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers",
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining("\"enabled\":true"),
        }),
      )
    })

    // Back on the list, edit the new card and save again → PATCH, no 2nd POST.
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Renamed hook" } })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/15",
        expect.objectContaining({ method: "PATCH" }),
      )
    })
    const postCalls = apiRequestMock.mock.calls.filter(
      ([url, init]) => url === "http://api.local/api/agents/42/triggers" && init?.method === "POST",
    )
    expect(postCalls).toHaveLength(1)
  })

  it("keeps the dialog open on Escape when a fresh create just revealed a webhook secret", async () => {
    const onOpenChange = vi.fn()
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        return Promise.resolve(
          jsonResponse(
            makeTrigger({
              id: 12,
              name: "API / Webhook",
              webhook_token: "tok",
              webhook_secret: "wh_escape_secret",
            }),
          ),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={onOpenChange}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    const [webhookCardSwitch] = await screen.findAllByRole("switch")
    fireEvent.click(webhookCardSwitch)

    // Toggle-on opens the draft; Save creates it and reveals the secret.
    await screen.findByLabelText("triggers.form.secret")
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))
    expect(await screen.findByText("wh_escape_secret")).toBeInTheDocument()

    // Escape must not drop a secret that only exists because it was just
    // generated — unlike an ordinary validation failure, it is unrecoverable.
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    await waitFor(() => {
      expect(screen.getByText("wh_escape_secret")).toBeInTheDocument()
    })
    expect(onOpenChange).not.toHaveBeenCalledWith(false)

    // Only once the secret is explicitly acknowledged does Escape close.
    fireEvent.click(screen.getByRole("button", { name: "triggers.secret.dismiss" }))
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it("keeps showing a fresh secret after Back navigates to the overview", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        return Promise.resolve(
          jsonResponse(
            makeTrigger({
              id: 13,
              name: "API / Webhook",
              webhook_token: "tok",
              webhook_secret: "wh_back_secret",
            }),
          ),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    const [webhookCardSwitch] = await screen.findAllByRole("switch")
    fireEvent.click(webhookCardSwitch)

    // Toggle-on opens the draft; Save creates it and reveals the secret.
    await screen.findByLabelText("triggers.form.secret")
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))
    expect(await screen.findByText("wh_back_secret")).toBeInTheDocument()

    // Back navigates to the overview like any other exit path (nothing to
    // "commit" anymore) — the secret alert renders on the overview too, so
    // it stays visible until the user explicitly dismisses it.
    fireEvent.click(screen.getByRole("button", { name: "common.back" }))
    await waitFor(() => {
      expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()
    })
    expect(screen.getByText("wh_back_secret")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "triggers.secret.dismiss" }))
    await waitFor(() => {
      expect(screen.queryByText("wh_back_secret")).not.toBeInTheDocument()
    })
  })

  it("lands on the connect-Gmail empty state (switch left off) when toggled on with no accounts connected", async () => {
    const onOpenChange = vi.fn()
    const onConnectGmail = vi.fn()
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={onOpenChange}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
        onConnectGmail={onConnectGmail}
      />,
    )

    await screen.findByText("triggers.cards.webhook.title")
    const switches = screen.getAllByRole("switch")
    fireEvent.click(switches[2]) // Gmail card: no accounts connected

    // There is nothing to enable yet, so the switch stays off and the
    // "connect Gmail" empty state shows instead of the watch-label form.
    expect(await screen.findByText("triggers.cards.gmail.empty.title")).toBeInTheDocument()
    expect(screen.queryByLabelText("triggers.form.watchLabel")).not.toBeInTheDocument()
    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "false")

    fireEvent.click(screen.getByRole("button", { name: /triggers.cards.gmail.empty.cta/ }))
    expect(onConnectGmail).toHaveBeenCalledTimes(1)

    // Nothing was ever drafted, so Done just closes cleanly.
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(toastMocks.error).not.toHaveBeenCalled()
  })

  it("disables navigation while a detail toggle is in flight, and rolls back cleanly on rejection", async () => {
    const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"
    let rejectPatch: (err: Error) => void = () => {}

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === TRIGGERS_URL) {
        return Promise.resolve(
          jsonResponse([
            makeTrigger({ id: 20, name: "Backup hook" }),
            makeTrigger({ id: 21, name: "Primary hook" }),
          ]),
        )
      }
      if (url === `${TRIGGERS_URL}/20/runs` || url === `${TRIGGERS_URL}/21/runs`) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === `${TRIGGERS_URL}/21` && init?.method === "PATCH") {
        // Never resolves on its own — held open so navigation controls can be
        // asserted disabled, then rejected explicitly below.
        return new Promise<Response>((_resolve, reject) => {
          rejectPatch = reject
        })
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    // Manage list, newest first: [Primary hook (21), Backup hook (20)].
    await screen.findByText("Primary hook")

    // switches: [header master, card 21, card 20]
    const [, primaryCardSwitch] = screen.getAllByRole("switch")
    expect(primaryCardSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(primaryCardSwitch)

    // While the PATCH is pending every navigation/mutation control is
    // disabled, so the eventual failure can't land on an unrelated view.
    await waitFor(() => {
      expect(primaryCardSwitch).toBeDisabled()
    })
    expect(screen.getByRole("button", { name: "common.back" })).toBeDisabled()
    for (const editButton of screen.getAllByRole("button", { name: "triggers.actions.edit" })) {
      expect(editButton).toBeDisabled()
    }

    rejectPatch(new Error("network error"))

    // The switch is derived from the (unchanged) trigger list, so a rejected
    // PATCH leaves it exactly where it started.
    await waitFor(() => {
      expect(toastMocks.error).toHaveBeenCalled()
    })
    expect(primaryCardSwitch).toHaveAttribute("aria-checked", "true")
    expect(screen.getByText("Primary hook")).toBeInTheDocument()
  })

  it("resyncs the trigger list when a batch disable partially fails", async () => {
    const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"
    let triggers = [
      makeTrigger({ id: 30, name: "Hook A" }),
      makeTrigger({ id: 31, name: "Hook B" }),
    ]
    let getCallsAfterFailure = 0
    let patchAttempted = false

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        if (patchAttempted) getCallsAfterFailure += 1
        return Promise.resolve(jsonResponse(triggers))
      }
      if (url === `${TRIGGERS_URL}/30` && init?.method === "PATCH") {
        triggers = triggers.map((item) => (item.id === 30 ? { ...item, enabled: false } : item))
        return Promise.resolve(jsonResponse(triggers[0]))
      }
      if (url === `${TRIGGERS_URL}/31` && init?.method === "PATCH") {
        patchAttempted = true
        return Promise.reject(new Error("boom"))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    const cardSwitch = (
      await screen.findAllByRole("switch")
    ).find((el) => el.getAttribute("aria-checked") === "true")
    fireEvent.click(cardSwitch!)

    // One PATCH in the batch rejected: the catch resyncs via a fresh GET
    // instead of trusting the local list (which would otherwise wrongly
    // report both triggers disabled).
    await waitFor(() => {
      expect(toastMocks.error).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(getCallsAfterFailure).toBeGreaterThan(0)
    })
  })

  it("removes a deleted trigger's card and keeps the remaining ones", async () => {
    const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"
    let triggers = [
      makeTrigger({ id: 50, name: "Older hook" }),
      makeTrigger({ id: 51, name: "Newer hook" }),
    ]

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse(triggers))
      }
      if (url === `${TRIGGERS_URL}/50/runs` || url === `${TRIGGERS_URL}/51/runs`) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === `${TRIGGERS_URL}/51` && init?.method === "DELETE") {
        triggers = triggers.filter((item) => item.id !== 51)
        return Promise.resolve(jsonResponse({}))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    // Manage list, newest first: [Newer hook (51), Older hook (50)].
    await screen.findByText("Newer hook")

    // Delete the newest card via its trash button + confirmation popover.
    const [newerHookDelete] = screen.getAllByRole("button", { name: "triggers.actions.delete" })
    fireEvent.click(newerHookDelete)
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        `${TRIGGERS_URL}/51`,
        expect.objectContaining({ method: "DELETE" }),
      )
    })
    await waitFor(() => {
      expect(screen.queryByText("Newer hook")).not.toBeInTheDocument()
    })
    expect(screen.getByText("Older hook")).toBeInTheDocument()
  })

  it("keeps each overview switch's busy guard independent across two types toggled back-to-back", async () => {
    const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"
    const triggers = [
      makeTrigger({ id: 40, name: "Hook" }),
      makeTrigger({ id: 41, type: "scheduled", name: "Schedule", config: { interval_seconds: 3600 } }),
    ]

    let resolveWebhookPatch: ((value: Response) => void) | undefined
    const webhookPatchPromise = new Promise<Response>((resolve) => {
      resolveWebhookPatch = resolve
    })
    let resolveScheduledPatch: ((value: Response) => void) | undefined
    const scheduledPatchPromise = new Promise<Response>((resolve) => {
      resolveScheduledPatch = resolve
    })

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse(triggers))
      }
      if (url === `${TRIGGERS_URL}/40` && init?.method === "PATCH") return webhookPatchPromise
      if (url === `${TRIGGERS_URL}/41` && init?.method === "PATCH") return scheduledPatchPromise
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    // TRIGGER_TYPES order is webhook, scheduled, gmail.
    const [webhookSwitch, scheduledSwitch] = await screen.findAllByRole("switch")
    expect(webhookSwitch).toHaveAttribute("aria-checked", "true")
    expect(scheduledSwitch).toHaveAttribute("aria-checked", "true")

    fireEvent.click(webhookSwitch)
    await waitFor(() => {
      expect(webhookSwitch).toBeDisabled()
    })
    expect(scheduledSwitch).not.toBeDisabled()

    fireEvent.click(scheduledSwitch)
    await waitFor(() => {
      expect(scheduledSwitch).toBeDisabled()
    })

    // Resolve the scheduled toggle first. A scalar busy-guard would have
    // cleared entirely here and wrongly re-enabled webhook's switch while
    // its own PATCH was still in flight — the Set-based guard keeps them
    // independent.
    resolveScheduledPatch?.(jsonResponse({ ...triggers[1], enabled: false }))
    await waitFor(() => {
      expect(scheduledSwitch).not.toBeDisabled()
    })
    expect(webhookSwitch).toBeDisabled()

    resolveWebhookPatch?.(jsonResponse({ ...triggers[0], enabled: false }))
    await waitFor(() => {
      expect(webhookSwitch).not.toBeDisabled()
    })
  })

  it("starts a one-click test run from the editor and refreshes recent runs", async () => {
    let runsCalls = 0
    let patchCalls = 0
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse(gmailAccounts))
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(jsonResponse([baseTrigger9]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9/runs") {
        runsCalls += 1
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9" && init?.method === "PATCH") {
        // Test always saves first, so the on-screen draft is what runs.
        patchCalls += 1
        const patch = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse({ ...baseTrigger9, ...patch }))
      }
      if (url === "http://api.local/api/agents/42/triggers/9/test" && init?.method === "POST") {
        return Promise.resolve(jsonResponse({ trigger_run: { id: 77 }, duplicate: false }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))

    const testButton = await screen.findByRole("button", { name: "triggers.actions.test" })
    expect(testButton).not.toBeDisabled()
    const runsCallsBeforeTest = runsCalls
    fireEvent.click(testButton)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/9/test",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            payload: { message: "test trigger" },
            source_event_id: null,
          }),
        }),
      )
    })
    await waitFor(() => {
      expect(toastMocks.success).toHaveBeenCalledWith("triggers.messages.testStarted")
    })
    // The unsaved-edits-safe contract: the trigger was saved (PATCH) before
    // the test fired, and the runs list refreshed to show the new run.
    expect(patchCalls).toBe(1)
    expect(runsCalls).toBeGreaterThan(runsCallsBeforeTest)
  })

  it("saves an unsaved draft first, then starts the test, when Test trigger is clicked", async () => {
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/agents/42/triggers" && init?.method === "POST") {
        const body = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse(makeTrigger({ id: 21, ...body })))
      }
      if (url === "http://api.local/api/agents/42/triggers/21/runs") {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers/21/test" && init?.method === "POST") {
        return Promise.resolve(jsonResponse({ trigger_run: { id: 5 }, duplicate: false }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // Webhook type has no triggers: toggling on opens a new-webhook draft.
    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)

    await screen.findByLabelText("triggers.form.secret")
    const testButton = screen.getByRole("button", { name: "triggers.actions.test" })
    expect(testButton).not.toBeDisabled()
    fireEvent.click(testButton)

    // The draft is persisted first (exactly one POST), then the test fires
    // against the fresh id — and the editor stays open on the saved trigger.
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/21/test",
        expect.objectContaining({ method: "POST" }),
      )
    })
    const createCalls = apiRequestMock.mock.calls.filter(
      ([url, init]) => url === "http://api.local/api/agents/42/triggers" && init?.method === "POST",
    )
    expect(createCalls).toHaveLength(1)
    const createIndex = apiRequestMock.mock.calls.findIndex(
      ([url, init]) => url === "http://api.local/api/agents/42/triggers" && init?.method === "POST",
    )
    const testIndex = apiRequestMock.mock.calls.findIndex(
      ([url]) => url === "http://api.local/api/agents/42/triggers/21/test",
    )
    expect(createIndex).toBeLessThan(testIndex)
    await waitFor(() => {
      expect(toastMocks.success).toHaveBeenCalledWith("triggers.messages.testStarted")
    })
    expect(screen.getByLabelText("triggers.form.name")).toBeInTheDocument()
  })

  it("keeps a draft's typed fields when the header switch is clicked mid-composition", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // Webhook type has no triggers: toggle on → draft editor, type a name.
    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Half-typed draft" } })

    // Clicking the (derived, still-off) header switch while composing the
    // draft must be a no-op — not a form reset.
    const [headerSwitch] = screen.getAllByRole("switch")
    fireEvent.click(headerSwitch)

    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("Half-typed draft")
  })

  it("round-trips a match-anything Gmail watch label as a blank field", async () => {
    const starTrigger = makeTrigger({
      id: 9,
      type: "gmail",
      name: "Star inbox",
      config: { watch_label: "*", oauth_account_id: 7 },
    })
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse(gmailAccounts))
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(jsonResponse([starTrigger]))
      }
      if (url === "http://api.local/api/agents/42/triggers/9" && init?.method === "PATCH") {
        const patch = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse({ ...starTrigger, ...patch }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    // The card labels the "*" sentinel as "all incoming emails", not "*".
    expect(await screen.findByText(/triggers.item.gmailAllEmails/)).toBeInTheDocument()

    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    // The editor shows the sentinel as a blank field ("leave blank = all").
    expect(await screen.findByLabelText("triggers.form.watchLabel")).toHaveValue("")

    // Saving the untouched blank field writes the sentinel back.
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSettings" }))
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/9",
        expect.objectContaining({
          method: "PATCH",
          body: expect.stringContaining('"watch_label":"*"'),
        }),
      )
    })
  })

  it("lets a bound Gmail trigger be re-bound via the change-account button", async () => {
    gmailAccounts = [
      { id: 7, provider: "gmail", email: "gerard.santos@gmail.com" },
      { id: 8, provider: "gmail", email: "work@company.com" },
    ]

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))

    // Bound: avatar header with the change-account affordance, no picker.
    expect(screen.queryByText("triggers.form.gmailAccount")).not.toBeInTheDocument()
    fireEvent.click(
      await screen.findByRole("button", { name: "triggers.gmail.changeAccount" }),
    )

    // Unbound: the account picker is back for an explicit new choice.
    expect(await screen.findByText("triggers.form.gmailAccount")).toBeInTheDocument()
    expect(screen.getByText("triggers.form.gmailAccountPlaceholder")).toBeInTheDocument()
  })

  it("keeps unsaved field edits when the header switch is toggled while editing", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    const watchInput = await screen.findByLabelText("triggers.form.watchLabel")
    fireEvent.change(watchInput, { target: { value: "Edited but unsaved" } })

    // In the editor the only switch is the type-level header one; toggling
    // it patches the trigger's enabled state but must not wipe the draft.
    const [headerSwitch] = screen.getAllByRole("switch")
    fireEvent.click(headerSwitch)

    await waitFor(() => {
      expect(headerSwitch).toHaveAttribute("aria-checked", "false")
    })
    expect(watchInput).toHaveValue("Edited but unsaved")
  })
})

describe("AgentTriggersDialog staging mode (agent not created yet)", () => {
  function stagedWebhook(clientId: number, name: string): StagedTrigger {
    return {
      clientId,
      type: "webhook",
      name,
      enabled: true,
      config: {},
      prompt_template: null,
      secret: null,
    }
  }

  function renderStaging(triggers: StagedTrigger[]) {
    const onChange = vi.fn()
    render(
      <AgentTriggersDialog
        agentId={null}
        open
        onOpenChange={vi.fn()}
        staged={{ triggers, onChange }}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )
    return onChange
  }

  // Unlike renderStaging's vi.fn(), this harness feeds onChange back into the
  // staged prop like agent-builder does, so list updates round-trip and the
  // form-sync behavior under real re-renders is exercised.
  function StatefulStagingHarness({
    initial,
    onChangeSpy,
    onOpenChange,
  }: {
    initial: StagedTrigger[]
    onChangeSpy?: (next: StagedTrigger[]) => void
    onOpenChange?: (open: boolean) => void
  }) {
    const [triggers, setTriggers] = React.useState(initial)
    return (
      <AgentTriggersDialog
        agentId={null}
        open
        onOpenChange={onOpenChange ?? vi.fn()}
        staged={{
          triggers,
          onChange: (next) => {
            onChangeSpy?.(next)
            setTriggers(next)
          },
        }}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />
    )
  }

  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue(jsonResponse([]))
  })

  afterEach(() => {
    cleanup()
  })

  it("opens a draft editor when a type is toggled on, staging only on Save", async () => {
    const onChangeSpy = vi.fn()
    render(<StatefulStagingHarness initial={[]} onChangeSpy={onChangeSpy} />)

    await screen.findByText("triggers.cards.webhook.title")

    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)

    // Toggling on with no staged webhook opens the draft editor without
    // staging anything yet.
    await screen.findByLabelText("triggers.form.name")
    expect(onChangeSpy).not.toHaveBeenCalled()

    // Save stages the draft, enabled (that's what the toggle-on meant).
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))
    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, type: "webhook", enabled: true }),
      ])
    })
    // Back on the list, the type-level header switch is now on.
    const [headerSwitch] = screen.getAllByRole("switch")
    expect(headerSwitch).toHaveAttribute("aria-checked", "true")
  })

  it("shows the empty state (not a form) when a type with no triggers is opened via its title", async () => {
    renderStaging([])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    expect(await screen.findByText("triggers.cards.webhook.empty.title")).toBeInTheDocument()
    expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()

    // The empty state's own CTA opens the same draft form.
    fireEvent.click(screen.getByRole("button", { name: /triggers.cards.webhook.empty.cta/ }))
    await screen.findByLabelText("triggers.form.name")
    expect(screen.getByText("triggers.editor.webhookNew")).toBeInTheDocument()
  })

  it("Save stages the toggled-on draft exactly once, with the edited name", async () => {
    const onChangeSpy = vi.fn()
    render(<StatefulStagingHarness initial={[]} onChangeSpy={onChangeSpy} />)

    const [webhookSwitch] = await screen.findAllByRole("switch")
    fireEvent.click(webhookSwitch)

    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Toggled hook" } })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    await waitFor(() => {
      expect(onChangeSpy).toHaveBeenCalledWith([
        expect.objectContaining({
          clientId: -1,
          type: "webhook",
          name: "Toggled hook",
          enabled: true,
        }),
      ])
    })
    expect(onChangeSpy).toHaveBeenCalledTimes(1)
  })

  it("applies the card switch to the staged trigger without pressing save", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Hook one")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("Hook one")

    // switches: [header master, card]
    const [, cardSwitch] = screen.getAllByRole("switch")
    expect(cardSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(cardSwitch)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, enabled: false }),
      ])
    })
  })

  it("appends a new staged trigger via Add instead of overwriting an existing one", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "First hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("First hook")

    fireEvent.click(screen.getByRole("button", { name: /triggers.actions.addAnotherWebhook/ }))

    // Creation state: an empty draft form.
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "Second hook" },
    })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "First hook" }),
        // "Add another" drafts save enabled, like the reference design.
        expect.objectContaining({ clientId: -2, name: "Second hook", type: "webhook", enabled: true }),
      ])
    })
  })

  it("lists staged triggers newest first", async () => {
    renderStaging([stagedWebhook(-1, "Old hook"), stagedWebhook(-2, "New hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    // Newest staged trigger (-2) precedes the older one in the card list.
    const newCard = await screen.findByText("New hook")
    const oldCard = screen.getByText("Old hook")
    expect(
      newCard.compareDocumentPosition(oldCard) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("removes a staged trigger after confirming in the card's popover", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Doomed hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    // The X opens a confirmation popover; the destructive button deletes.
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.delete" }))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([])
    })
  })

  it("keeps the trigger when the delete popover is cancelled", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Kept hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.delete" }))
    // The form itself also has a "common.cancel" button now (next to Save),
    // so scope this query to the delete-confirm popover specifically.
    const popover = (await screen.findByText("triggers.deleteConfirm")).parentElement as HTMLElement
    fireEvent.click(within(popover).getByRole("button", { name: "common.cancel" }))

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: "triggers.actions.confirmDelete" }),
      ).not.toBeInTheDocument()
    })
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByText("Kept hook")).toBeInTheDocument()
  })

  it("deleting one card keeps the other card intact after the staged list round-trips", async () => {
    render(
      <StatefulStagingHarness
        initial={[stagedWebhook(-1, "Old hook"), stagedWebhook(-2, "New hook")]}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("New hook")

    // Delete the older card (cards list newest first, so its trash is second).
    const [, oldHookDelete] = screen.getAllByRole("button", { name: "triggers.actions.delete" })
    fireEvent.click(oldHookDelete)
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(screen.queryByText("Old hook")).not.toBeInTheDocument()
    })
    expect(screen.getByText("New hook")).toBeInTheDocument()
  })

  it("keeps the header switch usable alongside unsaved edits after a round-trip", async () => {
    render(<StatefulStagingHarness initial={[stagedWebhook(-1, "Hook")]} />)

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Renamed but unsaved" } })

    // The immediate type-level toggle round-trips the staged list; the
    // pending name edit must survive it.
    const [headerSwitch] = screen.getAllByRole("switch")
    fireEvent.click(headerSwitch)
    await waitFor(() => {
      expect(headerSwitch).toHaveAttribute("aria-checked", "false")
    })
    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("Renamed but unsaved")
  })

  it("discards unsaved edits (without attempting to save) when the dialog is dismissed via Escape", async () => {
    const onChangeSpy = vi.fn()
    const onOpenChange = vi.fn()
    render(
      <StatefulStagingHarness
        initial={[stagedWebhook(-1, "Old name")]}
        onChangeSpy={onChangeSpy}
        onOpenChange={onOpenChange}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Unsaved on escape" } })

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })

    // Dismissal never attempts to save the draft (unlike Save, which is the
    // only thing that persists edits) — it just closes.
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(onChangeSpy).not.toHaveBeenCalled()
  })

  it("closes on Done without validating a Gmail quick-toggle draft, but Save still enforces it", async () => {
    const onOpenChange = vi.fn()
    // Two connected accounts: the quick toggle can't guess which one to bind,
    // so it opens the draft form (switch preset on) for an explicit choice —
    // unlike the zero-accounts case, there IS something to enable here.
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(
          jsonResponse([
            { id: 3, provider: "gmail", email: "first@gmail.com" },
            { id: 4, provider: "gmail", email: "second@gmail.com" },
          ]),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })
    render(
      <StatefulStagingHarness initial={[]} onOpenChange={onOpenChange} />,
    )

    await screen.findByText("triggers.cards.webhook.title")
    const switches = screen.getAllByRole("switch")
    fireEvent.click(switches[2])

    // The quick toggle opens a draft editor for an explicit account choice.
    await screen.findByLabelText("triggers.form.watchLabel")
    expect(screen.getByText("triggers.editor.gmailNew")).toBeInTheDocument()

    // Save would enforce validation (no account picked yet)...
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSettings" }))
    await waitFor(() => {
      expect(toastMocks.error).toHaveBeenCalled()
    })
    expect(onOpenChange).not.toHaveBeenCalledWith(false)

    // ...but Done never validates at all — it just discards the draft.
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it("deleting one card leaves the other staged triggers untouched in onChange", async () => {
    const onChange = renderStaging([
      stagedWebhook(-1, "Old hook"),
      stagedWebhook(-2, "New hook"),
    ])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("New hook")

    // Delete the older card (-1) via its trash button (cards newest first).
    const [, oldHookDelete] = screen.getAllByRole("button", { name: "triggers.actions.delete" })
    fireEvent.click(oldHookDelete)
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -2, name: "New hook" }),
      ])
    })
  })

  it("saves pending edits to the staged trigger being edited via the Save button", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Old name")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("Old name")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "New name" },
    })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "New name", type: "webhook" }),
      ])
    })
  })

  it("discards unsaved edits when navigating back to the overview", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Old name")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("Old name")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "Renamed but unsaved" },
    })
    fireEvent.click(screen.getByRole("button", { name: "common.back" }))

    // Back landed on the overview without ever calling onChange — the edit
    // was never saved.
    expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })

  it("closes without changes when Done is pressed on an untouched form", async () => {
    const onOpenChange = vi.fn()
    const onChange = vi.fn()
    render(
      <AgentTriggersDialog
        agentId={null}
        open
        onOpenChange={onOpenChange}
        staged={{ triggers: [stagedWebhook(-1, "Untouched hook")], onChange }}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.edit" }))
    await screen.findByLabelText("triggers.form.name")
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(onChange).not.toHaveBeenCalled()
  })

  it("tests a staged trigger locally inside the editor, rendering its prompt", async () => {
    const onChange = vi.fn()
    const onOpenChange = vi.fn()
    render(
      <AgentTriggersDialog
        agentId={null}
        open
        onOpenChange={onOpenChange}
        staged={{ triggers: [], onChange }}
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    // Toggle webhook on → draft editor; give it a template and hit Test.
    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)
    const promptInput = await screen.findByLabelText("triggers.form.webhookPrompt")
    fireEvent.change(promptInput, {
      target: { value: "Lead: {{payload}} (test={{test}}, type={{trigger_type}})" },
    })

    const testButton = screen.getByRole("button", { name: "triggers.actions.test" })
    expect(testButton).not.toBeDisabled()
    fireEvent.click(testButton)

    // A run row appears right inside the editor (like the reference design),
    // with the rendered prompt — template variables substituted, exactly
    // what a real firing would send to the agent.
    expect(await screen.findByText("triggers.runs.title")).toBeInTheDocument()
    expect(screen.getByText("triggers.test.stagedPreviewNote")).toBeInTheDocument()
    expect(screen.getByText("triggers.runStatus.completed")).toBeInTheDocument()
    expect(screen.getByText(/trigger-run:test:draft:/)).toBeInTheDocument()
    const rendered = screen.getByText(/test=true, type=webhook/)
    expect(rendered.textContent).toContain("test trigger")
    expect(rendered.textContent).not.toContain("{{payload}}")

    // Everything happens in place: no staging, no API call, dialog stays open.
    expect(onChange).not.toHaveBeenCalled()
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/test"),
      expect.anything(),
    )
  })

  it("disables every staged trigger of a type when its switch is toggled off", async () => {
    const onChange = renderStaging([
      stagedWebhook(-1, "Hook one"),
      stagedWebhook(-2, "Hook two"),
    ])

    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    expect(webhookSwitch).toHaveAttribute("aria-checked", "true")

    fireEvent.click(webhookSwitch)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, enabled: false }),
        expect.objectContaining({ clientId: -2, enabled: false }),
      ])
    })
  })
})

describe("AgentTriggersDialog empty states", () => {
  const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"

  beforeEach(() => {
    apiRequestMock.mockReset()
    // A fresh Response per call — mockResolvedValue would reuse the same
    // Response object, and a body can only be read once. POST needs to
    // return a real trigger object: the empty state's CTA now eagerly
    // creates one, and the response drives the edit view that follows.
    let nextId = 100
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === TRIGGERS_URL && init?.method === "POST") {
        const body = init.body ? JSON.parse(init.body) : {}
        return Promise.resolve(jsonResponse(makeTrigger({ id: nextId++, ...body })))
      }
      return Promise.resolve(jsonResponse([]))
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("shows the webhook empty state until the switch is turned on", async () => {
    render(<AgentTriggersDialog agentId={42} open onOpenChange={vi.fn()} />)

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    expect(await screen.findByText("triggers.cards.webhook.empty.title")).toBeInTheDocument()
    expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /triggers.cards.webhook.empty.cta/ }))

    expect(await screen.findByLabelText("triggers.form.name")).toBeInTheDocument()
    expect(screen.queryByText("triggers.cards.webhook.empty.title")).not.toBeInTheDocument()
  })

  it("shows the schedule empty state until a schedule is created", async () => {
    render(<AgentTriggersDialog agentId={42} open onOpenChange={vi.fn()} />)

    fireEvent.click(await screen.findByText("triggers.cards.scheduled.title"))

    expect(await screen.findByText("triggers.cards.scheduled.empty.title")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /triggers.cards.scheduled.empty.cta/ }))

    expect(await screen.findByText("triggers.schedule.recurrenceLabel")).toBeInTheDocument()
  })
})

describe("AgentTriggersDialog schedule recurrence", () => {
  const TRIGGERS_URL = "http://api.local/api/agents/42/triggers"

  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  // Toggling the schedule switch on now eagerly creates (POST) a default
  // hourly schedule; a later Save (after picking a different recurrence)
  // updates it via PATCH. `getBody()` always reflects the latest of either.
  function mockCreate() {
    let lastBody: Record<string, unknown> | null = null
    apiRequestMock.mockImplementation((url: string, init?: { method?: string; body?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) return Promise.resolve(jsonResponse([]))
      if (url === TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === TRIGGERS_URL && init?.method === "POST") {
        lastBody = init.body ? JSON.parse(init.body) : null
        return Promise.resolve(
          jsonResponse(makeTrigger({ id: 90, type: "scheduled", config: (lastBody?.config as Record<string, unknown>) ?? {} })),
        )
      }
      if (url === `${TRIGGERS_URL}/90` && init?.method === "PATCH") {
        lastBody = init.body ? JSON.parse(init.body) : null
        return Promise.resolve(
          jsonResponse(makeTrigger({ id: 90, type: "scheduled", config: (lastBody?.config as Record<string, unknown>) ?? {} })),
        )
      }
      return Promise.resolve(jsonResponse([]))
    })
    return () => lastBody
  }

  async function openScheduleDraft() {
    render(<AgentTriggersDialog agentId={42} open onOpenChange={vi.fn()} />)
    const [, scheduledCardSwitch] = await screen.findAllByRole("switch")
    fireEvent.click(scheduledCardSwitch)
    await screen.findByText("triggers.schedule.recurrenceLabel")
  }

  it("saves an hourly schedule with the default recurrence", async () => {
    const getBody = mockCreate()
    await openScheduleDraft()

    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSchedule" }))

    await waitFor(() => {
      expect(getBody()).toMatchObject({
        config: { recurrence: "hourly", interval_seconds: 3600 },
      })
    })
    // The user's IANA timezone and the (defaulted-to-today) start anchor
    // always ride along, so the backend can honor the picked local time.
    const config = (getBody() as { config: Record<string, unknown> }).config
    expect(config.timezone).toBeTruthy()
    expect(config.next_run_at).toBeTruthy()
  })

  it("saves a daily schedule", async () => {
    const getBody = mockCreate()
    await openScheduleDraft()

    fireEvent.click(screen.getByText("triggers.schedule.daily"))
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSchedule" }))

    await waitFor(() => {
      expect(getBody()).toMatchObject({
        config: { recurrence: "daily", interval_seconds: 86400 },
      })
    })
  })

  it("saves a weekly schedule with the selected weekdays", async () => {
    const getBody = mockCreate()
    await openScheduleDraft()

    fireEvent.click(screen.getByText("triggers.schedule.weekly"))
    // Default weekday selection is Monday (index 0); add Wednesday (index 2).
    fireEvent.click(await screen.findByText("triggers.schedule.weekdayWed"))
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSchedule" }))

    await waitFor(() => {
      expect(getBody()).toMatchObject({
        config: { recurrence: "weekly", weekdays: [0, 2] },
      })
    })
  })

  it("saves a monthly schedule with the default day of month", async () => {
    const getBody = mockCreate()
    await openScheduleDraft()

    fireEvent.click(screen.getByText("triggers.schedule.monthly"))
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSchedule" }))

    await waitFor(() => {
      expect(getBody()).toMatchObject({
        config: { recurrence: "monthly", day_of_month: 1 },
      })
    })
  })

  it("saves a custom schedule converting amount+unit into interval_seconds", async () => {
    const getBody = mockCreate()
    await openScheduleDraft()

    fireEvent.click(screen.getByText("triggers.schedule.custom"))
    const amountInput = await screen.findByLabelText("triggers.schedule.runEvery")
    fireEvent.change(amountInput, { target: { value: "2" } })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveSchedule" }))

    await waitFor(() => {
      // Default unit is minutes: 2 minutes = 120 seconds.
      expect(getBody()).toMatchObject({
        config: { recurrence: "custom", interval_seconds: 120 },
      })
    })
  })
})

describe("AgentTriggersDialog owner routing", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation(() => Promise.resolve(jsonResponse([])))
  })

  afterEach(() => {
    cleanup()
  })

  it("loads triggers from the workforce route when owner is a workforce", async () => {
    render(
      <AgentTriggersDialog
        agentId={null}
        owner={{ kind: "workforce", id: 5 }}
        open
        onOpenChange={vi.fn()}
      />,
    )

    await waitFor(() =>
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/workforces/5/triggers",
      ),
    )
    // The workforce owner must never fall through to the agent route.
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/agents/"),
    )
  })

  it("loads triggers from the agent route when no explicit owner is given", async () => {
    render(<AgentTriggersDialog agentId={42} open onOpenChange={vi.fn()} />)

    await waitFor(() =>
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers",
      ),
    )
  })

  it("toggles a workforce-owned trigger via PATCH on the workforce route", async () => {
    const WORKFORCE_TRIGGERS_URL = "http://api.local/api/workforces/5/triggers"
    const trigger = makeTrigger({ id: 60, name: "Workforce hook", enabled: true })

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === WORKFORCE_TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([trigger]))
      }
      if (url === `${WORKFORCE_TRIGGERS_URL}/60` && init?.method === "PATCH") {
        return Promise.resolve(jsonResponse({ ...trigger, enabled: false }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={null}
        owner={{ kind: "workforce", id: 5 }}
        open
        onOpenChange={vi.fn()}
      />,
    )

    const cardSwitch = (
      await screen.findAllByRole("switch")
    ).find((el) => el.getAttribute("aria-checked") === "true")
    expect(cardSwitch).toBeDefined()
    fireEvent.click(cardSwitch!)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        `${WORKFORCE_TRIGGERS_URL}/60`,
        expect.objectContaining({ method: "PATCH" }),
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/agents/"),
      expect.anything(),
    )
  })

  it("toggles an existing workforce-owned trigger from the detail view via PATCH", async () => {
    const WORKFORCE_TRIGGERS_URL = "http://api.local/api/workforces/5/triggers"
    const trigger = makeTrigger({ id: 63, name: "Workforce hook", enabled: true })

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === WORKFORCE_TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([trigger]))
      }
      if (url === `${WORKFORCE_TRIGGERS_URL}/63/runs`) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === `${WORKFORCE_TRIGGERS_URL}/63` && init?.method === "PATCH") {
        return Promise.resolve(jsonResponse({ ...trigger, enabled: false }))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={null}
        owner={{ kind: "workforce", id: 5 }}
        open
        onOpenChange={vi.fn()}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("Workforce hook")
    const [headerSwitch] = screen.getAllByRole("switch")
    expect(headerSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(headerSwitch)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        `${WORKFORCE_TRIGGERS_URL}/63`,
        expect.objectContaining({ method: "PATCH" }),
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/agents/"),
      expect.anything(),
    )
  })

  it("creates a workforce-owned trigger via POST on the workforce route", async () => {
    const WORKFORCE_TRIGGERS_URL = "http://api.local/api/workforces/5/triggers"

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === WORKFORCE_TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === WORKFORCE_TRIGGERS_URL && init?.method === "POST") {
        return Promise.resolve(jsonResponse(makeTrigger({ id: 61, enabled: true })))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={null}
        owner={{ kind: "workforce", id: 5 }}
        open
        onOpenChange={vi.fn()}
      />,
    )

    await screen.findByText("triggers.cards.webhook.title")
    const [webhookCardSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookCardSwitch)
    await screen.findByLabelText("triggers.form.secret")
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.saveWebhook" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        WORKFORCE_TRIGGERS_URL,
        expect.objectContaining({ method: "POST" }),
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/agents/"),
      expect.anything(),
    )
  })

  it("deletes a workforce-owned trigger via DELETE on the workforce route", async () => {
    const WORKFORCE_TRIGGERS_URL = "http://api.local/api/workforces/5/triggers"
    const trigger = makeTrigger({ id: 62, name: "Workforce hook" })

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      if (url === WORKFORCE_TRIGGERS_URL && (!init?.method || init.method === "GET")) {
        return Promise.resolve(jsonResponse([trigger]))
      }
      if (url === `${WORKFORCE_TRIGGERS_URL}/62` && init?.method === "DELETE") {
        return Promise.resolve(jsonResponse({}))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={null}
        owner={{ kind: "workforce", id: 5 }}
        open
        onOpenChange={vi.fn()}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByText("Workforce hook")
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.delete" }))
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        `${WORKFORCE_TRIGGERS_URL}/62`,
        expect.objectContaining({ method: "DELETE" }),
      )
    })
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/api/agents/"),
      expect.anything(),
    )
  })
})
