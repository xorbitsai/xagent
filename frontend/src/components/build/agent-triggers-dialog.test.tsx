/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
  useI18n: () => ({ t: translateMock }),
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
import type { StagedTrigger } from "@/lib/agent-triggers-api"

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

const GMAIL_ACCOUNTS_URL = "http://api.local/api/cloud/accounts?provider=gmail"

describe("AgentTriggersDialog", () => {
  let gmailAccounts: Array<{ id: number; provider: string; email: string | null }>

  beforeEach(() => {
    apiRequestMock.mockReset()
    routerPushMock.mockReset()
    gmailAccounts = [
      { id: 7, provider: "gmail", email: "gerard.santos@gmail.com" },
    ]
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse(gmailAccounts))
      }
      if (url === "http://api.local/api/agents/42/triggers") {
        return Promise.resolve(
          jsonResponse([
            {
              id: 9,
              user_id: 1,
              agent_id: 42,
              type: "gmail",
              name: "Support inbox",
              enabled: true,
              config: {
                watch_label: "INBOX",
                sender_filter: "boss@company.com",
                subject_keyword: "urgent",
                oauth_account_id: 7,
              },
              prompt_template: "Reply to {{payload}}",
              webhook_token: null,
              webhook_secret: null,
              next_run_at: null,
              last_run_at: null,
              last_error: null,
              created_at: null,
              updated_at: null,
            },
          ]),
        )
      }
      if (url === "http://api.local/api/agents/42/triggers/9/runs") {
        return Promise.resolve(jsonResponse([]))
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

    fireEvent.click(screen.getByText("triggers.cards.gmail.title"))

    expect(await screen.findByLabelText("triggers.form.watchLabel")).toHaveValue("INBOX")
    expect(screen.getByText("triggers.form.watchLabelHelp")).toBeInTheDocument()
    expect(screen.getByLabelText("triggers.form.senderFilter")).toHaveValue("boss@company.com")
    expect(screen.getByLabelText("triggers.form.subjectKeyword")).toHaveValue("urgent")
    expect(screen.getByText("triggers.gmail.connected")).toBeInTheDocument()
    // Shown both in the account selector and the connection banner.
    expect(screen.getAllByText("gerard.santos@gmail.com").length).toBeGreaterThan(0)
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

    expect(await screen.findByText("triggers.form.gmailAccount")).toBeInTheDocument()
    expect(await screen.findByText("gerard.santos@gmail.com")).toBeInTheDocument()
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

    expect(
      await screen.findByText("triggers.form.gmailAccountPlaceholder"),
    ).toBeInTheDocument()
    expect(screen.queryByText("first@gmail.com")).not.toBeInTheDocument()
    expect(screen.queryByText("second@gmail.com")).not.toBeInTheDocument()
  })

  it("disables the account selector when no Gmail accounts are connected", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      return Promise.resolve(jsonResponse([]))
    })

    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        initialType="gmail"
        gmailConnection={{ isConnected: false, connectedAccount: null }}
      />,
    )

    expect(await screen.findByText("triggers.gmail.noAccounts")).toBeInTheDocument()
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

    expect(await screen.findByText("triggers.gmail.accountMissing")).toBeInTheDocument()
  })

  it("persists the detail switch immediately without pressing save", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    await screen.findByLabelText("triggers.form.watchLabel")

    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(detailSwitch)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers/9",
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({ enabled: false }),
        }),
      )
    })
    expect(detailSwitch).toHaveAttribute("aria-checked", "false")
  })

  it("shows the one-time webhook secret on the overview after a quick-toggle create", async () => {
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: 11,
            user_id: 1,
            agent_id: 42,
            type: "webhook",
            name: "API / Webhook",
            enabled: true,
            config: {},
            prompt_template: null,
            webhook_token: "tok",
            webhook_secret: "wh_secret_once",
            next_run_at: null,
            last_run_at: null,
            last_error: null,
            created_at: null,
            updated_at: null,
          }),
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

    await screen.findByText("triggers.cards.webhook.title")
    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)

    // The secret alert appears on the overview itself — no navigation happens.
    expect(await screen.findByText("wh_secret_once")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "common.back" })).not.toBeInTheDocument()
  })

  it("calls onChanged exactly once for a Done that commits an edit (no redundant refetch on close)", async () => {
    // Mirrors the builder's wiring: onChanged is the sole resync signal:
    // onOpenChange(false) must not ALSO trigger a refetch, or every Done
    // commits fires the same GET twice.
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
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Renamed inbox" } })
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    await waitFor(() => {
      expect(onChanged).toHaveBeenCalledTimes(1)
    })
  })

  it("creates the trigger when the switch is turned on in the creation state (live)", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    // The webhook type has no triggers yet, so this opens the creation form.
    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByLabelText("triggers.form.secret")

    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "false")
    fireEvent.click(detailSwitch)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/agents/42/triggers",
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining("\"enabled\":true"),
        }),
      )
    })
  })

  it("keeps the dialog open on Escape when a fresh create just revealed a webhook secret", async () => {
    const onOpenChange = vi.fn()
    apiRequestMock.mockImplementation((url: string, options?: { method?: string }) => {
      if (url === GMAIL_ACCOUNTS_URL) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/agents/42/triggers" && options?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: 12,
            user_id: 1,
            agent_id: 42,
            type: "webhook",
            name: "API / Webhook",
            enabled: true,
            config: {},
            prompt_template: null,
            webhook_token: "tok",
            webhook_secret: "wh_escape_secret",
            next_run_at: null,
            last_run_at: null,
            last_error: null,
            created_at: null,
            updated_at: null,
          }),
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

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByLabelText("triggers.form.secret")
    const [detailSwitch] = screen.getAllByRole("switch")
    fireEvent.click(detailSwitch)

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

  it("clears the dirty flag when the Gmail quick-toggle intent is reversed, so Done closes cleanly", async () => {
    const onOpenChange = vi.fn()
    // Zero connected accounts: the quick toggle must open the creation form
    // instead of silently auto-binding (the default mock has exactly one).
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
      />,
    )

    await screen.findByText("triggers.cards.webhook.title")
    const switches = screen.getAllByRole("switch")
    fireEvent.click(switches[2]) // Gmail card: no accounts connected

    await screen.findByLabelText("triggers.form.watchLabel")
    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "true")

    // Reverse the intent before picking an account.
    fireEvent.click(detailSwitch)
    expect(detailSwitch).toHaveAttribute("aria-checked", "false")

    // Nothing else was edited, so Done must close without attempting (and
    // failing) a phantom create.
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(toastMocks.error).not.toHaveBeenCalled()
  })

  it("keeps unsaved field edits when the detail switch is toggled", async () => {
    render(
      <AgentTriggersDialog
        agentId={42}
        open
        onOpenChange={vi.fn()}
        gmailConnection={{ isConnected: true, connectedAccount: null }}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.gmail.title"))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Edited but unsaved" } })

    const [detailSwitch] = screen.getAllByRole("switch")
    fireEvent.click(detailSwitch)

    await waitFor(() => {
      expect(detailSwitch).toHaveAttribute("aria-checked", "false")
    })
    expect(nameInput).toHaveValue("Edited but unsaved")
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

  it("stages a default trigger when a type is toggled on", async () => {
    const onChange = renderStaging([])

    expect(await screen.findByText("triggers.staging.info")).toBeInTheDocument()

    const [webhookSwitch] = screen.getAllByRole("switch")
    fireEvent.click(webhookSwitch)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({
          clientId: -1,
          type: "webhook",
          enabled: true,
          name: "triggers.defaults.webhookName",
        }),
      ])
    })

    // The toggle stays on the overview instead of jumping into the config view.
    expect(screen.getByText("triggers.staging.info")).toBeInTheDocument()
    expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()
  })

  it("starts a new trigger form with the switch off, matching the overview", async () => {
    renderStaging([])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByLabelText("triggers.form.name")

    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "false")
  })

  it("stages the trigger when the switch is turned on in the creation state", async () => {
    const onChange = renderStaging([])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Toggled hook" } })

    const [detailSwitch] = screen.getAllByRole("switch")
    fireEvent.click(detailSwitch)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({
          clientId: -1,
          type: "webhook",
          name: "Toggled hook",
          enabled: true,
        }),
      ])
    })
  })

  it("applies the detail switch to the staged trigger without pressing save", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Hook one")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    await screen.findByLabelText("triggers.form.name")

    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "true")
    fireEvent.click(detailSwitch)

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, enabled: false }),
      ])
    })
  })

  it("appends a new staged trigger via Add instead of overwriting the selected one", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "First hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("First hook")

    fireEvent.click(screen.getByRole("button", { name: /triggers.actions.addAnother/ }))

    // Creation state: empty form; delete lives on each existing pill's X
    // button, so exactly one remains (for "First hook").
    await waitFor(() => {
      expect(screen.getByLabelText("triggers.form.name")).toHaveValue("")
    })
    expect(screen.getAllByRole("button", { name: "triggers.actions.delete" })).toHaveLength(1)

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "Second hook" },
    })
    // No Save button: Done commits the pending creation.
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "First hook" }),
        // New forms default to disabled; the switch was not touched here.
        expect.objectContaining({ clientId: -2, name: "Second hook", type: "webhook", enabled: false }),
      ])
    })
  })

  it("lists staged triggers newest first and selects the primary one", async () => {
    renderStaging([stagedWebhook(-1, "Old hook"), stagedWebhook(-2, "New hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    // Newest staged trigger (-2) is the primary selection…
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("New hook")
    // …and precedes the older one in the picker.
    const newPill = screen.getByText("New hook")
    const oldPill = screen.getByText("Old hook")
    expect(
      newPill.compareDocumentPosition(oldPill) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("removes a staged trigger after confirming in the pill's popover", async () => {
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
    fireEvent.click(await screen.findByRole("button", { name: "common.cancel" }))

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: "triggers.actions.confirmDelete" }),
      ).not.toBeInTheDocument()
    })
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByText("Kept hook")).toBeInTheDocument()
  })

  it("keeps unsaved edits when another pill is deleted and the staged list round-trips", async () => {
    render(
      <StatefulStagingHarness
        initial={[stagedWebhook(-1, "Old hook"), stagedWebhook(-2, "New hook")]}
      />,
    )

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    expect(nameInput).toHaveValue("New hook")
    fireEvent.change(nameInput, { target: { value: "Unsaved edit" } })

    // Delete the non-selected pill ("Old hook"); the parent state update
    // re-renders the dialog with fresh pseudo-trigger identities.
    const [, oldHookDelete] = screen.getAllByRole("button", { name: "triggers.actions.delete" })
    fireEvent.click(oldHookDelete)
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(screen.queryByText("Old hook")).not.toBeInTheDocument()
    })
    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("Unsaved edit")
  })

  it("keeps the detail switch usable alongside unsaved edits after a round-trip", async () => {
    render(<StatefulStagingHarness initial={[stagedWebhook(-1, "Hook")]} />)

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Renamed but unsaved" } })

    // The immediate enabled toggle round-trips the staged list; the pending
    // name edit must survive it.
    const [detailSwitch] = screen.getAllByRole("switch")
    fireEvent.click(detailSwitch)
    await waitFor(() => {
      expect(detailSwitch).toHaveAttribute("aria-checked", "false")
    })
    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("Renamed but unsaved")
  })

  it("commits pending edits when the dialog is dismissed via Escape", async () => {
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
    const nameInput = await screen.findByLabelText("triggers.form.name")
    fireEvent.change(nameInput, { target: { value: "Saved on escape" } })

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(onChangeSpy).toHaveBeenCalledWith([
      expect.objectContaining({ clientId: -1, name: "Saved on escape" }),
    ])
  })

  it("keeps the Gmail quick-toggle intent as a dirty preset that Done cannot silently drop", async () => {
    const onOpenChange = vi.fn()
    render(
      <StatefulStagingHarness initial={[]} onOpenChange={onOpenChange} />,
    )

    await screen.findByText("triggers.staging.info")
    // No Gmail accounts connected: the quick toggle opens the creation form
    // with the enable intent preset instead of creating anything.
    const switches = screen.getAllByRole("switch")
    fireEvent.click(switches[2])

    await screen.findByLabelText("triggers.form.watchLabel")
    const [detailSwitch] = screen.getAllByRole("switch")
    expect(detailSwitch).toHaveAttribute("aria-checked", "true")

    // Done must attempt the creation and fail validation (no account picked),
    // keeping the dialog open rather than silently dropping the intent.
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))
    await waitFor(() => {
      expect(toastMocks.error).toHaveBeenCalled()
    })
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })

  it("keeps the form being edited when another pill is deleted", async () => {
    const onChange = renderStaging([
      stagedWebhook(-1, "Old hook"),
      stagedWebhook(-2, "New hook"),
    ])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    // Newest (-2) is selected; edit its name without saving.
    const nameInput = await screen.findByLabelText("triggers.form.name")
    expect(nameInput).toHaveValue("New hook")
    fireEvent.change(nameInput, { target: { value: "Unsaved edit" } })

    // Delete the other pill (-1, "Old hook") via its X button.
    const [, oldHookDelete] = screen.getAllByRole("button", { name: "triggers.actions.delete" })
    fireEvent.click(oldHookDelete)
    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -2, name: "New hook" }),
      ])
    })
    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("Unsaved edit")
  })

  it("commits pending edits to the selected staged trigger on Done", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Old name")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("Old name")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "New name" },
    })
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "New name", type: "webhook" }),
      ])
    })
  })

  it("commits pending edits when navigating back to the overview", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Old name")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("Old name")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "Renamed on back" },
    })
    fireEvent.click(screen.getByRole("button", { name: "common.back" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "Renamed on back" }),
      ])
    })
    // Back landed on the overview.
    expect(screen.queryByLabelText("triggers.form.name")).not.toBeInTheDocument()
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
    await screen.findByLabelText("triggers.form.name")
    fireEvent.click(screen.getByRole("button", { name: "common.done" }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(onChange).not.toHaveBeenCalled()
  })

  it("disables every staged trigger of a type when its switch is toggled off", async () => {
    const onChange = renderStaging([
      stagedWebhook(-1, "Hook one"),
      stagedWebhook(-2, "Hook two"),
    ])

    await screen.findByText("triggers.staging.info")
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
})
