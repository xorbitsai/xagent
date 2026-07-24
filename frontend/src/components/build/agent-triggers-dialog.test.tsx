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

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
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
  })

  it("appends a new staged trigger via Add instead of overwriting the selected one", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "First hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("First hook")

    fireEvent.click(screen.getByRole("button", { name: /triggers.actions.addAnother/ }))

    // Creation state: empty form and the create label, no leaked delete action.
    expect(screen.getByLabelText("triggers.form.name")).toHaveValue("")
    expect(screen.getByRole("button", { name: "triggers.actions.enable" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "triggers.actions.delete" })).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "Second hook" },
    })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.enable" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "First hook" }),
        expect.objectContaining({ clientId: -2, name: "Second hook", type: "webhook" }),
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

  it("removes a staged trigger after delete confirmation", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Doomed hook")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))

    fireEvent.click(await screen.findByRole("button", { name: "triggers.actions.delete" }))
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.confirmDelete" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([])
    })
  })

  it("updates the selected staged trigger in place on save", async () => {
    const onChange = renderStaging([stagedWebhook(-1, "Old name")])

    fireEvent.click(await screen.findByText("triggers.cards.webhook.title"))
    expect(await screen.findByLabelText("triggers.form.name")).toHaveValue("Old name")

    fireEvent.change(screen.getByLabelText("triggers.form.name"), {
      target: { value: "New name" },
    })
    fireEvent.click(screen.getByRole("button", { name: "triggers.actions.save" }))

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith([
        expect.objectContaining({ clientId: -1, name: "New name", type: "webhook" }),
      ])
    })
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
