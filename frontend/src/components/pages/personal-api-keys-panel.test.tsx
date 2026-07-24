import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { PersonalApiKeyListResponse } from "@/lib/personal-api-keys-api"

const createPersonalApiKeyMock = vi.hoisted(() => vi.fn())
const listPersonalApiKeysMock = vi.hoisted(() => vi.fn())
const revokePersonalApiKeyMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string, vars?: Record<string, string>) => {
  const translations: Record<string, string> = {
    "personalApiKeys.create": "Create Personal Key",
    "personalApiKeys.createForMe": "Create Personal Key for Me",
    "personalApiKeys.columns.owner": "Owner",
    "personalApiKeys.actions.revoke": "Revoke",
    "personalApiKeys.actions.copy": "Copy personal API key",
    "personalApiKeys.columns.key": "Secret Key",
    "personalApiKeys.columns.status": "Status",
    "personalApiKeys.columns.expires": "Expiry",
    "personalApiKeys.columns.created": "Created",
    "personalApiKeys.status.active": "Active",
    "personalApiKeys.status.expired": "Expired",
    "personalApiKeys.status.revoked": "Revoked",
    "personalApiKeys.neverExpires": "Never",
    "personalApiKeys.reveal.title": "Personal API Key Created",
    "personalApiKeys.reveal.warning": "Copy this key now — it is shown only once.",
    "personalApiKeys.confirm.revokeTitle": "Revoke personal API key?",
    "personalApiKeys.confirm.revokeOwnDescription": "Revoking immediately invalidates this key.",
    "personalApiKeys.confirm.revokeOtherDescription": "Revoke this personal key for {owner}?",
  }
  return (translations[key] ?? key).replace("{owner}", vars?.owner ?? "")
}))

vi.mock("@/lib/personal-api-keys-api", () => ({
  createPersonalApiKey: createPersonalApiKeyMock,
  listPersonalApiKeys: listPersonalApiKeysMock,
  revokePersonalApiKey: revokePersonalApiKeyMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: translateMock,
  }),
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) => open ? <div>{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}))

vi.mock("@/components/ui/confirm-dialog", () => ({
  ConfirmDialog: ({ isOpen, description, confirmText, onConfirm }: {
    isOpen: boolean
    description?: string
    confirmText?: string
    onConfirm: () => void
  }) => isOpen ? <div><p>{description}</p><button onClick={onConfirm}>{confirmText}</button></div> : null,
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import { PersonalApiKeysPanel } from "./personal-api-keys-panel"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

function PersonalKeysTabsHarness() {
  const [tab, setTab] = React.useState("personal")

  return (
    <Tabs value={tab} onValueChange={setTab}>
      <TabsList>
        <TabsTrigger value="agent">Agent Keys</TabsTrigger>
        <TabsTrigger value="personal">Personal Keys</TabsTrigger>
      </TabsList>
      <TabsContent value="agent">agent-panel</TabsContent>
      <TabsContent value="personal" forceMount>
        <PersonalApiKeysPanel active={tab === "personal"} />
      </TabsContent>
    </Tabs>
  )
}

function listResponse(canManageOthers: boolean): PersonalApiKeyListResponse {
  const items = [
    {
      id: 1,
      key_prefix: "self123",
      masked_key: "xag_personal_self123_••••••••",
      status: "active" as const,
      revoked_at: null as string | null,
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 1, username: "alice", email: "alice@example.com" },
    },
  ]

  if (canManageOthers) {
    items.push({
      id: 2,
      key_prefix: "other456",
      masked_key: "xag_personal_other456_••••••••",
      status: "active" as const,
      revoked_at: null,
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 2, username: "bob", email: "bob@example.com" },
    })
  }

  return {
    can_manage_others: canManageOthers,
    items,
  }
}

describe("PersonalApiKeysPanel", () => {
  beforeEach(() => {
    createPersonalApiKeyMock.mockReset()
    listPersonalApiKeysMock.mockReset()
    revokePersonalApiKeyMock.mockReset()
  })

  afterEach(cleanup)

  it("loads only after the Personal tab is first activated", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))

    const { rerender } = render(<PersonalApiKeysPanel active={false} />)

    expect(listPersonalApiKeysMock).not.toHaveBeenCalled()

    rerender(<PersonalApiKeysPanel active />)
    await waitFor(() => expect(listPersonalApiKeysMock).toHaveBeenCalledOnce())

    rerender(<PersonalApiKeysPanel active={false} />)
    rerender(<PersonalApiKeysPanel active />)
    expect(listPersonalApiKeysMock).toHaveBeenCalledOnce()
  })

  it("renders a self-only list without owner controls", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))

    render(<PersonalApiKeysPanel active />)

    expect(await screen.findByText("xag_personal_self123_••••••••")).toBeInTheDocument()
    expect(screen.queryByText("bob")).not.toBeInTheDocument()
    expect(screen.queryByText("Owner")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Personal Key" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Create Personal Key for Me" })).not.toBeInTheDocument()
  })

  it("renders owner data and the self-only creation copy for managed scopes", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(true))

    render(<PersonalApiKeysPanel active />)

    expect(await screen.findByText("bob")).toBeInTheDocument()
    expect(screen.getByText("Owner")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create Personal Key for Me" })).toBeInTheDocument()
  })

  it("reveals the plaintext key only after creation", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))
    createPersonalApiKeyMock.mockResolvedValue({
      id: 3,
      full_key: "xag_personal_created_secret",
      key_prefix: "created",
      created_at: "2026-07-22T00:00:00Z",
      expires_at: null,
    })

    render(<PersonalApiKeysPanel active />)

    await screen.findByText("xag_personal_self123_••••••••")
    expect(screen.queryByText("xag_personal_created_secret")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create Personal Key" }))

    expect(await screen.findByText("xag_personal_created_secret")).toBeInTheDocument()
    expect(createPersonalApiKeyMock).toHaveBeenCalledOnce()
    expect(screen.getByText("Copy this key now — it is shown only once.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Copy personal API key" })).toBeInTheDocument()
  })

  it("names another owner in the revoke confirmation", async () => {
    listPersonalApiKeysMock.mockResolvedValue(listResponse(true))

    render(<PersonalApiKeysPanel active />)

    await screen.findByText("bob")
    fireEvent.click(screen.getAllByRole("button", { name: "Revoke" })[1])

    expect(screen.getByText("Revoke this personal key for bob?")).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Revoke" })[2])
    await waitFor(() => expect(revokePersonalApiKeyMock).toHaveBeenCalledWith(2))
  })

  it("keeps a mutation-success refresh when an earlier list request resolves late", async () => {
    let resolveInitial: (value: ReturnType<typeof listResponse>) => void
    const initial = new Promise<ReturnType<typeof listResponse>>((resolve) => {
      resolveInitial = resolve
    })
    const refreshed = listResponse(false)
    refreshed.items.push({
      id: 3,
      key_prefix: "new789",
      masked_key: "xag_personal_new789_••••••••",
      status: "active",
      revoked_at: null,
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 1, username: "alice", email: "alice@example.com" },
    })
    const stale = listResponse(false)
    stale.items[0].masked_key = "xag_personal_stale000_••••••••"

    listPersonalApiKeysMock
      .mockReturnValueOnce(initial)
      .mockResolvedValueOnce(refreshed)
    createPersonalApiKeyMock.mockResolvedValue({
      id: 3,
      full_key: "xag_personal_new789_secret",
      key_prefix: "new789",
      created_at: "2026-07-22T00:00:00Z",
      expires_at: null,
    })

    render(<PersonalApiKeysPanel active />)

    fireEvent.click(screen.getByRole("button", { name: "Create Personal Key" }))
    expect(await screen.findByText("xag_personal_new789_••••••••")).toBeInTheDocument()

    await act(async () => resolveInitial!(stale))

    expect(screen.queryByText("xag_personal_stale000_••••••••")).not.toBeInTheDocument()
    expect(screen.getByText("xag_personal_new789_••••••••")).toBeInTheDocument()
  })

  it("does not offer Revoke for a revoked personal key", async () => {
    const response = listResponse(false)
    response.items.push({
      id: 4,
      key_prefix: "revoked",
      masked_key: "xag_personal_revoked_••••••••",
      status: "revoked",
      revoked_at: "2026-07-22T00:00:00Z",
      expires_at: null,
      created_at: "2026-07-22T00:00:00Z",
      owner: { id: 1, username: "alice", email: "alice@example.com" },
    })
    listPersonalApiKeysMock.mockResolvedValue(response)

    render(<PersonalApiKeysPanel active />)

    const revokedRow = (await screen.findByText("xag_personal_revoked_••••••••")).closest("tr")
    expect(revokedRow).not.toBeNull()
    expect(within(revokedRow!).queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument()
  })

  it("renders lifecycle status and expiry and only offers Revoke for active keys", async () => {
    const response = listResponse(false)
    const expiry = "2030-01-02T03:04:05Z"
    response.items[0].expires_at = expiry
    response.items.push(
      {
        id: 6,
        key_prefix: "expired",
        masked_key: "xag_personal_expired_••••••••",
        status: "expired",
        revoked_at: null,
        expires_at: "2020-01-02T03:04:05Z",
        created_at: "2020-01-01T00:00:00Z",
        owner: { id: 1, username: "alice", email: "alice@example.com" },
      },
      {
        id: 7,
        key_prefix: "revoked-expired",
        masked_key: "xag_personal_revoked-expired_••••••••",
        status: "revoked",
        revoked_at: "2020-01-03T00:00:00Z",
        expires_at: "2020-01-02T03:04:05Z",
        created_at: "2020-01-01T00:00:00Z",
        owner: { id: 1, username: "alice", email: "alice@example.com" },
      },
    )
    listPersonalApiKeysMock.mockResolvedValue(response)

    render(<PersonalApiKeysPanel active />)

    const activeRow = (await screen.findByText("xag_personal_self123_••••••••")).closest("tr")
    const expiredRow = screen.getByText("xag_personal_expired_••••••••").closest("tr")
    const revokedRow = screen.getByText("xag_personal_revoked-expired_••••••••").closest("tr")
    expect(screen.getByText("Status")).toBeInTheDocument()
    expect(screen.getByText("Expiry")).toBeInTheDocument()
    expect(within(activeRow!).getByText("Active")).toBeInTheDocument()
    expect(within(activeRow!).getByText(new Date(expiry).toLocaleDateString())).toBeInTheDocument()
    expect(within(activeRow!).getByRole("button", { name: "Revoke" })).toBeInTheDocument()
    expect(within(expiredRow!).getByText("Expired")).toBeInTheDocument()
    expect(within(expiredRow!).queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument()
    expect(within(revokedRow!).getByText("Revoked")).toBeInTheDocument()
    expect(within(revokedRow!).queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument()
  })

  it("keeps a deferred create reveal while the Agent tab is selected", async () => {
    let resolveCreate: (value: {
      id: number
      full_key: string
      key_prefix: string
      created_at: string
      expires_at: null
    }) => void
    const create = new Promise<{
      id: number
      full_key: string
      key_prefix: string
      created_at: string
      expires_at: null
    }>((resolve) => {
      resolveCreate = resolve
    })
    listPersonalApiKeysMock.mockResolvedValue(listResponse(false))
    createPersonalApiKeyMock.mockReturnValue(create)

    render(<PersonalKeysTabsHarness />)

    await screen.findByText("xag_personal_self123_••••••••")
    fireEvent.click(screen.getByRole("button", { name: "Create Personal Key" }))
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Agent Keys" }), { button: 0 })
    expect(screen.getByText("agent-panel")).toBeVisible()

    await act(async () => resolveCreate!({
      id: 5,
      full_key: "xag_personal_deferred_secret",
      key_prefix: "deferred",
      created_at: "2026-07-22T00:00:00Z",
      expires_at: null,
    }))

    fireEvent.mouseDown(screen.getByRole("tab", { name: "Personal Keys" }), { button: 0 })
    expect(screen.getByText("xag_personal_deferred_secret")).toBeVisible()
  })
})
