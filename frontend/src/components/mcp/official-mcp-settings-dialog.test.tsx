import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { OfficialMcpSettingsDialog } from "./official-mcp-settings-dialog"
import type { AppIntegration } from "./types"

const apiRequestMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

const authState = vi.hoisted(() => ({ inTeam: false }))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token", inTeam: authState.inTeam }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}))

function app(overrides: Partial<AppIntegration> = {}): AppIntegration {
  return {
    id: "chrome",
    name: "Chrome",
    description: "Automate a Chrome browser.",
    icon: "",
    is_connected: true,
    transport: "stdio",
    auth_type: "keyless",
    ...overrides,
  } as AppIntegration
}

function renderDialog(props: Partial<React.ComponentProps<typeof OfficialMcpSettingsDialog>> = {}) {
  return render(
    <OfficialMcpSettingsDialog
      open
      onOpenChange={vi.fn()}
      app={app()}
      isGloballyConnected
      {...props}
    />,
  )
}

describe("OfficialMcpSettingsDialog connected-state actions", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
    authState.inTeam = false
  })

  it("offers only disconnect for a connected keyless app", () => {
    // canConfigure: true and it must still not show -- proving the keyless
    // gate is independent of configurability, not a side effect of the
    // connected-entry default being false somewhere.
    renderDialog({ canConfigure: true })

    // No key to manage and no editable config — the configure/manage-key
    // button must be suppressed for keyless apps.
    expect(
      screen.queryByRole("button", { name: /manageKey|configure/ }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /disconnect/ }),
    ).toBeInTheDocument()
  })

  it("still offers manage-key for a connected api_key app", () => {
    const onManageKey = vi.fn()
    renderDialog({
      app: app({ id: "google-maps", name: "Google Maps", auth_type: "api_key" }),
      onManageKey,
    })

    const manageButton = screen.getByRole("button", { name: /manageKey/ })
    fireEvent.click(manageButton)
    expect(onManageKey).toHaveBeenCalledTimes(1)
  })

  it("offers Configure for an unconnected entry the viewer may still configure", () => {
    // This is the bug's shape in the settings dialog: is_connected is false
    // (no grant was ever written for a hook-resolved connector), yet the
    // owner's personal association means the edit route would resolve.
    const onConfigure = vi.fn()
    renderDialog({
      app: app({
        is_connected: false,
        is_custom: true,
        auth_type: "mcp_oauth",
        transport: "streamable_http",
        server_id: 9,
      }),
      isGloballyConnected: false,
      canConfigure: true,
      onConfigure,
    })

    const configureButton = screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    fireEvent.click(configureButton)
    expect(onConfigure).toHaveBeenCalledTimes(1)
    // Neither element gated on isGloballyConnected renders in this shape:
    // Configure is the only door this population gets.
    expect(
      screen.queryByRole("button", { name: /disconnect|deleteService/ }),
    ).toBeNull()
    expect(screen.queryByRole("button", { name: /share|unshare/ })).toBeNull()
  })

  it("keeps sharing and disconnect on the connection gate when only Configure is unlocked", () => {
    authState.inTeam = true
    const onConfigure = vi.fn()
    renderDialog({
      app: app({
        server_id: 9,
        transport: "streamable_http",
        auth_type: "mcp_oauth",
        is_custom: true,
      }),
      isGloballyConnected: false, // the precondition this test proves against
      canConfigure: true,
      onConfigure,
    })

    // Configure appears (the new gate).
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    // Share does not: inTeam is true and server_id is an integer, so the only
    // thing still withholding it is isGloballyConnected being false — exactly
    // what this test exists to pin.
    expect(screen.queryByRole("button", { name: /share|unshare/ })).toBeNull()
    // Disconnect does not either -- its gate does not read inTeam at all, so
    // this holds under either harness default.
    expect(screen.queryByRole("button", { name: /disconnect|deleteService/ })).toBeNull()
  })

  it("withholds Configure for a connected entry the viewer may not configure", () => {
    renderDialog({
      app: app({ auth_type: "mcp_oauth", is_custom: true, server_id: 9 }),
      isGloballyConnected: true,
      canConfigure: false,
    })

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
    // Disconnect is unaffected -- only Configure was withheld.
    screen.getByRole("button", { name: /disconnect|deleteService/ })
  })

  it("falls back to the connection gate when no canConfigure is provided", () => {
    // The Tools page's call site: it always passes isGloballyConnected={true}
    // and never passes canConfigure at all.
    renderDialog({
      app: app({ auth_type: "api_key" }),
      isGloballyConnected: true,
    })

    screen.getByRole("button", { name: "tools.mcp.dialog.manageKey" })
  })

  it("withholds Configure when no canConfigure is provided and the connection gate is closed", () => {
    // Mirrors the fallback case above with isGloballyConnected flipped to
    // false: the omitted-prop shape must read the connection gate rather
    // than defaulting Configure on regardless of it.
    renderDialog({
      app: app({ auth_type: "mcp_oauth" }),
      isGloballyConnected: false,
    })

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
  })

  it("withholds Configure for a non-owned team tool even when configurable", () => {
    renderDialog({
      app: app({
        auth_type: "mcp_oauth",
        is_custom: true,
        server_id: 9,
        shared: true,
        is_owner: false,
      }),
      isGloballyConnected: true,
      canConfigure: true,
    })

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
  })

  it("shows the ownership badge for an unconnected entry that carries a sharing status (#1623)", () => {
    authState.inTeam = true
    renderDialog({
      app: app({
        auth_type: "mcp_oauth",
        is_custom: true,
        server_id: 9,
        shared: true,
        is_owner: false,
        needs_config: false,
      }),
      isGloballyConnected: false,
    })

    expect(screen.getByText("tools.mcp.sharing.teamTool")).toBeInTheDocument()
  })

  it("withholds the ownership badge for an entry with no sharing status (#1623)", () => {
    authState.inTeam = true
    // The Tools page's hand-built shape: a server_id, isGloballyConnected
    // true, and no shared/is_owner/needs_config at all.
    renderDialog({
      app: app({ auth_type: "builtin_oauth", server_id: 9 }),
      isGloballyConnected: true,
    })

    // Anchor: proves the dialog actually rendered its content, so the three
    // negatives below aren't trivially satisfied by an empty document (the
    // component's own `if (!app) return null` is the failure mode this
    // guards against).
    expect(screen.getByText("Chrome")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(screen.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(screen.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
  })

  it("withholds the ownership badge for a team entry with no connector id (#1623)", () => {
    authState.inTeam = true
    // A well-formed sharing triple, but no server_id at all -- the factory
    // default carries none -- so the badge must stay withheld regardless of
    // shared/is_owner/needs_config.
    renderDialog({
      app: app({ auth_type: "mcp_oauth", shared: true, is_owner: false, needs_config: false }),
      isGloballyConnected: true,
    })

    // Anchor: proves the dialog actually rendered its content before the
    // three negatives below are checked.
    expect(screen.getByText("Chrome")).toBeInTheDocument()
    expect(screen.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(screen.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(screen.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
  })
})

describe("OfficialMcpSettingsDialog connect trigger", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("fires onConnectStart once and shows a pending state while connecting", () => {
    const onConnectStart = vi.fn()
    const { rerender } = renderDialog({
      app: app({ is_connected: false }),
      isGloballyConnected: false,
      onConnectStart,
    })

    const connectButton = screen.getByRole("button", { name: /dialog\.connect$/ })
    fireEvent.click(connectButton)
    expect(onConnectStart).toHaveBeenCalledTimes(1)

    // Parent flips isConnecting while its POST is in flight: the trigger
    // must disable (rapid double-clicks fired overlapping requests before)
    // and show the pending label — the catalog card spinner is hidden
    // behind this dialog, so this is the flow's only visible feedback.
    rerender(
      <OfficialMcpSettingsDialog
        open
        onOpenChange={vi.fn()}
        app={app({ is_connected: false })}
        isGloballyConnected={false}
        onConnectStart={onConnectStart}
        isConnecting
      />,
    )

    const pendingButton = screen.getByRole("button", { name: /connecting/ })
    expect(pendingButton).toBeDisabled()
    fireEvent.click(pendingButton)
    expect(onConnectStart).toHaveBeenCalledTimes(1)
  })
})
