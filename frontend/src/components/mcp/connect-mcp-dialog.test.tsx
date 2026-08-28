import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ConnectMcpDialog } from "./connect-mcp-dialog"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const toastWarningMock = vi.hoisted(() => vi.fn())
const useAuthMock = vi.hoisted(() => vi.fn(() => ({ token: "token", inTeam: false })))
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: useAuthMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [] }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: toastSuccessMock, warning: toastWarningMock },
}))

vi.mock("@/components/ui/dialog", () => ({
  // role="dialog" + both Radix dismiss paths wired to the real onOpenChange
  // prop — Escape via keyDown and outside-click via the overlay button — so
  // tests exercise the component's own onOpenChange guard logic (e.g.
  // refusing to close while a connect request is in flight) through the
  // same callback Radix would invoke, instead of just no-op'ing it.
  Dialog: ({
    open,
    onOpenChange,
    children,
  }: {
    open: boolean
    onOpenChange?: (open: boolean) => void
    children: React.ReactNode
  }) =>
    open ? (
      <div
        role="dialog"
        onKeyDown={(event) => {
          if (event.key === "Escape") onOpenChange?.(false)
        }}
      >
        <button
          type="button"
          data-testid="dialog-overlay"
          onClick={() => onOpenChange?.(false)}
        >
          overlay
        </button>
        {children}
      </div>
    ) : null,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
}))

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsTrigger: ({ children, onClick }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button type="button" onClick={onClick}>{children}</button>
  ),
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock("./custom-api-form", () => ({
  CustomApiForm: ({
    mcpFormData,
    setMcpFormData,
  }: {
    mcpFormData: { name?: string }
    setMcpFormData: React.Dispatch<React.SetStateAction<Record<string, unknown>>>
  }) => (
    <div>
      <output data-testid="custom-api-edit-name">{mcpFormData.name ?? ""}</output>
      {/* Fills the two fields handleSaveCustomMcp requires before it will
          POST a create (name, url), for the #1390 custom_api create path. */}
      <button
        type="button"
        onClick={() => setMcpFormData((previous) => ({
          ...previous,
          name: "records-mcp",
          transport: "custom_api",
          url: "https://api.example.com",
          method: "POST",
        }))}
      >
        name-new-custom-api
      </button>
    </div>
  ),
}))

vi.mock("./custom-mcp-form", () => ({
  CustomMcpForm: ({
    mcpFormData,
    setMcpFormData,
  }: {
    mcpFormData: { user_env?: Record<string, string> }
    setMcpFormData: React.Dispatch<React.SetStateAction<Record<string, unknown>>>
  }) => (
    <div>
      <output data-testid="mcp-edit-state">{JSON.stringify(mcpFormData)}</output>
      <button
        type="button"
        onClick={() => setMcpFormData((previous) => ({
          ...previous,
          description: "Updated MCP description",
        }))}
      >
        change-mcp-description
      </button>
      {/* The create path has no detail fetch to hydrate the form, so the
          #1390 tests fill in the one field handleSaveCustomMcp validates
          (and the one the post-create lookup matches the listing on). */}
      <button
        type="button"
        onClick={() => setMcpFormData((previous) => ({
          ...previous,
          name: "records-mcp",
          transport: "streamable_http",
          config: { url: "https://mcp.example.com" },
        }))}
      >
        name-new-mcp
      </button>
      <button
        type="button"
        onClick={() => setMcpFormData((previous) => ({
          ...previous,
          user_env: {
            ...((previous.user_env as Record<string, string> | undefined) ?? {}),
            NEW_TOKEN: "new-secret",
          },
        }))}
      >
        add-mcp-env
      </button>
    </div>
  ),
}))

vi.mock("./official-mcp-settings-dialog", () => ({
  OfficialMcpSettingsDialog: ({
    app,
    open,
    isConnecting,
    onConfigure,
    onOpenChange,
    onConnectStart,
  }: {
    app?: { name?: string } | null
    open?: boolean
    isConnecting?: boolean
    onConfigure: (app: object) => void
    onOpenChange: (open: boolean) => void
    onConnectStart: (app: object) => void
  }) => (
    <div>
      {/* Surfaces the real component's isConnecting prop for the cross-app
          clobber regression test — this dialog mock is shared by every app,
          so the value reflects whichever app is currently selectedApp. */}
      <div data-testid="settings-is-connecting">
        {isConnecting ? "connecting" : "idle"}
      </div>
      {/* Which app the detail modal is open for (empty when closed). The mock
          renders unconditionally, so `open` is the only way a test can tell
          whether a card click routed to the modal or toggled the selection. */}
      <div data-testid="settings-open-app">{open ? app?.name ?? "" : ""}</div>
      <button
        type="button"
        onClick={() => onConfigure(customApiApp(1, "aggregated-a"))}
      >
        configure-a
      </button>
      <button
        type="button"
        onClick={() => onConfigure(customApiApp(2, "aggregated-b"))}
      >
        configure-b
      </button>
      <button type="button" onClick={() => onConfigure(mcpApp(3, "aggregated-mcp"))}>
        configure-mcp
      </button>
      <button type="button" onClick={() => onConnectStart(mcpOauthApp())}>
        connect-granola
      </button>
      <button type="button" onClick={() => onConnectStart(customMcpOauthApp())}>
        connect-records
      </button>
      <button type="button" onClick={() => onConnectStart(keylessApp())}>
        connect-chrome
      </button>
      {/* A second, distinct keyless app for the cross-app clobber
          regression test — same trigger shape as connect-chrome, different
          app id, so a test can fire both without needing real Card-driven
          app selection. */}
      <button type="button" onClick={() => onConnectStart(keylessApp("keyless-b", "Keyless B"))}>
        connect-keyless-b
      </button>
      <button type="button" onClick={() => onConnectStart(builtinOauthApp())}>
        connect-builtin
      </button>
      <button type="button" onClick={() => onConnectStart(apiKeyApp())}>
        connect-apikey
      </button>
      <button type="button" onClick={() => onOpenChange(false)}>
        close-settings
      </button>
    </div>
  ),
}))

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

// The fixtures below carry the can_attach/can_authorize pair list_mcp_apps
// emits for each shape (#1347). They are the picker's only inputs for both
// decisions now, so any fixture reaching a select-mode assertion must carry
// them — omitting the pair models a connector the backend never emits, and
// renders the card unattachable with no other symptom. Inline literals in
// tests that never enter select mode (the connect-flow tests below) are
// exempt, and are left as they are rather than gaining two inert keys each.
function customApiApp(id: number, name: string) {
  return {
    id: name,
    name,
    description: "",
    icon: "",
    is_custom: true,
    is_connected: true,
    is_local: true,
    server_id: id,
    transport: "custom_api",
    // A Custom API has no OAuth consent step of any kind.
    can_attach: true,
    can_authorize: false,
  }
}

function mcpApp(id: number, name: string) {
  return {
    id: name,
    name,
    description: "",
    icon: "",
    is_custom: true,
    is_connected: true,
    is_local: true,
    server_id: id,
    transport: "streamable_http",
    can_attach: true,
    can_authorize: false,
  }
}

// A remote-MCP OAuth (DCR) catalog app, e.g. Granola: no provider, no
// launch command — connecting must go through /apps/{id}/oauth/connect.
function mcpOauthApp() {
  return {
    id: "granola",
    name: "Granola",
    description: "",
    icon: "",
    is_connected: false,
    transport: "streamable_http",
    auth_type: "mcp_oauth",
    // Unconnected catalog app: no association row exists for this user, so
    // there is no connector to attach. can_authorize is false for every
    // catalog entry — their Connect is dispatched on auth_type, through
    // /apps/{id}/oauth/connect rather than the per-server route.
    can_attach: false,
    can_authorize: false,
    // The catalog branch's can_configure equals is_connected: its Configure
    // equivalent (manage-my-key / re-run OAuth) only exists once connected.
    can_configure: false,
  }
}

// A user-added mcp_oauth MCP server whose consent was never completed: it is
// listed under location=local with a server_id, and authorizing it must go
// through the per-server /api/mcp/{server_id}/oauth/connect — the catalog
// route resolves app ids only and can never see it (#1313).
function customMcpOauthApp() {
  return {
    id: "records",
    name: "records",
    description: "",
    icon: "",
    is_connected: false,
    is_custom: true,
    is_local: true,
    server_id: 9,
    transport: "streamable_http",
    auth_type: "mcp_oauth",
    // Standalone shape: nothing can supply tokens for a server whose consent
    // was never completed, so attaching it would fail at run time — but the
    // consent flow itself is available and is the recovery.
    can_attach: false,
    can_authorize: true,
    // The owner holds the personal association the edit routes require, so
    // this base shape is configurable -- independent of can_attach/
    // can_authorize, which is the entire point of the field.
    can_configure: true,
  }
}

// A keyless catalog app, e.g. Chrome: a local stdio command with no secrets —
// connecting POSTs straight to /apps/{id}/connect with no env and no dialog.
function keylessApp(id = "chrome", name = "Chrome") {
  return {
    id,
    name,
    description: "",
    icon: "",
    is_connected: false,
    transport: "stdio",
    auth_type: "keyless",
    // Every unconnected catalog entry: no association row, so nothing to
    // attach, and its Connect is dispatched on auth_type, not can_authorize.
    can_attach: false,
    can_authorize: false,
  }
}

// A static-provider builtin OAuth catalog app, e.g. Zoom/Gmail.
function builtinOauthApp() {
  return {
    id: "zoom",
    name: "Zoom",
    description: "",
    icon: "",
    is_connected: false,
    transport: "oauth",
    auth_type: "builtin_oauth",
    provider: "zoom",
    can_attach: false,
    can_authorize: false,
  }
}

// A key-based catalog app, e.g. Google Maps: opens the real (unmocked)
// key-entry dialog via openKeyConnect, unlike the other fixtures above which
// all bypass it.
function apiKeyApp() {
  return {
    id: "google-maps",
    name: "Google Maps",
    description: "",
    icon: "",
    is_connected: false,
    transport: "stdio",
    auth_type: "api_key",
    launch_config: { required_env: ["GOOGLE_MAPS_API_KEY"] },
    can_attach: false,
    can_authorize: false,
  }
}

function detailResponse(id: number, name: string) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      id,
      user_id: 7,
      name,
      description: `${name} description`,
      url: `https://${name}.example.com`,
      method: "POST",
      headers: { "X-Test": name },
      body: "{}",
      env: { TOKEN: "********" },
      runtime_input_schema: {
        type: "object",
        properties: { delegated_token: { type: "string" } },
      },
      runtime_bindings: [{ source: "delegated_token", target: "header.Authorization" }],
      allow_delegated_authorization: true,
    }),
  }
}

function mcpDetailResponse(id: number, name: string) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      id,
      user_id: 7,
      name,
      transport: "streamable_http",
      description: "authoritative MCP",
      config: { url: "https://mcp.example.com" },
      is_active: true,
      is_default: false,
      user_env: { EXISTING_TOKEN: "********" },
      env_source: "own",
      runtime_input_schema: {
        context: { account_id: { type: "string", required: true } },
      },
      runtime_bindings: [
        {
          source: { input_type: "context", key: "account_id" },
          target: { target_type: "mcp_meta", key: "account_id" },
        },
      ],
      allow_delegated_authorization: true,
      can_edit_global: false,
      transport_display: "Streamable HTTP",
      created_at: null,
      updated_at: null,
    }),
  }
}

const selectedMcpServers: string[] = []
function renderDialog() {
  return render(
    <ConnectMcpDialog
      open
      onOpenChange={vi.fn()}
      selectedMcpServers={selectedMcpServers}
    />,
  )
}

function saveMcpEditor() {
  const editor = screen.getByTestId("mcp-edit-state").closest(".max-w-2xl")
  if (!editor) throw new Error("MCP editor container was not rendered")
  fireEvent.click(within(editor as HTMLElement).getByRole("button", {
    name: "tools.mcp.buttons.save",
  }))
}

// The Custom API tab's own Save, which is a different button under a
// different container from the MCP editor's — both tabs render at once
// under the stubbed Tabs, so the container has to be the anchor.
function saveCustomApiEditor() {
  const editor = screen.getByTestId("custom-api-edit-name").closest(".max-w-2xl")
  if (!editor) throw new Error("Custom API editor container was not rendered")
  fireEvent.click(within(editor as HTMLElement).getByRole("button", {
    name: "tools.mcp.buttons.save",
  }))
}

describe("ConnectMcpDialog Custom API detail loading", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
    toastWarningMock.mockReset()
    // Clear call history only (mockClear, not mockReset): the identity
    // implementation must survive, and without clearing, an interpolation
    // assertion like toHaveBeenCalledWith(key, { name }) could be satisfied
    // by a matching call from an earlier test in this file.
    translateMock.mockClear()
    useAuthMock.mockReturnValue({ token: "token", inTeam: false })
  })

  afterEach(() => {
    cleanup()
  })

  it("keeps the latest requested Custom API when detail responses finish out of order", async () => {
    const detailA = deferred<ReturnType<typeof detailResponse>>()
    const detailB = deferred<ReturnType<typeof detailResponse>>()

    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url.endsWith("/api/custom-apis/1")) return detailA.promise
      if (url.endsWith("/api/custom-apis/2")) return detailB.promise
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "configure-a" }))
    fireEvent.click(screen.getByRole("button", { name: "configure-b" }))

    await act(async () => {
      detailB.resolve(detailResponse(2, "authoritative-b"))
      await detailB.promise
    })

    await waitFor(() => {
      expect(screen.getByTestId("custom-api-edit-name")).toHaveTextContent("authoritative-b")
    })

    await act(async () => {
      detailA.resolve(detailResponse(1, "authoritative-a"))
      await detailA.promise
    })

    expect(screen.getByTestId("custom-api-edit-name")).toHaveTextContent("authoritative-b")
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("ignores a late detail response after the settings dialog closes", async () => {
    const detailA = deferred<ReturnType<typeof detailResponse>>()

    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url.endsWith("/api/custom-apis/1")) return detailA.promise
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()

    fireEvent.click(screen.getByRole("button", { name: "configure-a" }))
    fireEvent.click(screen.getByRole("button", { name: "close-settings" }))

    await act(async () => {
      detailA.resolve(detailResponse(1, "authoritative-a"))
      await detailA.promise
    })

    expect(screen.getByTestId("custom-api-edit-name")).toHaveTextContent("")
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("hydrates authoritative MCP detail and saves only an unrelated delta", async () => {
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url.endsWith("/api/mcp/servers/3")) {
        if (options?.method === "PUT") {
          return Promise.resolve({ ok: true, json: async () => ({}) })
        }
        return Promise.resolve(mcpDetailResponse(3, "authoritative-mcp"))
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "configure-mcp" }))

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/mcp/servers/3")
      expect(JSON.parse(screen.getByTestId("mcp-edit-state").textContent || "{}")).toMatchObject({
        name: "authoritative-mcp",
        user_env: { EXISTING_TOKEN: "********" },
        can_edit_global: false,
      })
    })

    fireEvent.click(screen.getByRole("button", { name: "change-mcp-description" }))
    saveMcpEditor()

    await waitFor(() => {
      const updateCall = apiRequestMock.mock.calls.find(([, options]) => options?.method === "PUT")
      expect(updateCall?.[0]).toBe("http://api.local/api/mcp/servers/3")
      expect(JSON.parse(updateCall?.[1]?.body as string)).toEqual({
        description: "Updated MCP description",
      })
    })
  })

  it("connects a remote-MCP OAuth catalog app through the oauth/connect endpoint", async () => {
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    try {
      apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          expect(options?.method).toBe("POST")
          expect((options?.headers as Record<string, string>)?.Accept).toBe("application/json")
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize?client_id=dyn-1" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      renderDialog()
      fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))

      await waitFor(() => {
        expect(popup.location.href).toBe("https://auth.granola.ai/authorize?client_id=dyn-1")
      })
      // A features string must be passed: without one, browsers open a full
      // new tab instead of the small centered popup the builtin flow uses.
      expect(openSpy).toHaveBeenCalledWith(
        "about:blank",
        "mcp-oauth",
        expect.stringContaining("width=600,height=700"),
      )
      expect(popup.close).not.toHaveBeenCalled()
      expect(toastErrorMock).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
    }
  })

  it("connects a custom mcp_oauth server through the per-server oauth/connect endpoint", async () => {
    // #1313: this Connect button used to be unreachable — local entries
    // carried no auth_type, so the dispatch fell through to the
    // mis-authored-entry toast. With the hint in place it must also avoid
    // the catalog route: /api/mcp/apps/{id}/oauth/connect resolves catalog
    // app ids only and would 404 on a user-added server.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    try {
      apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/9/oauth/connect") {
          expect(options?.method).toBe("POST")
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.example.com/authorize" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      renderDialog()
      fireEvent.click(screen.getByRole("button", { name: "connect-records" }))

      await waitFor(() => {
        expect(popup.location.href).toBe("https://auth.example.com/authorize")
      })
      expect(popup.close).not.toHaveBeenCalled()
      expect(toastErrorMock).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
    }
  })

  it("rechecks a closed custom mcp_oauth popup against the local listing and auto-selects it", async () => {
    // The popup's opener link is severed, so a closed popup is ambiguous and
    // the handler asks the backend what actually happened. A custom server
    // only exists under location=local — querying the remote branch (as the
    // catalog path does) would report every custom connect as a failure.
    //
    // Passing onConnectSelected also puts the dialog in select mode, which is
    // what makes the connect-records trigger's autoSelect true — so a
    // recheck that confirms the connection must also land "records" in the
    // committed selection (#1332).
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    const onConnectSelected = vi.fn()
    let localListCalls = 0
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/9/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.example.com/authorize" }),
          })
        }
        if (url === "http://api.local/api/mcp/apps?location=local") {
          localListCalls += 1
          return Promise.resolve({
            ok: true,
            json: async () => [{ ...customMcpOauthApp(), is_connected: true }],
          })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
          onConnectSelected={onConnectSelected}
        />,
      )

      // Fake timers only from here on: the poll's setInterval must be created
      // under them to be advanceable, and @testing-library's async queries
      // hang once they are active.
      vi.useFakeTimers()
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-records" }))
      })

      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })

      expect(localListCalls).toBe(1)
      expect(onSuccess).toHaveBeenCalled()

      // Commit the selection to observe it. Fake timers are still active, so
      // this must stay on synchronous queries (getByRole) — findBy*/waitFor
      // would hang.
      act(() => {
        fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
      })
      expect(onConnectSelected).toHaveBeenCalledWith(["records"])
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  // Pins the ordering the F5 fix depends on: a dialog close must beat a
  // still-in-flight connect POST that resolves afterwards. isMountedRef
  // alone cannot see this — it only flips false on unmount, and both real
  // consumers of this dialog keep the component mounted across open/close
  // (only the `open` prop toggles), so isMountedRef.current is still true
  // here. Before the fix, that made the post-await guard pass regardless of
  // the close, registering a poll after clearMcpOauthPollState() had already
  // run — a poll nothing would ever clear, which would go on to recheck
  // location=local and fire onSuccess against a dialog the user already
  // closed.
  it("skips registering a poll when the dialog closes before the connect POST resolves", async () => {
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    let localListCalls = 0
    const connectResponse = deferred<{ ok: boolean; json: () => Promise<{ authorization_url: string }> }>()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/9/oauth/connect") {
          return connectResponse.promise
        }
        if (url === "http://api.local/api/mcp/apps?location=local") {
          localListCalls += 1
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      const { rerender } = render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )

      // Fake timers from here on, same as the recheck test above: the poll's
      // setInterval (if one is wrongly registered) must be created under
      // them to be advanceable, and this test issues no further async
      // queries that would hang once they are active.
      vi.useFakeTimers()

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-records" }))
      })

      // Close the dialog while the connect POST above is still unresolved —
      // only the `open` prop changes, so the component itself stays mounted
      // and isMountedRef never flips. This is what runs
      // clearMcpOauthPollState() and bumps the generation counter, ahead of
      // the POST settling.
      await act(async () => {
        rerender(
          <ConnectMcpDialog
            open={false}
            onOpenChange={vi.fn()}
            selectedMcpServers={selectedMcpServers}
            onSuccess={onSuccess}
          />,
        )
      })

      await act(async () => {
        connectResponse.resolve({
          ok: true,
          json: async () => ({ authorization_url: "https://auth.example.com/authorize" }),
        })
        await connectResponse.promise
      })

      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })

      expect(localListCalls).toBe(0)
      expect(onSuccess).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("fires exactly one oauth/connect POST when a custom mcp_oauth connect is double-clicked in the same tick", async () => {
    // #1330: the DCR case. This server has no configured client_id, so each
    // POST that reaches the backend and finds no MCPOAuthClient row performs a
    // *real* dynamic client registration at the third-party authorization
    // server. The loser is discarded locally by the registration_lookup_hash
    // IntegrityError fallback, but the registration it created at the provider
    // stays there — which is why the second POST has to be stopped in the
    // browser, not reconciled afterwards. disabled={isConnecting} can't do it:
    // it reads loadingApps, React state, so it lags a commit cycle behind two
    // clicks fired back to back before any render flushes.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    let connectCalls = 0
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/9/oauth/connect") {
          connectCalls += 1
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.example.com/authorize" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      renderDialog()
      const connectButton = screen.getByRole("button", { name: "connect-records" })
      // The settings-dialog mock's trigger carries no disabled prop, so both
      // clicks reach the handler — which is the point: this asserts the
      // handler's own guard, the one that exists precisely because the real
      // trigger's disabled prop cannot be relied on to have committed yet.
      fireEvent.click(connectButton)
      fireEvent.click(connectButton)

      await waitFor(() => {
        expect(popup.location.href).toBe("https://auth.example.com/authorize")
      })
      expect(connectCalls).toBe(1)
      // One popup, not two aliases of the same shared window name.
      expect(openSpy).toHaveBeenCalledTimes(1)
    } finally {
      openSpy.mockRestore()
    }
  })

  it("fires exactly one oauth/connect POST when a catalog mcp_oauth connect is double-clicked in the same tick", async () => {
    // Same window as the custom-server case above, through the catalog route.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    let connectCalls = 0
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          connectCalls += 1
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      renderDialog()
      const connectButton = screen.getByRole("button", { name: "connect-granola" })
      fireEvent.click(connectButton)
      fireEvent.click(connectButton)

      await waitFor(() => {
        expect(popup.location.href).toBe("https://auth.granola.ai/authorize")
      })
      expect(connectCalls).toBe(1)
      expect(openSpy).toHaveBeenCalledTimes(1)
    } finally {
      openSpy.mockRestore()
    }
  })

  it("opens one popup and registers one message listener when a builtin_oauth connect is double-clicked in the same tick", async () => {
    // #1330: the builtin flow has no provider-side registration to duplicate,
    // but an unguarded second click opens a second provider popup and adds a
    // second postMessage listener — so one 'oauth-success' message would fire
    // onSuccess and loadApps() twice.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const addListenerSpy = vi.spyOn(window, "addEventListener")
    const onSuccess = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      const connectButton = screen.getByRole("button", { name: "connect-builtin" })
      fireEvent.click(connectButton)
      fireEvent.click(connectButton)

      expect(openSpy).toHaveBeenCalledTimes(1)
      const messageListeners = addListenerSpy.mock.calls.filter(([type]) => type === "message")
      expect(messageListeners).toHaveLength(1)

      await act(async () => {
        window.dispatchEvent(
          Object.assign(new Event("message"), { data: { type: "oauth-success" } }),
        )
      })
      expect(onSuccess).toHaveBeenCalledTimes(1)
    } finally {
      addListenerSpy.mockRestore()
      openSpy.mockRestore()
    }
  })

  it("frees the app's connect slot when the dialog closes mid-OAuth-flow, so it stays connectable", async () => {
    // #1330's trap: a bespoke per-handler in-flight ref would be released only
    // by handleConnectMcpOAuthApp's own exit points, and clearMcpOauthPollState
    // — which drops loadingApps wholesale when the dialog closes — knows
    // nothing about one. Abandoning an OAuth flow by closing the dialog would
    // then leave the app permanently unconnectable until a page reload. The
    // guard shadows loadingApps precisely so every path that clears the
    // spinner also frees the slot.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    let connectCalls = 0
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/9/oauth/connect") {
          connectCalls += 1
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.example.com/authorize" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      const props = {
        onOpenChange: vi.fn(),
        selectedMcpServers,
      }
      const { rerender } = render(<ConnectMcpDialog open {...props} />)
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-records" }))
      })
      expect(connectCalls).toBe(1)

      // The user closes the dialog while the popup is still waiting for
      // consent, then reopens it and tries again.
      rerender(<ConnectMcpDialog open={false} {...props} />)
      rerender(<ConnectMcpDialog open {...props} />)

      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-records" }))
      })
      expect(connectCalls).toBe(2)
    } finally {
      openSpy.mockRestore()
    }
  })

  it("fires exactly one POST when a keyless connect is double-clicked in the same tick", async () => {
    // Round-9: keylessConnectsRef's guard had zero test coverage — removing
    // its .has()/.add() calls would leave every other test green. Without
    // it, disabled={isConnecting} alone doesn't help here: it's React state,
    // so it lags one commit cycle behind two synchronous clicks fired back
    // to back before any render flushes, producing duplicate POSTs, success
    // toasts, and loadApps() refreshes. The ref is synchronous and closes
    // that window outright.
    let connectCalls = 0
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        connectCalls += 1
        return Promise.resolve({ ok: true, json: async () => ({ id: 1 }) })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    const connectButton = screen.getByRole("button", { name: "connect-chrome" })
    // Not wrapped in separate act() calls / awaits — the point is to land
    // both clicks before React (or a prior await) gets a chance to update
    // the disabled state in between.
    fireEvent.click(connectButton)
    fireEvent.click(connectButton)

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledTimes(1)
    })
    expect(connectCalls).toBe(1)
  })

  it("connects a keyless catalog app directly through the connect endpoint", async () => {
    let connectBody: string | undefined
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        expect(options?.method).toBe("POST")
        connectBody = options?.body as string
        return Promise.resolve({ ok: true, json: async () => ({ id: 1 }) })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))

    // No key dialog, no popup — just the POST. is_active is sent explicitly
    // (true) so reconnecting after a dormant association reactivates it
    // instead of silently staying disconnected; there's no env to send.
    await waitFor(() => {
      expect(connectBody).toBe(JSON.stringify({ is_active: true }))
    })
    expect(toastErrorMock).not.toHaveBeenCalled()
    // Pins the connectSuccess key (not the old buttons.save "Save" label)
    // for the shared connectCatalogApp helper. submitKeyConnect uses the
    // exact same call site, so this also pins the api_key path's copy —
    // there is no separate implementation to diverge. translateMock ignores
    // interpolation params and returns the key verbatim, so toast.success
    // receives only that one string argument here — the { name } param
    // itself is exercised for real by the i18n locale files, not this mock.
    expect(toastSuccessMock).toHaveBeenCalledWith("tools.mcp.dialog.connectSuccess")
    // Round-9: the assertion above only pinned the key, leaving the new
    // {name} interpolation itself unexercised (translateMock drops the
    // params argument entirely). Assert the component actually passed it to
    // t(), independent of what the mock does with it.
    expect(translateMock).toHaveBeenCalledWith("tools.mcp.dialog.connectSuccess", { name: "Chrome" })
  })

  it("surfaces the backend error when a keyless connect fails", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        return Promise.resolve({
          ok: false,
          status: 400,
          json: async () => ({ detail: "This app cannot be connected via the connect endpoint" }),
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith(
        "This app cannot be connected via the connect endpoint",
      )
    })
  })

  it("keeps the dialog open while a keyless connect is in flight (Escape and outside click)", async () => {
    const onOpenChange = vi.fn()
    // A deferred connect response: held open until resolveConnect() runs, so
    // the test can assert the mid-flight (guard active) behavior before the
    // request settles.
    let resolveConnect: (() => void) | undefined
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        return new Promise((resolve) => {
          resolveConnect = () => resolve({ ok: true, json: async () => ({ id: 1 }) })
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    render(
      <ConnectMcpDialog open onOpenChange={onOpenChange} selectedMcpServers={selectedMcpServers} />,
    )
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))

    // The connect POST fired (mock captured the resolver) but hasn't
    // resolved yet — this is the in-flight window the guard must cover.
    await waitFor(() => {
      expect(resolveConnect).toBeDefined()
    })

    // Both dismiss paths are refused mid-flight: Escape and outside click
    // (the mock wires the overlay button to the same onOpenChange callback
    // Radix's outside-click dismissal invokes).
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    fireEvent.click(screen.getAllByTestId("dialog-overlay")[0])
    expect(onOpenChange).not.toHaveBeenCalledWith(false)

    // Settle the request and wait on a state-based signal (the success
    // toast fires in the same continuation that clears the guard)...
    resolveConnect?.()
    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalled()
    })

    // ...then a single Escape closes normally again.
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it("does not let one app's finished connect clobber another app's in-flight state", async () => {
    // Regression test: loadingApp is a single shared string across the whole
    // catalog, not per-app. Before the finally-block match-guard, app A
    // finishing while app B was still pending would unconditionally clear
    // loadingApp to null, silently dropping B's spinner/disabled state
    // mid-flight — this drives the real Card -> settings-dialog ->
    // connect path (not the direct-button shortcut the other tests use) so
    // it actually exercises isConnecting the way production does.
    let resolveChrome: (() => void) | undefined
    let resolveKeylessB: (() => void) | undefined
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            { id: "chrome", name: "Chrome", description: "", icon: "", is_connected: false, transport: "stdio", auth_type: "keyless" },
            { id: "keyless-b", name: "Keyless B", description: "", icon: "", is_connected: false, transport: "stdio", auth_type: "keyless" },
          ],
        })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        return new Promise((resolve) => {
          resolveChrome = () => resolve({ ok: true, json: async () => ({ id: 1 }) })
        })
      }
      if (url === "http://api.local/api/mcp/apps/keyless-b/connect") {
        return new Promise((resolve) => {
          resolveKeylessB = () => resolve({ ok: true, json: async () => ({ id: 2 }) })
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Chrome")

    // Select Chrome, start its connect, and confirm the settings dialog
    // reflects it as connecting.
    fireEvent.click(screen.getByText("Chrome"))
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))
    await waitFor(() => expect(resolveChrome).toBeDefined())
    await waitFor(() => {
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("connecting")
    })

    // Close Chrome's settings (its request is still pending) and select
    // Keyless B instead — a real, reachable sequence today, since the
    // settings dialog's close is not guarded against an in-flight connect.
    fireEvent.click(screen.getByRole("button", { name: "close-settings" }))
    fireEvent.click(screen.getByText("Keyless B"))
    expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("idle")

    fireEvent.click(screen.getByRole("button", { name: "connect-keyless-b" }))
    await waitFor(() => expect(resolveKeylessB).toBeDefined())
    await waitFor(() => {
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("connecting")
    })

    // Chrome's request finishes while Keyless B is still pending. Its
    // finally must not clear loadingApp out from under Keyless B.
    resolveChrome?.()
    await waitFor(() => {
      // translateMock ignores interpolation params, so this only pins the
      // key, not which app it was for — the point of this test is the
      // isConnecting assertion right below, not the toast content.
      expect(toastSuccessMock).toHaveBeenCalledWith("tools.mcp.dialog.connectSuccess")
    })
    expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("connecting")

    // Finishing Keyless B's own request clears it normally.
    resolveKeylessB?.()
    await waitFor(() => {
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("idle")
    })
  })

  it("does not let an mcp_oauth popup's timeout-cleanup clobber a different app's in-flight keyless connect", async () => {
    // Round-5 M1: the previous fix (test above) only covered keyless<->keyless.
    // loadingApps is shared across every connect *mechanism*, not just every
    // app, so a cross-mechanism sequence — start an OAuth popup connect,
    // switch apps, start a keyless connect, then let the OAuth popup's own
    // cleanup fire — has to be tested separately.
    //
    // Fake timers are enabled only from the connect-granola click onward
    // (its setInterval poll must be created under them to be advanceable) —
    // not from the start, because @testing-library's async queries
    // (findByText/waitFor) poll via real timers and hang forever once fake
    // timers are active. Everything after that point uses fireEvent + act
    // flushes instead of findBy*/waitFor for the same reason.
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    let resolveChrome: (() => void) | undefined
    // The default tab (activeLocation="remote", no search/category/status)
    // fetches the exact same URL — "?location=remote" — that the mcp_oauth
    // popup-closed handler later uses to check whether the connect actually
    // completed. Distinguish them by call order: 1st = the initial catalog
    // render (needs a Chrome card to click), 2nd+ = the post-popup-close check.
    let remoteListCalls = 0
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        if (url === "http://api.local/api/mcp/apps?location=remote") {
          remoteListCalls += 1
          if (remoteListCalls === 1) {
            return Promise.resolve({
              ok: true,
              json: async () => [
                { id: "granola", name: "Granola", description: "", icon: "", is_connected: false, transport: "streamable_http", auth_type: "mcp_oauth" },
                { id: "chrome", name: "Chrome", description: "", icon: "", is_connected: false, transport: "stdio", auth_type: "keyless" },
              ],
            })
          }
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "granola", name: "Granola", description: "", icon: "", is_connected: false }],
          })
        }
        if (url === "http://api.local/api/mcp/apps/chrome/connect") {
          return new Promise((resolve) => {
            resolveChrome = () => resolve({ ok: true, json: async () => ({ id: 1 }) })
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog open onOpenChange={vi.fn()} selectedMcpServers={selectedMcpServers} />,
      )
      // Real timers here: findByText's polling would hang under fake ones.
      await screen.findByText("Chrome")

      vi.useFakeTimers()

      // Start Granola's OAuth popup connect (loadingApps gains "granola").
      // Its checkPopup setInterval is created here, under fake timers.
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))
      })

      // Switch to Chrome and start its keyless connect — a real, reachable
      // sequence, since closing the settings dialog is not guarded against
      // an in-flight OAuth popup wait (only the whole-catalog dialog's close
      // is guarded, and only for the keyless/key-based POST itself). Select
      // the Chrome card so isConnecting (derived from selectedApp) reflects it.
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "close-settings" }))
        fireEvent.click(screen.getByText("Chrome"))
        fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))
      })
      // resolveChrome is captured synchronously inside apiRequestMock, before
      // connectCatalogApp's first await — no wait needed.
      expect(resolveChrome).toBeDefined()
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("connecting")

      // Granola's popup "times out" (closed by the user without completing
      // auth) and its poll fires, clearing "granola" from loadingApps. Under
      // the pre-fix single-slot state this unconditionally cleared to null,
      // silently dropping Chrome's spinner/disabled state mid-flight.
      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("connecting")

      // Chrome's own request finishing still clears it normally.
      await act(async () => {
        resolveChrome?.()
      })
      expect(screen.getByTestId("settings-is-connecting")).toHaveTextContent("idle")
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("refuses to close via the Custom MCP tab's Cancel button while a catalog connect is in flight", async () => {
    // Round-6 MAJOR-1: the Dialog's own onOpenChange guard only covered
    // Escape/outside-click. Three buttons (Custom API Cancel, Custom MCP
    // Cancel, the select-mode footer Connect button) called the raw
    // onOpenChange(false) prop directly, bypassing it entirely — closing the
    // whole dialog out from under a still-pending connect POST just by
    // switching tabs and hitting Cancel. All three now route through the
    // same requestClose() the Dialog itself uses; this exercises one of them
    // (getAllByRole is needed because the Custom API tab has an identically
    // labeled Cancel button under this file's tab mock, which renders every
    // TabsContent regardless of the active tab).
    const onOpenChange = vi.fn()
    let resolveChrome: (() => void) | undefined
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        return new Promise((resolve) => {
          resolveChrome = () => resolve({ ok: true, json: async () => ({ id: 1 }) })
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    render(
      <ConnectMcpDialog open onOpenChange={onOpenChange} selectedMcpServers={selectedMcpServers} />,
    )
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))
    await waitFor(() => expect(resolveChrome).toBeDefined())

    const cancelButtons = screen.getAllByRole("button", { name: "tools.mcp.buttons.cancel" })
    expect(cancelButtons.length).toBeGreaterThan(0)
    // Round-8 N2: the buttons are disabled while in flight, so a real click
    // doesn't even reach requestClose — but that leaves Escape/outside-click
    // as the only paths that still call it while blocked. Verify both:
    for (const button of cancelButtons) {
      expect(button).toBeDisabled()
    }
    fireEvent.click(cancelButtons[0])
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
    // Round-9 N2: a blocked close attempt (Escape/outside-click/header X all
    // route through the same requestClose) previously no-op'd with zero
    // feedback. It now surfaces a toast instead of appearing frozen.
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    expect(toastWarningMock).toHaveBeenCalledWith("tools.mcp.alerts.closeBlockedWhileConnecting")

    resolveChrome?.()
    await waitFor(() => {
      fireEvent.click(cancelButtons[0])
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it("disables the select-mode footer Connect button while a catalog connect is in flight", async () => {
    // Round-7 MAJOR: this button commits localSelectedServers to the parent
    // as a one-shot snapshot via onConnectSelected, with no later
    // reconciliation. If fired while a connect is still pending, the
    // snapshot can miss the app that connect will add via autoSelect once
    // it succeeds. Disabling the trigger (rather than silently skipping the
    // commit) prevents that snapshot from ever being taken early.
    const onConnectSelected = vi.fn()
    let resolveChrome: (() => void) | undefined
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        return new Promise((resolve) => {
          resolveChrome = () => resolve({ ok: true, json: async () => ({ id: 1 }) })
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={selectedMcpServers}
        onConnectSelected={onConnectSelected}
      />,
    )

    const footerConnect = screen.getByRole("button", { name: "tools.mcp.dialog.connect" })
    expect(footerConnect).not.toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))
    await waitFor(() => expect(resolveChrome).toBeDefined())
    expect(footerConnect).toBeDisabled()

    // Disabled buttons don't fire onClick in jsdom, but assert the intent
    // directly: no premature commit while the connect is still pending.
    fireEvent.click(footerConnect)
    expect(onConnectSelected).not.toHaveBeenCalled()

    resolveChrome?.()
    await waitFor(() => expect(footerConnect).not.toBeDisabled())
  })

  it("bounds the connect POST with an abort signal and surfaces a distinct timeout toast", async () => {
    // Round-7: no prior test asserted the round-6 timeout fix actually wires
    // a signal, or that a TimeoutError produces the dedicated toast instead
    // of the generic saveFailed one.
    let capturedInit: RequestInit | undefined
    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/chrome/connect") {
        capturedInit = init
        return Promise.reject(new DOMException("The operation timed out.", "TimeoutError"))
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "connect-chrome" }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("tools.mcp.alerts.connectTimedOut")
    })
    expect(capturedInit?.signal).toBeInstanceOf(AbortSignal)
  })

  it("times out shareConnector independently, so a hang there cannot wedge the close guard open forever", async () => {
    // Round-7 MAJOR: shareConnector previously had no signal at all. Awaited
    // from inside connectCatalogApp's success branch, ahead of the finally
    // that clears catalogConnectsInFlight, an unbounded hang there would
    // keep the dialog's close guard active permanently -- reopening the
    // exact wedge the connect-POST timeout was added to close, one layer
    // deeper. Drives the real api_key "share with team" path end to end.
    const onOpenChange = vi.fn()
    // The ownership radio (private/team) only renders for a team member.
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    let connectBody: Record<string, unknown> | undefined
    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === "http://api.local/api/mcp/apps/google-maps/connect") {
        connectBody = JSON.parse(init?.body as string)
        return Promise.resolve({ ok: true, json: async () => ({ id: 42 }) })
      }
      if (url === "http://api.local/api/connectors/mcp/42/share") {
        return Promise.reject(new DOMException("The operation timed out.", "TimeoutError"))
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    const { container } = render(
      <ConnectMcpDialog open onOpenChange={onOpenChange} selectedMcpServers={selectedMcpServers} />,
    )
    fireEvent.click(screen.getByRole("button", { name: "connect-apikey" }))

    // Reach the key dialog and choose team sharing before submitting. The
    // shared ownershipRadio JSX is rendered three times (custom API tab,
    // custom MCP tab, and here) because this file's Tabs mock renders every
    // TabsContent regardless of the active tab (same reason the round-6
    // Cancel-button test needs getAllByRole) -- which leaves three elements
    // with the same id="ownership-team", breaking the label-based
    // accessible-name lookup getByRole(..., {name}) relies on. shareChoice
    // is one state shared by all three instances, so which one gets clicked
    // doesn't matter -- select by id directly instead.
    await waitFor(() => {
      expect(container.querySelectorAll("#ownership-team").length).toBeGreaterThan(0)
    })
    fireEvent.click(container.querySelectorAll("#ownership-team")[0])
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))

    // Round-8 N4: the share step's timeout gets its own copy — the connect
    // itself already succeeded, so the generic "connection timed out, try
    // again" toast would prompt retrying an established connection.
    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("tools.mcp.alerts.shareTimedOut")
    })

    // submitKeyConnect's own request-body wiring: env_source travels with
    // the connect POST (env itself is omitted here since keyEnvSource
    // defaults to "own" with an empty required_env value pre-filled).
    expect(connectBody?.env_source).toBe("own")
    expect(connectBody).toHaveProperty("env")

    // connectingKeyApp was reset to null despite shareConnector's failure —
    // only the main dialog's role="dialog" remains (getByRole below would
    // throw on ambiguity if the key dialog were still open).
    expect(screen.getAllByRole("dialog")).toHaveLength(1)

    // The guard must have cleared once shareConnector's own timeout settled
    // the catch block — not stayed wedged open by the unbounded await.
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it("surfaces the backend error and closes the popup when oauth/connect fails", async () => {
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: false,
            status: 400,
            json: async () => ({ detail: "This app is not a remote-OAuth connector" }),
          })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      renderDialog()
      fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))

      await waitFor(() => {
        expect(toastErrorMock).toHaveBeenCalledWith("This app is not a remote-OAuth connector")
      })
      expect(popup.close).toHaveBeenCalled()
      expect(popup.location.href).toBe("")
    } finally {
      openSpy.mockRestore()
    }
  })

  it("fires onSuccess and auto-selects only once the closed popup is confirmed connected", async () => {
    vi.useFakeTimers()
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    const onConnectSelected = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        if (url === "http://api.local/api/mcp/apps?location=remote") {
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "granola", name: "Granola", description: "", icon: "", is_connected: true }],
          })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
          onConnectSelected={onConnectSelected}
        />,
      )
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))
      })

      // The popup closes only after the redirect (simulating the user
      // completing the real authorization in it).
      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })

      expect(onSuccess).toHaveBeenCalled()

      // Commit the selection to observe the auto-select this test is named
      // for: the append is local state, so onConnectSelected only sees it
      // through the footer. Granola is the fixture that makes the assertion
      // discriminating — its id ("granola") and display name ("Granola")
      // differ, so this fails if the append ever switches to app.id, whereas
      // the custom-server equivalent below cannot tell the two apart (its id
      // and name are both "records").
      act(() => {
        fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
      })
      expect(onConnectSelected).toHaveBeenCalledWith(["Granola"])
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("does not fire onSuccess when the popup closes on a cancelled/failed authorization", async () => {
    vi.useFakeTimers()
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        if (url === "http://api.local/api/mcp/apps?location=remote") {
          // The user closed the error popup by hand (or cancelled consent) —
          // the backend never recorded a completed grant, so is_connected
          // stays false even though the association row already exists (M1).
          return Promise.resolve({
            ok: true,
            json: async () => [{ id: "granola", name: "Granola", description: "", icon: "", is_connected: false }],
          })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))
      })

      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(500)
      })

      expect(onSuccess).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("stops the loading spinner without firing success after the 5-minute poll timeout", async () => {
    vi.useFakeTimers()
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))
      })

      const callsBeforeTimeout = apiRequestMock.mock.calls.length

      // Popup never closes (user walked away with it open); the poll must
      // still give up after 5 minutes rather than spinning forever (N3).
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5 * 60 * 1000 + 1000)
      })

      expect(onSuccess).not.toHaveBeenCalled()
      // The timeout path stops polling without ever issuing the
      // is-it-really-connected check the popup-closed path makes.
      expect(apiRequestMock.mock.calls.length).toBe(callsBeforeTimeout)
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("also times out the sibling builtin_oauth poll after 5 minutes (N3)", async () => {
    vi.useFakeTimers()
    const popup = { closed: false, close: vi.fn() }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      fireEvent.click(screen.getByRole("button", { name: "connect-builtin" }))

      // Popup never closes; the builtin_oauth poll previously had no timeout
      // cap at all (N3), unlike the mcp_oauth handler above.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5 * 60 * 1000 + 1000)
      })

      expect(onSuccess).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("clears the in-flight OAuth poll and loading state when the dialog closes, not just unmounts (F6)", async () => {
    vi.useFakeTimers()
    const popup = { closed: false, close: vi.fn(), opener: {}, location: { href: "" } }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const onSuccess = vi.fn()
    try {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === "http://api.local/api/mcp/apps/granola/oauth/connect") {
          return Promise.resolve({
            ok: true,
            json: async () => ({ authorization_url: "https://auth.granola.ai/authorize" }),
          })
        }
        if (url.includes("/api/mcp/apps?")) {
          return Promise.resolve({ ok: true, json: async () => [] })
        }
        throw new Error(`Unexpected request: ${url}`)
      })

      const { rerender } = render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "connect-granola" }))
      })

      const callsBeforeClose = apiRequestMock.mock.calls.length

      // The parent typically keeps ConnectMcpDialog mounted and just flips
      // `open` — only the dialog's own content unmounts, not this component,
      // so the unmount-only cleanup effect alone would miss this (F6).
      rerender(
        <ConnectMcpDialog
          open={false}
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )

      popup.closed = true
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000)
      })

      expect(onSuccess).not.toHaveBeenCalled()
      // The poll was already cleared on close, so popup.closed flipping true
      // afterward must not trigger the is-it-connected follow-up check.
      expect(apiRequestMock.mock.calls.length).toBe(callsBeforeClose)
    } finally {
      openSpy.mockRestore()
      vi.useRealTimers()
    }
  })

  it("removes the builtin_oauth postMessage listener on dialog close, not just its interval (F6)", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      throw new Error(`Unexpected request: ${url}`)
    })
    const onSuccess = vi.fn()
    const openSpy = vi.spyOn(window, "open").mockReturnValue({} as Window)
    try {
      const { rerender } = render(
        <ConnectMcpDialog
          open
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )
      fireEvent.click(screen.getByRole("button", { name: "connect-builtin" }))

      // Close the dialog while the popup (and its postMessage listener) is
      // still outstanding — only clearing the interval (and not this
      // listener too) would let a real success message fired afterward
      // still call onSuccess against a dialog the user already closed.
      rerender(
        <ConnectMcpDialog
          open={false}
          onOpenChange={vi.fn()}
          selectedMcpServers={selectedMcpServers}
          onSuccess={onSuccess}
        />,
      )

      window.dispatchEvent(
        new MessageEvent("message", { data: { type: "oauth-success" } }),
      )

      expect(onSuccess).not.toHaveBeenCalled()
    } finally {
      openSpy.mockRestore()
    }
  })

  it("keeps masked baseline entries in an MCP user-env replacement", async () => {
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url.endsWith("/api/mcp/servers/3")) {
        if (options?.method === "PUT") {
          return Promise.resolve({ ok: true, json: async () => ({}) })
        }
        return Promise.resolve(mcpDetailResponse(3, "authoritative-mcp"))
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    fireEvent.click(screen.getByRole("button", { name: "configure-mcp" }))
    await screen.findByText("authoritative-mcp")

    fireEvent.click(screen.getByRole("button", { name: "add-mcp-env" }))
    saveMcpEditor()

    await waitFor(() => {
      const updateCall = apiRequestMock.mock.calls.find(([, options]) => options?.method === "PUT")
      expect(JSON.parse(updateCall?.[1]?.body as string)).toEqual({
        user_env: {
          EXISTING_TOKEN: "********",
          NEW_TOKEN: "new-secret",
        },
      })
    })
  })

  it("filters by the Operations category sidebar entry (PR review: AWS connector had no way to be found except via All)", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())
    apiRequestMock.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Operations" }))

    await waitFor(() => {
      const url = apiRequestMock.mock.calls.at(-1)?.[0] as string
      expect(new URLSearchParams(url.split("?")[1]).get("category")).toBe("Operations")
    })
  })

  it("cleanly deselects a card whose stored selection is an id, instead of duplicating it", async () => {
    // Regression test for a Major finding on #1280: a consumer (e.g.
    // agent-builder.tsx, which re-seeds selectedMcpServers with a
    // connector's resolved server-row name after a save) can pass this
    // component a selection that names the connector by id
    // ("chrome-devtools") rather than display name ("Chrome"). isSelected
    // matches by id OR name, so the card renders selected either way -- but
    // the toggle previously removed only by name, so clicking a
    // visually-selected id-only entry appended the name instead of
    // removing the id: both ended up present, the card stayed selected,
    // and further clicks just oscillated between duplicated and
    // stuck-selected, never reaching "deselected".
    const onConnectSelected = vi.fn()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: "chrome-devtools",
              name: "Chrome",
              description: "",
              icon: "",
              is_connected: true,
              transport: "stdio",
              auth_type: "keyless",
              can_attach: true,
              can_authorize: false,
            },
          ],
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={["chrome-devtools"]}
        onConnectSelected={onConnectSelected}
      />,
    )
    await screen.findByText("Chrome")

    // The card must render selected from the id-only seed before the click —
    // isSelected matches by id OR name, not just name.
    expect(screen.getByTestId("connector-card-chrome-devtools")).toHaveAttribute(
      "data-selected",
      "true",
    )

    // One click on the visually-selected card must deselect it outright.
    fireEvent.click(screen.getByText("Chrome"))

    expect(screen.getByTestId("connector-card-chrome-devtools")).toHaveAttribute(
      "data-selected",
      "false",
    )

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenCalledWith([])
  })

  // The two populations the split in #1347 exists to separate. Both are
  // custom mcp_oauth connectors the editor holds no MCPOAuthGrant for, so
  // is_connected is false for both and the old predicate could not tell them
  // apart -- it attached both, and offered both an Authorize button.
  //
  // Tokens supplied out of band through set_oauth_token_resolver_hook: no
  // grant row is ever written, there is no interactive consent, and no
  // identity the editor could sign in as. Attachable; nothing to authorize.
  const hookResolvedCustomMcp = () => ({
    ...customMcpOauthApp(),
    name: "Records MCP",
    can_attach: true,
    can_authorize: false,
  })

  // Consent simply never started (create_mcp_server writes the association row
  // long before it). Nothing can supply its tokens in a standalone
  // deployment, so attaching it would fail at run time with "MCP server
  // credentials are unavailable" -- refused here instead, with consent
  // advertised as the one-click recovery.
  const unauthorizedCustomMcp = () => ({ ...customMcpOauthApp(), name: "Records MCP" })

  // Deactivated *after* consent: toggle_mcp_server never revokes the grant, so
  // this still lists as connected (checkmark + Configure) while the runtime's
  // is_active filter drops it — can_attach is the only field that can say so.
  // The grant survives; the auth_type hint does not.
  const deactivatedCustomMcp = () => {
    // No auth_type: the backend emits it only alongside an *active* personal
    // association (_local_mcp_consent_association_ok), so a deactivated entry
    // never carries one however its consent went. Deleted rather than omitted
    // because the base factory is the connected shape, which does carry it.
    const entry: Record<string, unknown> = {
      ...unauthorizedCustomMcp(),
      is_connected: true,
      can_attach: false,
      can_authorize: false,
    }
    delete entry.auth_type
    return entry
  }

  // Deactivated *before* any consent: the same shape with is_connected false
  // too, so the card carries no button at all. The detail modal is reachable
  // but offers no recovery — its Connect has no auth_type to dispatch on.
  const dormantBeforeConsent = () => ({
    ...deactivatedCustomMcp(),
    is_connected: false,
  })

  // A team-owned stdio connector the viewer holds no personal association
  // for. The listing reports it connected -- the connected-state predicate
  // returns True unconditionally for every non-mcp_oauth shape -- while the
  // edit route answers 404 without a personal association row. That
  // combination is the *narrowing* direction of the disagreement between the
  // connected gate and can_configure; the widening direction (is_connected:
  // false + can_configure: true) is the population this change exists for,
  // covered by the hook-resolved fixtures above.
  const teamOwnedStdio = () => ({
    id: "team-files",
    name: "Team Files",
    description: "",
    icon: "",
    transport: "stdio",
    is_custom: true,
    is_local: true,
    server_id: 11,
    is_connected: true,
    can_attach: true,
    can_authorize: false,
    can_configure: false,
  })

  // Same disagreement on the Custom API half, where the listing reports
  // every entry connected unconditionally.
  const teamOwnedCustomApi = () => ({
    ...teamOwnedStdio(),
    id: "team-billing",
    name: "Team Billing",
    transport: "custom_api",
    server_id: 12,
  })

  // Shared by renderSelectModeWith below and the non-select-mode card-click
  // test, which otherwise duplicated this exact mockImplementation block.
  function mockAppsList(apps: object[]) {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => apps })
      }
      throw new Error(`Unexpected request: ${url}`)
    })
  }

  function renderSelectModeWith(apps: object[], onConnectSelected: () => void) {
    mockAppsList(apps)
    return render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={selectedMcpServers}
        onConnectSelected={onConnectSelected}
      />,
    )
  }

  it("selects and deselects a hook-resolved custom mcp_oauth connector (#1332)", async () => {
    const onConnectSelected = vi.fn()
    renderSelectModeWith([hookResolvedCustomMcp()], onConnectSelected)
    await screen.findByText("Records MCP")

    fireEvent.click(screen.getByText("Records MCP"))
    // The click must toggle the selection, not divert to the detail modal.
    expect(screen.getByTestId("settings-open-app").textContent).toBe("")
    // And no Authorize trigger: there is no consent flow behind it, so the
    // label would assert a step this connector does not have (#1347).
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()
    // Configure does show: the owner's personal association makes the edit
    // route resolve even though the connector is not "connected" by this
    // field's usual meaning -- the bug this change exists to fix.
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    // Becoming attachable must not change connected-state display: this
    // connector is still unconnected (is_connected: false), so it must stay
    // unchecked. Scoped through the card: connected-check is non-unique
    // across the grid by construction, so an ungrounded query would throw as
    // soon as a second connected app rendered.
    expect(
      within(screen.getByTestId("connector-card-records")).queryByTestId("connected-check"),
    ).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["Records MCP"])

    // Removal has to work too, or an accidental attach would be unrevertable.
    fireEvent.click(screen.getByText("Records MCP"))
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("opens the detail modal without toggling the selection when Configure is clicked in select mode", async () => {
    // Configure's onClick carries the same stopPropagation() as Authorize's:
    // without it, a click on the button would bubble to the card and toggle
    // this attachable entry's selection as a side effect.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([hookResolvedCustomMcp()], onConnectSelected)
    await screen.findByText("Records MCP")

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.configure" }))
    expect(screen.getByTestId("connector-card-records")).toHaveAttribute("data-selected", "false")
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("shows Configure and the connected checkmark, not Authorize, for a connected custom mcp_oauth entry (#1332)", async () => {
    // A connected entry shows Configure plus the checkmark, never Authorize --
    // including when can_authorize is true for it (re-consent is legitimate;
    // the Authorize block's own !isGloballyConnected term suppresses the
    // trigger on connected state, which is why the backend does not).
    const onConnectSelected = vi.fn()
    renderSelectModeWith(
      [{ ...unauthorizedCustomMcp(), is_connected: true, can_attach: true }],
      onConnectSelected,
    )
    await screen.findByText("Records MCP")

    // getBy* throws when absent, so the bare calls are the assertions.
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()
    // Scoped through the card: connected-check is non-unique across the grid
    // by construction, so an ungrounded query would throw as soon as a
    // second connected app rendered.
    within(screen.getByTestId("connector-card-records")).getByTestId("connected-check")
  })

  it("shows the team-ownership badge for a hook-resolved connector that lists as unconnected (#1623)", async () => {
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [hookResolvedCustomMcp()] })
      }
      if (url.endsWith("/api/connectors/status")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ "mcp:9": { shared: true, is_owner: false, needs_config: false } }),
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Records MCP")

    await waitFor(() => {
      within(screen.getByTestId("connector-card-records")).getByText(
        "tools.mcp.sharing.teamTool",
      )
    })
  })

  it("shows the private badge for an answered shared:false status (#1623)", async () => {
    // The only positive pin on this gate's Private arm: every other fixture in
    // this file answers shared: true or a malformed shape, so a gate that
    // regressed to truthiness (rendering nothing for shared: false) would pass
    // all of them.
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [hookResolvedCustomMcp()] })
      }
      if (url.endsWith("/api/connectors/status")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ "mcp:9": { shared: false, is_owner: true, needs_config: false } }),
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Records MCP")

    await waitFor(() => {
      within(screen.getByTestId("connector-card-records")).getByText(
        "tools.mcp.sharing.private",
      )
    })
  })

  it("withholds the ownership badge when the sharing route does not answer for an entry (#1623)", async () => {
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [hookResolvedCustomMcp()] })
      }
      if (url.endsWith("/api/connectors/status")) {
        return Promise.resolve({ ok: false, status: 500, json: async () => ({}) })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Records MCP")

    // The status request must actually have been issued -- a pure negative
    // here would also pass if the fixture lost its server_id or the route
    // was never called. This is the only case in this test, so a match here
    // can only come from this test's own mock.
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/connectors/status",
        expect.objectContaining({ method: "POST" }),
      )
    })

    const card = within(screen.getByTestId("connector-card-records"))
    expect(card.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
  })

  it("withholds the ownership badge when the sharing route answers with a malformed entry (#1623)", async () => {
    // Distinct from the non-ok case above, and the one that actually depends
    // on the merge going through sanitizeConnectorStatus rather than a raw
    // cast: needs_config is not a boolean, so the whole entry must be
    // dropped, not just the one field.
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [hookResolvedCustomMcp()] })
      }
      if (url.endsWith("/api/connectors/status")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ "mcp:9": { shared: true, is_owner: false, needs_config: "no" } }),
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Records MCP")

    // Own test, own mock, own call history (this file's beforeEach resets
    // apiRequestMock before every test) -- so this can only be satisfied by
    // the request this test's own render issued.
    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/connectors/status",
        expect.objectContaining({ method: "POST" }),
      )
    })

    const card = within(screen.getByTestId("connector-card-records"))
    expect(card.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
  })

  it("shows no ownership badge for a listing entry carrying shared but no connector id (#1623)", async () => {
    useAuthMock.mockReturnValue({ token: "token", inTeam: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{ ...mcpOauthApp(), shared: true, is_owner: false }],
        })
      }
      if (url.endsWith("/api/connectors/status")) {
        return Promise.resolve({ ok: true, json: async () => ({}) })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    renderDialog()
    await screen.findByText("Granola")

    const card = within(screen.getByTestId("connector-card-granola"))
    expect(card.queryByText("tools.mcp.sharing.private")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.shared")).toBeNull()
    expect(card.queryByText("tools.mcp.sharing.teamTool")).toBeNull()
  })

  it("offers Authorize, and refuses selection, for a connector whose consent was never started (#1323)", async () => {
    // The fail-early half of #1347: attaching this connector would load zero
    // tools, so the card is not selectable -- but the flow repaired in #1323
    // has to stay one click away, or the refusal would be a dead end.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([unauthorizedCustomMcp()], onConnectSelected)
    await screen.findByText("Records MCP")

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.authorize" }))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")

    // Opening the modal must not have counted as a selection toggle.
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])

    // The card click itself must not attach it either. Scoped through the
    // card: the stub settings dialog now also renders this name.
    fireEvent.click(
      within(screen.getByTestId("connector-card-records")).getByText("Records MCP"),
    )
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("offers Authorize alongside Configure for a never-authorized connector its owner can still edit", async () => {
    // The two fields answer different questions and neither implies the
    // other: this connector needs consent (can_authorize) and its owner can
    // also fix its configuration (can_configure) -- e.g. a wrong URL entered
    // before consent was ever attempted. A ternary could only ever show one.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([unauthorizedCustomMcp()], onConnectSelected)
    await screen.findByText("Records MCP")

    // getBy* throws when absent, so the bare calls are the assertions.
    screen.getByRole("button", { name: "tools.mcp.dialog.authorize" })
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
  })

  // #1390: the create path into the selection, which ignored can_attach
  // entirely while the click path above enforces it. Saves the create form of
  // whichever connector type is under test and answers the post-create lookup
  // with `localListing` — the filtered listing loadApps() sends (location=
  // remote, the sidebar default still in force when the save starts)
  // deliberately answers empty, which is exactly why the lookup cannot
  // reuse it.
  //
  // `pendingLookup`, when given, holds the lookup open so a caller can assert
  // on the window in which the save has succeeded but the decision has not
  // landed yet; its `resolve` finishes the create.
  async function createInSelectMode(
    localListing: object[] | null,
    onConnectSelected: () => void,
    options: { customApi?: boolean; pendingLookup?: Deferred<unknown> } = {},
  ) {
    const listingResponse = localListing
      ? { ok: true, json: async () => localListing }
      // A null listing models the lookup itself failing, the case that has to
      // read as "not attachable" rather than falling through.
      : { ok: false, status: 500, json: async () => ({}) }
    const createUrl = options.customApi
      ? "http://api.local/api/custom-apis"
      : "http://api.local/api/mcp/servers"

    apiRequestMock.mockImplementation((url: string, init?: { method?: string }) => {
      // Only the lookup carries an abort signal; the sidebar's own refresh of
      // the same URL does not. Matching on it keeps this stub — and the wait
      // below — pinned to the request under test.
      if (url === "http://api.local/api/mcp/apps?location=local" && init && "signal" in init) {
        return options.pendingLookup
          ? options.pendingLookup.promise.then(() => listingResponse)
          : Promise.resolve(listingResponse)
      }
      // The sidebar's own refresh of that same URL, fired by the switch to
      // the local tab. Serves the listing so the cards render, but cannot
      // stand in for the lookup: every other filter combination this dialog
      // sends — location=remote before the switch, plus any search/category
      // narrowing — answers empty, which is the whole reason the lookup
      // cannot reuse it.
      if (url === "http://api.local/api/mcp/apps?location=local") {
        return Promise.resolve({ ok: true, json: async () => localListing ?? [] })
      }
      if (url.includes("/api/mcp/apps?")) {
        return Promise.resolve({ ok: true, json: async () => [] })
      }
      if (url === createUrl && init?.method === "POST") {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ id: 9 }) })
      }
      throw new Error(`Unexpected request: ${url}`)
    })

    render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={selectedMcpServers}
        onConnectSelected={onConnectSelected}
      />,
    )

    fireEvent.click(screen.getByRole("button", {
      name: options.customApi ? "name-new-custom-api" : "name-new-mcp",
    }))
    await act(async () => {
      options.customApi ? saveCustomApiEditor() : saveMcpEditor()
    })
    // The create POST must have gone to the endpoint for this connector type,
    // carrying the name the lookup then matches the listing on.
    await waitFor(() =>
      expect(apiRequestMock).toHaveBeenCalledWith(
        createUrl,
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining('"name":"records-mcp"'),
        }),
      ),
    )
    await waitFor(() =>
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/apps?location=local",
        expect.objectContaining({ signal: expect.anything() }),
      ),
    )
    // Let the lookup's own response settle, so the selection state the
    // callers assert on has committed either way.
    if (!options.pendingLookup) await act(async () => {})
  }

  it("does not auto-select a just-created connector the listing reports unattachable (#1390)", async () => {
    // create_mcp_server writes the association long before any grant exists,
    // so a custom mcp_oauth server lists with can_attach false the moment it
    // is created. Selecting it here would commit a connector whose run fails
    // with "MCP server credentials are unavailable" — the failure can_attach
    // exists to prevent, reached through create instead of a card click.
    const onConnectSelected = vi.fn()
    await createInSelectMode(
      [{ ...unauthorizedCustomMcp(), id: "records-mcp", name: "records-mcp" }],
      onConnectSelected,
    )

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])

    // Created-but-unselected, not a dead end: the entry carries
    // can_authorize, so the local tab this switched to offers the recovery.
    await screen.findByRole("button", { name: "tools.mcp.dialog.authorize" })
  })

  it("still auto-selects a just-created connector the listing reports attachable (#1390)", async () => {
    // The other half of the gate: every non-mcp_oauth shape carries its
    // credentials on the server row, lists attachable immediately, and must
    // keep landing in the selection without a second click.
    const onConnectSelected = vi.fn()
    await createInSelectMode([mcpApp(9, "records-mcp")], onConnectSelected)

    await screen.findByText("tools.mcp.dialog.selected")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["records-mcp"])
  })

  it("reads the created connector's own row when a Custom API shares its name (#1390)", async () => {
    // Name is unique per table, not across the listing: an MCP server and a
    // Custom API may both be called records-mcp. Matching on name alone would
    // let the wrong row answer the attachability question — here the Custom
    // API, listed first and attachable, would mask the mcp_oauth server this
    // save actually created. The transport discriminator is what keeps the
    // two apart.
    const onConnectSelected = vi.fn()
    await createInSelectMode(
      [
        customApiApp(4, "records-mcp"),
        { ...unauthorizedCustomMcp(), id: "records-mcp", name: "records-mcp" },
      ],
      onConnectSelected,
    )

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("reads the created Custom API's own row when an MCP server shares its name (#1390)", async () => {
    // The same collision in the direction the backend actually produces:
    // list_mcp_apps appends every MCP server before the first Custom API, so
    // a name-only match always reaches the MCP row first. Creating the Custom
    // API must still be decided by the Custom API's own can_attach, not by
    // the unattachable mcp_oauth server that happens to share its name.
    const onConnectSelected = vi.fn()
    await createInSelectMode(
      [
        { ...unauthorizedCustomMcp(), id: "records-mcp", name: "records-mcp" },
        customApiApp(4, "records-mcp"),
      ],
      onConnectSelected,
      { customApi: true },
    )

    await screen.findByText("tools.mcp.dialog.selected")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["records-mcp"])
  })

  it("refuses to commit the selection while the attachability lookup is pending (#1390)", async () => {
    // A successful save returns to the library tab immediately, so the footer
    // is on screen while the lookup that may still add the new connector is
    // in flight. Committing there would snapshot the selection one entry
    // short — the same staleness the catalog-connect guard exists for.
    const pendingLookup = deferred<unknown>()
    const onConnectSelected = vi.fn()
    await createInSelectMode([mcpApp(9, "records-mcp")], onConnectSelected, { pendingLookup })

    expect(screen.getByRole("button", { name: "tools.mcp.dialog.connect" })).toBeDisabled()

    await act(async () => {
      pendingLookup.resolve(undefined)
      await pendingLookup.promise
    })

    await screen.findByText("tools.mcp.dialog.selected")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["records-mcp"])
  })

  it("leaves a just-created connector unselected when the lookup fails (#1390)", async () => {
    // No answer is not a yes: the click path needs an affirmative can_attach
    // too, and the local tab this switches to keeps the card one click away.
    const onConnectSelected = vi.fn()
    await createInSelectMode(null, onConnectSelected)

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("attaches a team-owned mcp_oauth connector without advertising a flow that would 404", async () => {
    // The #1338 population, and the case that motivated splitting the two
    // fields: the member holds no personal association, so
    // /{server_id}/oauth/connect would 404 -- but the team overlay listed the
    // connector precisely so members could attach it. Withholding auth_type
    // was the only way to say "not authorizable", and it said "not
    // attachable" too, which is why this entry carries no auth_type at all.
    const onConnectSelected = vi.fn()
    const teamOwned: Record<string, unknown> = {
      ...hookResolvedCustomMcp(),
      name: "Team Records",
      id: "team-records",
      // The base factory is the owner's shape, which is configurable; a member
      // reaching this connector through the team overlay holds no association
      // of their own, so the listing reports false here.
      can_configure: false,
    }
    delete teamOwned.auth_type
    renderSelectModeWith([teamOwned], onConnectSelected)
    await screen.findByText("Team Records")

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()
    fireEvent.click(screen.getByText("Team Records"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["Team Records"])
  })

  it("offers no Configure for a team-owned connector the viewer holds no association for", async () => {
    // The listing reports this connector connected (every non-mcp_oauth
    // shape does), but the viewer has no row of their own, so the edit
    // route would 404 -- the collapsing direction of the disagreement
    // between the connected gate and can_configure.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([teamOwnedStdio()], onConnectSelected)
    await screen.findByText("Team Files")

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
    // The collapse must not have cost the team member their ability to
    // select this connector -- can_attach is unaffected.
    fireEvent.click(screen.getByText("Team Files"))
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["Team Files"])
  })

  it("offers no Configure for a team-owned custom API the viewer holds no association for", async () => {
    const onConnectSelected = vi.fn()
    renderSelectModeWith([teamOwnedCustomApi()], onConnectSelected)
    await screen.findByText("Team Billing")

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
    fireEvent.click(screen.getByText("Team Billing"))
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith(["Team Billing"])
  })

  it("falls back to the connected gate for an entry that carries no can_configure field", async () => {
    // teamOwnedStdio() is the shape where the connected gate and
    // can_configure actively disagree (is_connected: true, can_configure:
    // false) -- deleting the field here proves the fallback reads the
    // connected gate rather than happening to agree with it by coincidence.
    const connectedNoField: Record<string, unknown> = teamOwnedStdio()
    delete connectedNoField.can_configure
    const onConnectSelected = vi.fn()
    renderSelectModeWith([connectedNoField], onConnectSelected)
    await screen.findByText("Team Files")
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    cleanup()

    const disconnectedNoField: Record<string, unknown> = {
      ...connectedNoField,
      is_connected: false,
    }
    renderSelectModeWith([disconnectedNoField], onConnectSelected)
    await screen.findByText("Team Files")
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
  })

  it("still gates selection on connected state for unconnected catalog apps", async () => {
    // mcpOauthApp() (Granola) is the load-bearing fixture: it carries the
    // exact same auth_type as the custom mcp_oauth population, but it is a
    // *catalog* entry, so its is_connected: false means the user has no
    // association row for it at all — connecting really is a prerequisite.
    // This used to be decided here by is_custom being absent from catalog
    // entries, an emission rule the backend never stated; can_attach: false
    // now says it outright (#1347). keylessApp (Chrome) is included as a
    // non-mcp_oauth catalog control. Both must keep routing the card click to
    // the detail modal instead of attaching a connector that does not exist
    // for them.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([mcpOauthApp(), keylessApp()], onConnectSelected)
    await screen.findByText("Chrome")

    fireEvent.click(screen.getByText("Granola"))
    // Assert right after each click: a later click would overwrite this.
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Granola")

    fireEvent.click(screen.getByText("Chrome"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Chrome")

    // No card-level Authorize trigger for either: a catalog entry's Connect
    // is dispatched on auth_type from the detail modal, so can_authorize is
    // false for the whole catalog branch.
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("offers no Configure for an unconnected catalog entry", async () => {
    // Guards against the dead-button regression an unconditional True would
    // reintroduce: Granola's Configure equivalent (manage-my-key / re-run
    // OAuth) does not exist until it is connected, so nothing here must show.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([mcpOauthApp()], onConnectSelected)
    await screen.findByText("Granola")

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.configure" })).toBeNull()
    // Connect stays reachable through the card click.
    fireEvent.click(screen.getByText("Granola"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Granola")
  })

  it("refuses a deactivated association even when it lists as connected", async () => {
    // toggle_mcp_server flips is_active and never revokes a completed grant,
    // so a connector deactivated *after* consent still lists as
    // is_connected: true — indistinguishable from an ordinary connected entry
    // in the one field the old predicate's first disjunct read, and attachable
    // through it. (auth_type is withheld here, but the old predicate never
    // reached its second disjunct for a connected entry.) The runtime's server query drops inactive
    // associations outright, so attaching it would silently load zero tools.
    // can_attach is the only field that can carry this (#1347), and
    // can_authorize is false alongside it because the per-server OAuth
    // endpoints require an active association: this connector needs
    // re-enabling, not re-authorization.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([deactivatedCustomMcp()], onConnectSelected)
    await screen.findByText("Records MCP")

    // Connected, so the footer shows Configure; the card click must still
    // not toggle a selection.
    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("deselects an entry that became unattachable while already selected", async () => {
    // can_attach can go false under a selection that already exists: a
    // connector disabled on the Tools page after being attached, or one
    // attached before the gate existed. The card still renders selected
    // (isSelected reads the selection, not the gate), so refusing the click
    // would leave it visibly selected and removable only from the builder's
    // own list — the un-removable state #1280 was filed for. Attaching stays
    // gated; removing does not.
    const onConnectSelected = vi.fn()
    mockAppsList([deactivatedCustomMcp()])
    render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={["Records MCP"]}
        onConnectSelected={onConnectSelected}
      />,
    )
    await screen.findByText("Records MCP")
    expect(screen.getByTestId("connector-card-records")).toHaveAttribute("data-selected", "true")

    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("connector-card-records")).toHaveAttribute("data-selected", "false")
    // Removing must not have opened the modal instead.
    expect(screen.getByTestId("settings-open-app").textContent).toBe("")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])

    // And it must not be re-attachable: the gate still refuses the second click.
    fireEvent.click(
      within(screen.getByTestId("connector-card-records")).getByText("Records MCP"),
    )
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("routes a deactivated-before-consent entry to its edit form and the modal", async () => {
    // No checkmark, no Authorize -- but its owner's personal association
    // still exists (only its is_active flag is false, which the edit routes
    // do not filter on), so Configure now opens. This does not re-enable the
    // connector: the edit form's save payload carries no is_active key, so
    // there is still no way to flip it back on from here. That gap is
    // unrelated to this field and tracked on its own.
    const onConnectSelected = vi.fn()
    renderSelectModeWith([dormantBeforeConsent()], onConnectSelected)
    await screen.findByText("Records MCP")

    expect(
      within(screen.getByTestId("connector-card-records")).queryByTestId("connected-check"),
    ).toBeNull()
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()

    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("deselects a deactivated-before-consent entry on the first click, then opens the modal on the second", async () => {
    // This shape now carries a Configure button, but it renders on the card
    // footer only, never on the card body the click below targets -- so the
    // removal still has to read from the card itself: the ring drops and the
    // footer count decrements. The second click, with nothing left to
    // remove, falls through to the modal like any other unattachable card.
    const onConnectSelected = vi.fn()
    mockAppsList([dormantBeforeConsent()])
    render(
      <ConnectMcpDialog
        open
        onOpenChange={vi.fn()}
        selectedMcpServers={["Records MCP"]}
        onConnectSelected={onConnectSelected}
      />,
    )
    await screen.findByText("Records MCP")
    expect(screen.getByTestId("connector-card-records")).toHaveAttribute("data-selected", "true")

    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("connector-card-records")).toHaveAttribute("data-selected", "false")
    expect(screen.getByTestId("settings-open-app").textContent).toBe("")

    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")
    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.connect" }))
    expect(onConnectSelected).toHaveBeenLastCalledWith([])
  })

  it("opens the detail modal on card click outside select mode", async () => {
    // The Tools page has no selection to toggle, so the card click keeps its
    // original destination for connected and unconnected entries alike — and
    // the Authorize trigger, which exists to explain a diverted click, stays
    // scoped to select mode even for an authorizable entry.
    mockAppsList([unauthorizedCustomMcp()])
    render(<ConnectMcpDialog open onOpenChange={vi.fn()} selectedMcpServers={selectedMcpServers} />)
    await screen.findByText("Records MCP")

    expect(screen.queryByRole("button", { name: "tools.mcp.dialog.authorize" })).toBeNull()
    // Configure is not scoped to select mode -- unlike Authorize, it does not
    // exist to explain a diverted click, so it renders here too.
    screen.getByRole("button", { name: "tools.mcp.dialog.configure" })
    fireEvent.click(screen.getByText("Records MCP"))
    expect(screen.getByTestId("settings-open-app").textContent).toBe("Records MCP")
  })
})
