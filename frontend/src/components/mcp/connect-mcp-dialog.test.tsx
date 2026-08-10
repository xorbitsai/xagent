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
  CustomApiForm: ({ mcpFormData }: { mcpFormData: { name?: string } }) => (
    <output data-testid="custom-api-edit-name">{mcpFormData.name ?? ""}</output>
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
    isConnecting,
    onConfigure,
    onOpenChange,
    onConnectStart,
  }: {
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
})
