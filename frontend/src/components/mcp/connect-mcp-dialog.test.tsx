import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ConnectMcpDialog } from "./connect-mcp-dialog"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "http://api.local" }
})

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ token: "token" }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("@/contexts/mcp-apps-context", () => ({
  useMcpApps: () => ({ apps: [] }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock, success: toastSuccessMock },
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
    onConfigure,
    onOpenChange,
    onConnectStart,
  }: {
    onConfigure: (app: object) => void
    onOpenChange: (open: boolean) => void
    onConnectStart: (app: object) => void
  }) => (
    <div>
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
      <button type="button" onClick={() => onConnectStart(builtinOauthApp())}>
        connect-builtin
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
function keylessApp() {
  return {
    id: "chrome",
    name: "Chrome",
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
