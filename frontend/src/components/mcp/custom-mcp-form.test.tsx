import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CustomMcpForm, isHttpMcpOAuthTransport } from "./custom-mcp-form"
import { MCPServerFormData } from "./custom-api-form"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => vi.fn((key: string) => key))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
  }
})

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: toastSuccessMock,
  },
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: translateMock,
  }),
}))

function okJson(data: unknown): Response {
  return {
    ok: true,
    json: vi.fn().mockResolvedValue(data),
  } as unknown as Response
}

function deferredResponse() {
  let resolve!: (response: Response) => void
  const promise = new Promise<Response>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

async function flushPromises() {
  await Promise.resolve()
  await Promise.resolve()
}

function renderMcpOAuthForm(overrides: Partial<MCPServerFormData> = {}) {
  const formData: MCPServerFormData = {
    name: "records",
    transport: "streamable_http",
    description: "",
    config: {
      url: "https://mcp.example.com/mcp",
      auth: {
        type: "mcp_oauth",
        resource: "https://mcp.example.com/mcp",
        issuer: "https://auth.example.com",
        scope: "records.read",
        client_id: "client-123",
      },
    },
    ...overrides,
  }

  return render(
    <CustomMcpForm
      mcpFormData={formData}
      setMcpFormData={vi.fn()}
      serverId={42}
    />
  )
}

describe("CustomMcpForm MCP OAuth", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    toastSuccessMock.mockReset()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    vi.restoreAllMocks()
    cleanup()
  })

  it("starts connect through the JSON authorization URL response", async () => {
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    const openMock = vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })

    expect(apiRequestMock).toHaveBeenCalledWith(
      "http://api.local/api/mcp/42/oauth/connect",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        headers: expect.objectContaining({
          Accept: "application/json",
          "Content-Type": "application/json",
        }),
      })
    )

    const connectCall = apiRequestMock.mock.calls.find(([url]) =>
      String(url).endsWith("/oauth/connect")
    )
    expect(connectCall).toBeTruthy()
    const [, connectOptions] = connectCall as [string, RequestInit]
    const body = JSON.parse(String(connectOptions.body))
    expect(body).toHaveProperty("redirect_after")
    expect(body).not.toHaveProperty("resource")
    expect(body).not.toHaveProperty("issuer")
    expect(body).not.toHaveProperty("scope")
    expect(body).not.toHaveProperty("resource_metadata_url")
    expect(body).not.toHaveProperty("access_token")
    expect(body).not.toHaveProperty("refresh_token")
    expect(body).not.toHaveProperty("resource_owner_key")
    expect(openMock).toHaveBeenCalledWith("about:blank", "_blank")
    expect(popup.opener).toBeNull()
    expect(popup.location.href).toBe("https://auth.example.com/authorize")
  })

  it("fires exactly one oauth/connect POST when Connect is double-clicked in the same tick", async () => {
    // #1330: this button drives the same per-server DCR endpoint the connector
    // dialog does, and its only guard is disabled={oauthAction === "connect"} —
    // React state, so it lags a commit cycle behind two clicks landing in the
    // same tick. The fixture drops client_id to model the DCR case: with no
    // configured client, each POST that reaches the backend registers a new
    // client at the third-party authorization server, and the one the local
    // IntegrityError fallback discards stays registered over there.
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    let connectCalls = 0
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        connectCalls += 1
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm({
      config: {
        url: "https://mcp.example.com/mcp",
        auth: {
          type: "mcp_oauth",
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scope: "records.read",
        },
      },
    })

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    const connectButton = screen.getByText("tools.mcp.dialog.oauthConnect")
    // Both clicks are dispatched inside one act() scope so React defers the
    // re-render until it exits — the second click therefore lands on a button
    // whose disabled prop has not been committed yet, which is exactly the
    // window scripted clicks or a client retry hit. Two separate fireEvent
    // calls would not reproduce it: React flushes discrete-event updates
    // synchronously, so the second click would find the button already
    // disabled and this test would pass with or without the guard.
    await act(async () => {
      fireEvent.click(connectButton)
      fireEvent.click(connectButton)
    })
    await act(async () => {
      await flushPromises()
    })

    expect(connectCalls).toBe(1)
    expect(popup.location.href).toBe("https://auth.example.com/authorize")
  })

  it("keeps the connect guard closed when an overlapping discover finishes first", async () => {
    // Review follow-up on #1330: the three OAuth buttons are each gated on
    // their own action identity, so a discover and a connect are allowed to
    // overlap. If completion cleared the shared slot unconditionally,
    // whichever finished first would reopen the other's guard mid-flight —
    // here, discover's completion would re-admit a second connect POST and
    // with it a second Dynamic Client Registration at the provider. Only the
    // action that owns the slot may release it.
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    const discoverResponse = deferredResponse()
    // The connect POST is left pending for the whole test: the guard is only
    // ever meant to hold while its own action is genuinely in flight, so a
    // connect that resolved would reopen it legitimately and prove nothing.
    const connectResponse = deferredResponse()
    let connectCalls = 0
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/discover") {
        return discoverResponse.promise
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        connectCalls += 1
        return connectResponse.promise
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    // Discover starts and stays in flight, then a connect overlaps it.
    await act(async () => {
      fireEvent.click(screen.getByText("tools.mcp.dialog.oauthDiscover"))
    })
    const connectButton = screen.getByText("tools.mcp.dialog.oauthConnect")
    await act(async () => {
      fireEvent.click(connectButton)
    })
    expect(connectCalls).toBe(1)

    // Discover now completes. It must not release the connect guard.
    await act(async () => {
      discoverResponse.resolve(
        okJson({
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scopes: ["records.read"],
        })
      )
      await flushPromises()
    })

    await act(async () => {
      fireEvent.click(connectButton)
      fireEvent.click(connectButton)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(connectCalls).toBe(1)
  })

  it("treats websocket as an MCP OAuth-capable transport", () => {
    expect(isHttpMcpOAuthTransport("streamable_http")).toBe(true)
    expect(isHttpMcpOAuthTransport("sse")).toBe(true)
    expect(isHttpMcpOAuthTransport("websocket")).toBe(true)
    expect(isHttpMcpOAuthTransport("stdio")).toBe(false)
  })

  it("does not start OAuth connect when the authorization popup is blocked", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })
    apiRequestMock.mockClear()

    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))

    expect(toastErrorMock).toHaveBeenCalledWith("tools.mcp.dialog.oauthConnectFailed")
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it("polls OAuth status until the authorization popup is closed", async () => {
    const onOAuthStatusChange = vi.fn()
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const formData: MCPServerFormData = {
      name: "records",
      transport: "streamable_http",
      description: "",
      config: {
        url: "https://mcp.example.com/mcp",
        auth: {
          type: "mcp_oauth",
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scope: "records.read",
          client_id: "client-123",
        },
      },
    }

    render(
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={vi.fn()}
        serverId={42}
        onOAuthStatusChange={onOAuthStatusChange}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })
    expect(popup.location.href).toBe("https://auth.example.com/authorize")

    const statusCallsBeforePolling = apiRequestMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/oauth/status")
    ).length

    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/oauth/status"))
        .length
    ).toBe(statusCallsBeforePolling + 1)
    expect(onOAuthStatusChange).not.toHaveBeenCalled()

    popup.closed = true
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(onOAuthStatusChange).toHaveBeenCalledTimes(1)
  })

  it("clears OAuth polling when the form unmounts", async () => {
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return Promise.resolve(okJson({ server_id: 42, grants: [] }))
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { unmount } = renderMcpOAuthForm()

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        "http://api.local/api/mcp/42/oauth/status"
      )
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })

    const statusCallsBeforeUnmount = apiRequestMock.mock.calls.filter(([url]) =>
      String(url).endsWith("/oauth/status")
    ).length

    unmount()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect(
      apiRequestMock.mock.calls.filter(([url]) => String(url).endsWith("/oauth/status"))
        .length
    ).toBe(statusCallsBeforeUnmount)
  })

  it("ignores pending OAuth status responses after unmount", async () => {
    const pendingStatus = deferredResponse()
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        return pendingStatus.promise
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const { unmount } = renderMcpOAuthForm()
    unmount()

    await act(async () => {
      pendingStatus.resolve(okJson({ server_id: 42, grants: [] }))
      await flushPromises()
    })

    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(toastSuccessMock).not.toHaveBeenCalled()
  })

  it("does not reschedule polling when unmounted while a poll is awaiting status", async () => {
    const onOAuthStatusChange = vi.fn()
    const popup = {
      closed: false,
      opener: window,
      close: vi.fn(),
      location: { href: "" },
    }
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window)
    const pendingPolledStatus = deferredResponse()
    let statusCalls = 0

    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/mcp/42/oauth/status") {
        statusCalls += 1
        if (statusCalls === 1) {
          return Promise.resolve(okJson({ server_id: 42, grants: [] }))
        }
        return pendingPolledStatus.promise
      }
      if (url === "http://api.local/api/mcp/42/oauth/connect") {
        return Promise.resolve(
          okJson({ authorization_url: "https://auth.example.com/authorize" })
        )
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })

    const formData: MCPServerFormData = {
      name: "records",
      transport: "streamable_http",
      description: "",
      config: {
        url: "https://mcp.example.com/mcp",
        auth: {
          type: "mcp_oauth",
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scope: "records.read",
          client_id: "client-123",
        },
      },
    }

    const { unmount } = render(
      <CustomMcpForm
        mcpFormData={formData}
        setMcpFormData={vi.fn()}
        serverId={42}
        onOAuthStatusChange={onOAuthStatusChange}
      />
    )

    await waitFor(() => {
      expect(statusCalls).toBe(1)
    })

    vi.useFakeTimers()
    fireEvent.click(screen.getByText("tools.mcp.dialog.oauthConnect"))
    await act(async () => {
      await flushPromises()
    })
    act(() => {
      vi.advanceTimersByTime(3000)
    })
    await act(async () => {
      await flushPromises()
    })
    expect(statusCalls).toBe(2)

    unmount()
    await act(async () => {
      pendingPolledStatus.resolve(okJson({ server_id: 42, grants: [] }))
      await flushPromises()
      await vi.advanceTimersByTimeAsync(3000)
    })

    expect(statusCalls).toBe(2)
    expect(onOAuthStatusChange).not.toHaveBeenCalled()
  })

  it("restores asynchronously loaded masked OAuth client secrets on blur", async () => {
    const setMcpFormData = vi.fn()
    apiRequestMock.mockResolvedValue(okJson({ server_id: 42, grants: [] }))
    const baseFormData: MCPServerFormData = {
      name: "records",
      transport: "streamable_http",
      description: "",
      config: {
        url: "https://mcp.example.com/mcp",
        auth: {
          type: "mcp_oauth",
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scope: "records.read",
          client_id: "client-123",
        },
      },
    }

    const { rerender } = render(
      <CustomMcpForm
        mcpFormData={baseFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    const loadedFormData: MCPServerFormData = {
      ...baseFormData,
      config: {
        ...baseFormData.config!,
        auth: {
          ...baseFormData.config!.auth,
          client_secret: "********",
        },
      },
    }
    rerender(
      <CustomMcpForm
        mcpFormData={loadedFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText("tools.mcp.dialog.oauthClientSecret")).toHaveValue(
        "********"
      )
    })

    const clearedFormData: MCPServerFormData = {
      ...loadedFormData,
      config: {
        ...loadedFormData.config!,
        auth: {
          ...loadedFormData.config!.auth,
          client_secret: "",
        },
      },
    }
    rerender(
      <CustomMcpForm
        mcpFormData={clearedFormData}
        setMcpFormData={setMcpFormData}
        serverId={42}
      />
    )

    fireEvent.blur(screen.getByLabelText("tools.mcp.dialog.oauthClientSecret"))

    const updater = setMcpFormData.mock.calls.at(-1)?.[0]
    expect(typeof updater).toBe("function")
    const nextState = updater(clearedFormData)
    expect(nextState.config.auth.client_secret).toBe("********")
  })

  it("keeps an explicitly cleared masked OAuth client secret empty on blur", async () => {
    apiRequestMock.mockResolvedValue(okJson({ server_id: 42, grants: [] }))
    const loadedFormData: MCPServerFormData = {
      name: "records",
      transport: "streamable_http",
      description: "",
      config: {
        url: "https://mcp.example.com/mcp",
        auth: {
          type: "mcp_oauth",
          resource: "https://mcp.example.com/mcp",
          issuer: "https://auth.example.com",
          scope: "records.read",
          client_id: "client-123",
          client_secret: "********",
        },
      },
    }

    function Harness() {
      const [formData, setFormData] = React.useState(loadedFormData)
      return (
        <CustomMcpForm
          mcpFormData={formData}
          setMcpFormData={setFormData}
          serverId={42}
        />
      )
    }

    render(<Harness />)
    const clientSecret = screen.getByLabelText("tools.mcp.dialog.oauthClientSecret")

    fireEvent.focus(clientSecret)
    expect(clientSecret).toHaveValue("")
    fireEvent.change(clientSecret, {
      target: { value: "temporary-secret" },
    })
    expect(clientSecret).toHaveValue("temporary-secret")
    fireEvent.change(clientSecret, {
      target: { value: "" },
    })
    fireEvent.blur(clientSecret)

    expect(clientSecret).toHaveValue("")
  })
})

describe("CustomMcpForm environment key identity", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    cleanup()
  })

  function renderUserEnvForm(userEnv: Record<string, string>) {
    function Harness() {
      const [formData, setFormData] = React.useState<MCPServerFormData>({
        name: "local",
        transport: "stdio",
        description: "",
        config: { command: "python" },
        user_env: userEnv,
        can_edit_global: false,
      })
      return (
        <CustomMcpForm
          mcpFormData={formData}
          setMcpFormData={setFormData}
          serverId={42}
        />
      )
    }

    return render(<Harness />)
  }

  function getRemoveButtonForEnvKey(key: string): HTMLButtonElement {
    const row = screen.getByDisplayValue(key).parentElement
    const button = row?.querySelector("button")
    if (!(button instanceof HTMLButtonElement)) {
      throw new Error(`Remove button not found for environment key ${key}`)
    }
    return button
  }

  it("keeps persisted masked env key identities immutable", async () => {
    function Harness() {
      const [formData, setFormData] = React.useState<MCPServerFormData>({
        name: "local",
        transport: "stdio",
        description: "",
        config: {
          command: "python",
          env: { GLOBAL_TOKEN: "********" },
        },
        user_env: { USER_TOKEN: "********" },
        can_edit_global: true,
      })
      return (
        <CustomMcpForm
          mcpFormData={formData}
          setMcpFormData={setFormData}
          serverId={42}
        />
      )
    }

    render(<Harness />)
    const keyInputs = screen.getAllByPlaceholderText("tools.mcp.dialog.envKeyPlaceholder")
    const valueInputs = screen.getAllByPlaceholderText("tools.mcp.dialog.envValuePlaceholder")

    expect(keyInputs).toHaveLength(2)
    expect(keyInputs[0]).toHaveValue("USER_TOKEN")
    expect(keyInputs[0]).toBeDisabled()
    expect(keyInputs[1]).toHaveValue("GLOBAL_TOKEN")
    expect(keyInputs[1]).toBeDisabled()

    fireEvent.change(valueInputs[0], { target: { value: "replacement-secret" } })
    await waitFor(() => expect(valueInputs[0]).toHaveValue("replacement-secret"))
    expect(keyInputs[0]).toBeDisabled()
    expect(keyInputs[1]).toBeDisabled()

    fireEvent.change(valueInputs[0], { target: { value: "" } })
    fireEvent.blur(valueInputs[0])
    await waitFor(() => expect(valueInputs[0]).toHaveValue("********"))
  })

  it("keeps a persisted env secret when deletion is not confirmed", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    renderUserEnvForm({ USER_TOKEN: "********" })

    fireEvent.click(getRemoveButtonForEnvKey("USER_TOKEN"))

    expect(confirmMock).toHaveBeenCalledWith("tools.mcp.dialog.removeSecretConfirm")
    expect(screen.getByDisplayValue("USER_TOKEN")).toBeInTheDocument()
  })

  it("removes a persisted env secret after deletion is confirmed", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(true)
    renderUserEnvForm({ USER_TOKEN: "********" })

    fireEvent.click(getRemoveButtonForEnvKey("USER_TOKEN"))

    expect(confirmMock).toHaveBeenCalledWith("tools.mcp.dialog.removeSecretConfirm")
    expect(screen.queryByDisplayValue("USER_TOKEN")).not.toBeInTheDocument()
  })

  it("removes a new env entry without confirmation", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false)
    renderUserEnvForm({})

    fireEvent.click(screen.getByRole("button", { name: "tools.mcp.dialog.addEnvVariable" }))
    const keyInput = screen.getByPlaceholderText("tools.mcp.dialog.envKeyPlaceholder")
    fireEvent.change(keyInput, { target: { value: "DRAFT_TOKEN" } })
    fireEvent.click(getRemoveButtonForEnvKey("DRAFT_TOKEN"))

    expect(confirmMock).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("DRAFT_TOKEN")).not.toBeInTheDocument()
  })
})
