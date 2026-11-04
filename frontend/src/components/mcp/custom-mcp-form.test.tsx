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

const SIGNED_UPLOADS = [
  {
    tool_name: "attach_document",
    file_argument: "document_file_id",
    issuing_tool_name: "request_upload",
    allowed_upload_hosts: ["uploads.example.test"],
    upload_path_prefix: "/staged/",
    content_types: { ".pdf": "application/pdf" },
  },
]
const SETTINGS_ERROR = "signed_uploads[0] must have exactly the documented keys"
// Stored entries that cannot be used: the detail echoes them as they are.
const BAD_SIGNED_UPLOADS = [{ tool_name: "attach_document", file_argument: "document_file_id" }]
const NOT_A_LIST_ERROR = "signed_uploads must be a list"

const UPLOAD_KEY = "tools.mcp.signedUploads"

describe("CustomMcpForm file uploads", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    translateMock.mockClear()
    cleanup()
  })

  function renderUploadsForm(overrides: Partial<MCPServerFormData> = {}) {
    const seen: MCPServerFormData[] = []
    const initial: MCPServerFormData = {
      name: "records_mcp",
      transport: "streamable_http",
      description: "",
      config: { url: "https://mcp.example.com/mcp" },
      can_edit_global: true,
      signed_uploads: null,
      signed_uploads_error: null,
      ...overrides,
    }

    function Harness({ onChange }: { onChange: (data: MCPServerFormData) => void }) {
      const [formData, setFormData] = React.useState<MCPServerFormData>(initial)
      React.useEffect(() => {
        onChange(formData)
      }, [formData, onChange])
      return (
        <CustomMcpForm mcpFormData={formData} setMcpFormData={setFormData} serverId={42} />
      )
    }

    render(<Harness onChange={(data) => seen.push(data)} />)
    return { initial, latest: () => seen[seen.length - 1] }
  }

  it.each([
    ["sse", true],
    ["streamable_http", true],
    ["stdio", false],
    ["websocket", false],
  ])("renders file uploads only for sse and streamable_http", (transport, shown) => {
    renderUploadsForm({ transport })

    if (shown) expect(screen.getByText(`${UPLOAD_KEY}.title`)).toBeInTheDocument()
    else expect(screen.queryByText(`${UPLOAD_KEY}.title`)).not.toBeInTheDocument()
  })

  it.each([
    ["valid settings", SIGNED_UPLOADS, null],
    ["entries that cannot be used", BAD_SIGNED_UPLOADS, SETTINGS_ERROR],
    ["no settings", null, null],
    ["a stored value that is not a list", null, NOT_A_LIST_ERROR],
    ["an empty list", [], null],
  ])("shows stored settings without writing to the form", async (_name, stored, error) => {
    const { initial, latest } = renderUploadsForm({
      signed_uploads: stored,
      signed_uploads_error: error,
    })
    await act(async () => {
      await flushPromises()
    })

    expect(latest()).toBe(initial)
    expect(latest().signed_uploads).toBe(stored)
    expect(latest().signed_uploads_error).toBe(error)
    if (stored === SIGNED_UPLOADS) {
      expect(screen.getByLabelText(`${UPLOAD_KEY}.toolName`)).toHaveValue("attach_document")
      expect(screen.getByLabelText(`${UPLOAD_KEY}.allowedUploadHosts`)).toHaveValue(
        "uploads.example.test",
      )
    }
    if (stored === BAD_SIGNED_UPLOADS) {
      expect(screen.getByLabelText(`${UPLOAD_KEY}.toolName`)).toHaveValue("attach_document")
    }
  })

  it("disables the whole block for a viewer who cannot edit the shared config", () => {
    renderUploadsForm({
      can_edit_global: false,
      signed_uploads: SIGNED_UPLOADS,
      signed_uploads_error: null,
    })

    const block = screen.getByText(`${UPLOAD_KEY}.title`).closest("fieldset")
    if (!block) throw new Error("File uploads block not found")
    const fields = block.querySelectorAll("input, textarea")
    expect(fields.length).toBeGreaterThan(0)
    fields.forEach((field) => expect(field).toBeDisabled())
    for (const name of ["addTool", "removeTool", "addContentType", "removeContentType"]) {
      expect(screen.getByRole("button", { name: `${UPLOAD_KEY}.${name}` })).toBeDisabled()
    }
  })

  it("disables Clear for a viewer who cannot edit the shared config", () => {
    renderUploadsForm({
      can_edit_global: false,
      signed_uploads: BAD_SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })

    expect(screen.getByRole("button", { name: "common.clear" })).toBeDisabled()
    expect(screen.getByRole("button", { name: `${UPLOAD_KEY}.addTool` })).toBeDisabled()
    expect(screen.getByLabelText(`${UPLOAD_KEY}.toolName`)).toBeDisabled()
  })

  it.each([
    ["entries that cannot be used", BAD_SIGNED_UPLOADS, SETTINGS_ERROR],
    ["a stored value that is not a list", null, NOT_A_LIST_ERROR],
  ])("shows the settings error with Clear and clears to an empty list", (_name, stored, error) => {
    const { latest } = renderUploadsForm({
      signed_uploads: stored,
      signed_uploads_error: error,
    })

    if (stored === null) {
      expect(screen.getByText(`${UPLOAD_KEY}.empty`)).toBeInTheDocument()
    } else {
      expect(screen.getByLabelText(`${UPLOAD_KEY}.toolName`)).toHaveValue("attach_document")
    }
    expect(screen.getByText(`${UPLOAD_KEY}.invalid`)).toBeInTheDocument()
    expect(translateMock).toHaveBeenCalledWith(`${UPLOAD_KEY}.invalid`, { error })

    fireEvent.click(screen.getByRole("button", { name: "common.clear" }))

    expect(latest().signed_uploads).toEqual([])
    expect(screen.queryByText(`${UPLOAD_KEY}.invalid`)).not.toBeInTheDocument()
    expect(screen.getByText(`${UPLOAD_KEY}.willBeCleared`)).toBeInTheDocument()
    expect(screen.getByText(`${UPLOAD_KEY}.empty`)).toBeInTheDocument()
    expect(screen.queryByLabelText(`${UPLOAD_KEY}.toolName`)).not.toBeInTheDocument()
  })

  it("lets the owner complete an invalid entry in place", () => {
    const { latest } = renderUploadsForm({
      signed_uploads: BAD_SIGNED_UPLOADS,
      signed_uploads_error: SETTINGS_ERROR,
    })

    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.issuingToolName`), {
      target: { value: "request_upload" },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.allowedUploadHosts`), {
      target: { value: "uploads.example.test" },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.uploadPathPrefix`), {
      target: { value: "/staged/" },
    })
    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addContentType` }))
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.extension`), {
      target: { value: ".pdf" },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.contentType`), {
      target: { value: "application/pdf" },
    })

    expect(latest().signed_uploads).toEqual(SIGNED_UPLOADS)
  })

  it.each([
    [
      "valid settings",
      { signed_uploads: SIGNED_UPLOADS, signed_uploads_error: null },
      true,
      true,
    ],
    [
      "a stored value that is not a list",
      { signed_uploads: null, signed_uploads_error: NOT_A_LIST_ERROR },
      true,
      true,
    ],
    ["no settings", { signed_uploads: null, signed_uploads_error: null }, true, false],
    [
      "a viewer who cannot edit the shared config",
      { transport: "stdio", can_edit_global: false, signed_uploads: SIGNED_UPLOADS },
      false,
      false,
    ],
  ])(
    "warns that settings will be cleared when an owner leaves sse or streamable_http",
    (_name, overrides, canSwitch, expected) => {
      renderUploadsForm(overrides as Partial<MCPServerFormData>)
      expect(screen.queryByText(`${UPLOAD_KEY}.willBeCleared`)).not.toBeInTheDocument()

      if (canSwitch) fireEvent.click(screen.getByRole("button", { name: "STDIO" }))

      if (expected) expect(screen.getByText(`${UPLOAD_KEY}.willBeCleared`)).toBeInTheDocument()
      else expect(screen.queryByText(`${UPLOAD_KEY}.willBeCleared`)).not.toBeInTheDocument()
    },
  )

  it("writes trimmed entries and drops blank host lines and empty rows", () => {
    const { latest } = renderUploadsForm()

    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addTool` }))
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.toolName`), {
      target: { value: " attach_document " },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.fileArgument`), {
      target: { value: " document_file_id " },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.issuingToolName`), {
      target: { value: " request_upload " },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.allowedUploadHosts`), {
      target: { value: "uploads.example.test\n\n  \n" },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.uploadPathPrefix`), {
      target: { value: " /staged/ " },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.extension`), {
      target: { value: " .pdf " },
    })
    fireEvent.change(screen.getByLabelText(`${UPLOAD_KEY}.contentType`), {
      target: { value: " application/pdf " },
    })
    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addContentType` }))

    expect(latest().signed_uploads).toEqual(SIGNED_UPLOADS)
  })

  it("keeps the later file type when two rows share an extension", () => {
    const { latest } = renderUploadsForm()

    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addTool` }))
    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addContentType` }))
    const extensions = screen.getAllByLabelText(`${UPLOAD_KEY}.extension`)
    const contentTypes = screen.getAllByLabelText(`${UPLOAD_KEY}.contentType`)
    fireEvent.change(extensions[0], { target: { value: ".pdf" } })
    fireEvent.change(contentTypes[0], { target: { value: "application/pdf" } })
    fireEvent.change(extensions[1], { target: { value: ".pdf" } })
    fireEvent.change(contentTypes[1], { target: { value: "application/x-pdf" } })

    const entries = latest().signed_uploads as Record<string, unknown>[]
    expect(entries).toHaveLength(1)
    expect(entries[0].content_types).toEqual({ ".pdf": "application/x-pdf" })
  })

  it("drops a tool entry whose fields are all empty", () => {
    const { latest } = renderUploadsForm({
      signed_uploads: SIGNED_UPLOADS,
      signed_uploads_error: null,
    })

    fireEvent.click(screen.getByRole("button", { name: `${UPLOAD_KEY}.addTool` }))

    expect(screen.getAllByLabelText(`${UPLOAD_KEY}.toolName`)).toHaveLength(2)
    expect(latest().signed_uploads).toEqual(SIGNED_UPLOADS)
  })

  it("removes a tool", () => {
    const second = { ...SIGNED_UPLOADS[0], tool_name: "attach_image" }
    const { latest } = renderUploadsForm({
      signed_uploads: [SIGNED_UPLOADS[0], second],
      signed_uploads_error: null,
    })

    fireEvent.click(screen.getAllByRole("button", { name: `${UPLOAD_KEY}.removeTool` })[0])

    expect(latest().signed_uploads).toEqual([second])
    expect(screen.getAllByLabelText(`${UPLOAD_KEY}.toolName`)).toHaveLength(1)
  })
})
