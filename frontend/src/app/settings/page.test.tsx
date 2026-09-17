import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import SettingsPage from "./page"
import { claimAuthLoginIntent, createAuthSession, inspectAuthSession, updateAuthSessionUser, type AuthSessionSnapshot, type AuthTokenPayload } from "@/lib/auth-cache"

const authState = vi.hoisted(() => ({
  user: { id: "1", username: "alice", email: "old@example.com" as string | null },
  session: null as AuthSessionSnapshot | null,
}))
const apiRequest = vi.hoisted(() => vi.fn())
const i18nState = vi.hoisted(() => ({ t: (key: string) => key, setLocale: vi.fn() }))

vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: i18nState.t, locale: "en", setLocale: i18nState.setLocale }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest }))
vi.mock("@/lib/task-runtime-ui-extension", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/task-runtime-ui-extension")
  >("@/lib/task-runtime-ui-extension")
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  return {
    ...actual,
    TaskRuntimeSettingsExtension: () => ReactModule.createElement(
      "section",
      { "data-testid": "task-runtime-settings-extension" },
      "runtime settings",
    ),
  }
})

afterEach(() => { cleanup(); vi.restoreAllMocks() })

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolvePromise => { resolve = resolvePromise })
  return { promise, resolve }
}
async function createSession(payload: AuthTokenPayload) {
  const claim = await claimAuthLoginIntent()
  if (claim.status !== "claimed") throw new Error("expected login intent")
  return createAuthSession(payload, claim.intent)
}

describe("SettingsPage auth profile synchronization", () => {
  beforeEach(async () => {
    localStorage.clear()
    apiRequest.mockReset()
    authState.user = { id: "1", username: "alice", email: "old@example.com" }
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_name: string, callback: () => Promise<unknown>) => callback()) },
    })
    const created = await createSession({
      user: authState.user,
      access_token: "old-access",
      refresh_token: "old-refresh",
    })
    if (created.status !== "created") throw new Error("expected session")
    authState.session = created.projection.snapshot
  })

  it("does not render the username when an email label is available", async () => {
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      success: true,
      user: authState.user,
    }), { headers: { "Content-Type": "application/json" } }))

    render(<SettingsPage />)

    await waitFor(() => expect(apiRequest).toHaveBeenCalledOnce())
    expect(screen.queryByDisplayValue("alice")).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("old@example.com")).toBeInTheDocument()
  })

  it("retains the username label for an email-less account", async () => {
    authState.user = { id: "1", username: "legacy-admin", email: null }
    const replacement = await createSession({
      user: authState.user,
      access_token: "legacy-access",
      refresh_token: "legacy-refresh",
    })
    if (replacement.status !== "created") throw new Error("expected replacement")
    authState.session = replacement.projection.snapshot
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      success: true,
      user: authState.user,
    }), { headers: { "Content-Type": "application/json" } }))

    render(<SettingsPage />)

    expect(screen.getByDisplayValue("legacy-admin")).toBeInTheDocument()
  })

  it("renders the distribution task runtime settings slot", async () => {
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      success: true,
      user: authState.user,
    }), { headers: { "Content-Type": "application/json" } }))

    render(<SettingsPage />)

    expect(screen.getByTestId("task-runtime-settings-extension")).toHaveTextContent(
      "runtime settings",
    )
  })

  it("does not apply an old profile response to a replacement login", async () => {
    let resolve!: (response: Response) => void
    apiRequest.mockReturnValue(new Promise<Response>(resolvePromise => { resolve = resolvePromise }))

    render(<SettingsPage />)
    await waitFor(() => expect(apiRequest).toHaveBeenCalledOnce())
    await createSession({
      user: { id: "1", username: "alice-new", email: "replacement@example.com" },
      access_token: "replacement-access",
      refresh_token: "replacement-refresh",
    })
    resolve(new Response(JSON.stringify({
      success: true,
      user: { id: "1", username: "alice", email: "stale@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))

    await waitFor(() => {
      expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: {
        token: "replacement-access",
        user: { email: "replacement@example.com" },
      } } })
    })
  })

  it("resyncs the displayed email from canonical storage when the profile has advanced", async () => {
    const response = deferred<Response>()
    apiRequest.mockReturnValue(response.promise)
    render(<SettingsPage />)
    await waitFor(() => expect(apiRequest).toHaveBeenCalledOnce())
    expect(await updateAuthSessionUser(authState.session!, {
      ...authState.user,
      email: "canonical@example.com",
    })).toMatchObject({ status: "updated" })
    response.resolve(new Response(JSON.stringify({
      success: true,
      user: { ...authState.user, email: "stale@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toHaveValue("canonical@example.com"))
  })

  it("does not let an older load response overwrite a newer session load", async () => {
    const first = deferred<Response>()
    const second = deferred<Response>()
    apiRequest.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
    const page = render(<SettingsPage />)
    await waitFor(() => expect(apiRequest).toHaveBeenCalledTimes(1))
    const replacement = await createSession({
      user: { id: "1", username: "alice", email: "replacement@example.com" },
      access_token: "replacement-access",
      refresh_token: "replacement-refresh",
    })
    if (replacement.status !== "created") throw new Error("expected replacement")
    authState.session = replacement.projection.snapshot
    page.rerender(<SettingsPage />)
    await waitFor(() => expect(apiRequest).toHaveBeenCalledTimes(2))
    second.resolve(new Response(JSON.stringify({
      success: true,
      user: { id: "1", username: "alice", email: "newest@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toHaveValue("newest@example.com"))
    first.resolve(new Response(JSON.stringify({
      success: true,
      user: { id: "1", username: "alice", email: "stale@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toHaveValue("newest@example.com"))
  })

  it("does not publish a pending email response before conditional mutation and resyncs after replacement", async () => {
    const update = deferred<Response>()
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: authState.user }), { headers: { "Content-Type": "application/json" } }))
      .mockReturnValueOnce(update.promise)
    render(<SettingsPage />)
    const input = await screen.findByLabelText("settings.email.current")
    await waitFor(() => expect(input).not.toBeDisabled())
    fireEvent.change(input, { target: { value: "pending@example.com" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.submit" }))
    await waitFor(() => expect(apiRequest).toHaveBeenCalledTimes(2))
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { user: { email: "old@example.com" } } } })
    expect(screen.queryByText("settings.email.success")).not.toBeInTheDocument()

    const replacement = await createSession({
      user: { id: "1", username: "alice", email: "replacement@example.com" },
      access_token: "replacement-access", refresh_token: "replacement-refresh",
    })
    if (replacement.status !== "created") throw new Error("expected replacement")
    update.resolve(new Response(JSON.stringify({
      success: true, user: { id: "1", username: "alice", email: "stale@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toHaveValue("replacement@example.com"))
    expect(screen.queryByText("settings.email.success")).not.toBeInTheDocument()
  })
})
