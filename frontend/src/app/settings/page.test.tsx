import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import SettingsPage from "./page"
import { claimAuthLoginIntent, createAuthSession, inspectAuthSession, updateAuthSessionUser, type AuthSessionSnapshot, type AuthTokenPayload } from "@/lib/auth-cache"

const authState = vi.hoisted(() => ({
  user: { id: "1", username: "alice", email: "old@example.com" } as { id: string; username: string; email: string | null; has_password?: boolean },
  session: null as AuthSessionSnapshot | null,
}))
const apiRequest = vi.hoisted(() => vi.fn())
const i18nState = vi.hoisted(() => ({ t: (key: string) => key, setLocale: vi.fn() }))

vi.mock("@/contexts/auth-context", () => ({ useAuth: () => authState }))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: i18nState.t, locale: "en", setLocale: i18nState.setLocale }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest }))
vi.mock("@/hooks/use-setup-status", () => ({
  useSetupStatus: () => ({ passwordMinLength: 8 }),
}))
vi.mock("@/components/settings/execution-budget", () => ({
  ExecutionBudgetSettings: () => <section data-testid="execution-budget-settings" />,
}))
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

function captureNavigation() {
  const implementation = (window.location as unknown as Record<symbol, unknown>)[
    Object.getOwnPropertySymbols(window.location).find(symbol => String(symbol) === "Symbol(impl)")!
  ] as { _locationObjectSetterNavigate: (url: { path: string[]; query: string | null }) => void }
  const targets: string[] = []
  vi.spyOn(implementation, "_locationObjectSetterNavigate").mockImplementation(url => {
    targets.push(`/${url.path.join("/")}${url.query ? `?${url.query}` : ""}`)
  })
  return targets
}

function submitPassword() {
  fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#current-password" }), { target: { value: "current-password" } })
  fireEvent.change(screen.getByLabelText("settings.password.new"), { target: { value: "new-password-123" } })
  fireEvent.change(screen.getByLabelText("settings.password.confirm"), { target: { value: "new-password-123" } })
  fireEvent.click(screen.getByRole("button", { name: "settings.password.submit" }))
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

  it.each([true, false])("signs out only when the password result succeeds (%s)", async success => {
    const targets = captureNavigation()
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: authState.user })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ success, message: "Password changed; sign in again" })))
    render(<SettingsPage />)
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toBeEnabled())
    submitPassword()
    if (success) {
      await waitFor(() => expect(targets).toEqual(["/login?password_changed=1"]))
      expect(inspectAuthSession().status).toBe("absent")
    } else {
      expect(await screen.findByText("Password changed; sign in again")).toBeInTheDocument()
      expect(screen.queryByText("settings.password.sign_in_again")).not.toBeInTheDocument()
      expect(targets).toEqual([])
      expect(inspectAuthSession().status).toBe("valid")
    }
  })

  it("does not sign out a replacement login when an old password response succeeds", async () => {
    const targets = captureNavigation()
    const pending = deferred<Response>()
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: authState.user })))
      .mockReturnValueOnce(pending.promise)
    render(<SettingsPage />)
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toBeEnabled())
    submitPassword()
    await waitFor(() => expect(apiRequest).toHaveBeenCalledTimes(2))
    await createSession({ user: authState.user, access_token: "replacement-access", refresh_token: "replacement-refresh" })
    pending.resolve(new Response(JSON.stringify({ success: true })))
    await screen.findByText("settings.password.sign_in_again")
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { token: "replacement-access" } } })
    expect(targets).toEqual([])
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

  it("keeps the recovery address unchanged until the emailed code is confirmed", async () => {
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: authState.user })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, pending_confirmation: true, user: authState.user })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: { ...authState.user, email: "new@example.com" } })))
    const view = render(<SettingsPage />)
    const input = screen.getByLabelText("settings.email.current")
    await waitFor(() => expect(input).not.toBeDisabled())
    const loaded = inspectAuthSession()
    if (loaded.status !== "valid") throw new Error("expected loaded profile")
    authState.session = loaded.projection.snapshot
    view.rerender(<SettingsPage />)
    fireEvent.change(input, { target: { value: "new@example.com" } })
    fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#email-current-password" }), { target: { value: "current-password" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.submit" }))
    await screen.findByText("settings.email.confirmationSent")
    expect(input).toHaveValue("old@example.com")
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { user: { email: "old@example.com" } } } })
    const staged = inspectAuthSession()
    if (staged.status !== "valid") throw new Error("expected staged profile")
    authState.session = staged.projection.snapshot
    view.rerender(<SettingsPage />)
    fireEvent.change(screen.getByLabelText("settings.email.confirmationCode"), { target: { value: "confirmation-secret" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.confirm" }))
    await screen.findByText("settings.email.success")
    expect(apiRequest.mock.calls[2][0]).toMatch(/\/api\/auth\/email\/confirm$/)
    expect(JSON.parse(apiRequest.mock.calls[2][1].body)).toEqual({ token: "confirmation-secret" })
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { user: { email: "new@example.com" } } } })
  })

  it.each(["smtp", "provider"])("explains and disables unavailable email changes (%s)", async reason => {
    authState.user.has_password = reason !== "provider"
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      success: true, user: { ...authState.user, email_change_available: reason !== "smtp" },
    })))
    render(<SettingsPage />)
    await screen.findByText(reason === "smtp" ? "settings.email.smtpRequired" : "settings.email.providerManaged")
    expect(screen.getByLabelText("settings.email.current")).toBeDisabled()
    expect(screen.getByRole("button", { name: "settings.email.submit" })).toBeDisabled()
    expect(screen.queryByLabelText("settings.email.confirmationCode")).not.toBeInTheDocument()
  })

  it("restores a pending confirmation after reload and preserves it on a confirmation error", async () => {
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: { ...authState.user, pending_email: "pending@example.com" } })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: false, message: "Invalid or expired confirmation code" })))
    const view = render(<SettingsPage />)
    await screen.findByLabelText("settings.email.confirmationCode")
    const loaded = inspectAuthSession()
    if (loaded.status !== "valid") throw new Error("expected profile")
    authState.session = loaded.projection.snapshot
    view.rerender(<SettingsPage />)
    expect(screen.getByLabelText("settings.email.current")).toHaveValue("old@example.com")
    fireEvent.change(screen.getByLabelText("settings.email.confirmationCode"), { target: { value: "bad-code" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.confirm" }))
    await screen.findByText("Invalid or expired confirmation code")
    expect(screen.getByLabelText("settings.email.confirmationCode")).toHaveValue("bad-code")
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { user: { email: "old@example.com" } } } })
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

  it.each([false, true])("reports an unchanged address without claiming an update (pending=%s)", async pending => {
    const pending_email = pending ? "pending@example.com" : null
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: { ...authState.user, pending_email } })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, unchanged: true, user: { ...authState.user, pending_email } })))
    render(<SettingsPage />)
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#email-current-password" }), { target: { value: "current-password" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.submit" }))
    await screen.findByText(pending ? "settings.email.unchangedPending" : "settings.email.unchanged")
    expect(screen.queryByText("settings.email.success")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("settings.email.confirmationCode") !== null).toBe(pending)
  })

  it.each(["profile", "request", "confirm"])("renders structured validation errors safely (%s)", async phase => {
    const invalid = { message: { invalid: true }, detail: [{ loc: ["body", "token"], msg: "Too long", type: "string_too_long" }] }
    const error = new Response(JSON.stringify(invalid), { status: 422 })
    if (phase === "profile") {
      apiRequest.mockResolvedValueOnce(error)
    } else {
      apiRequest
        .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: { ...authState.user, pending_email: "pending@example.com" } })))
        .mockResolvedValueOnce(error)
    }
    render(<SettingsPage />)
    if (phase !== "profile") {
      await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toBeEnabled())
      if (phase === "confirm") {
        const code = screen.getByLabelText("settings.email.confirmationCode")
        expect(code).toHaveAttribute("maxlength", "128")
        fireEvent.change(code, { target: { value: "x".repeat(129) } })
        fireEvent.click(screen.getByRole("button", { name: "settings.email.confirm" }))
      } else {
        fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#email-current-password" }), { target: { value: "current-password" } })
        fireEvent.click(screen.getByRole("button", { name: "settings.email.submit" }))
      }
    }
    await screen.findByText("settings.email.failed")
    expect(screen.getByLabelText("settings.email.current")).toHaveValue("old@example.com")
  })

  it("keeps the sent confirmation available when another profile refresh advances the cache", async () => {
    const response = deferred<Response>()
    apiRequest
      .mockResolvedValueOnce(new Response(JSON.stringify({ success: true, user: authState.user })))
      .mockReturnValueOnce(response.promise)
    const page = render(<SettingsPage />)
    const input = screen.getByLabelText("settings.email.current")
    await waitFor(() => expect(input).toBeEnabled())
    const loaded = inspectAuthSession()
    if (loaded.status !== "valid") throw new Error("expected loaded profile")
    authState.session = loaded.projection.snapshot
    page.rerender(<SettingsPage />)
    fireEvent.change(input, { target: { value: "pending@example.com" } })
    fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#email-current-password" }), { target: { value: "current-password" } })
    fireEvent.click(screen.getByRole("button", { name: "settings.email.submit" }))
    await waitFor(() => expect(apiRequest).toHaveBeenCalledTimes(2))
    expect(await updateAuthSessionUser(authState.session!, {
      ...authState.user, email: "canonical@example.com",
    })).toMatchObject({ status: "updated" })
    response.resolve(new Response(JSON.stringify({
      success: true, pending_confirmation: true,
      user: { ...authState.user, pending_email: "pending@example.com" },
    })))
    await screen.findByText("settings.email.confirmationSent")
    expect(screen.getByLabelText("settings.email.confirmationCode")).toBeEnabled()
    expect(input).toHaveValue("canonical@example.com")
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { user: { email: "canonical@example.com" } } } })
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

  it("places execution budget after the common account settings", async () => {
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      success: true,
      user: authState.user,
    }), { headers: { "Content-Type": "application/json" } }))

    render(<SettingsPage />)

    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toBeEnabled())
    const budget = screen.getByTestId("execution-budget-settings")
    expect(budget.parentElement?.lastElementChild).toBe(budget)
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
    fireEvent.change(screen.getByLabelText("settings.password.current", { selector: "#email-current-password" }), { target: { value: "current-password" } })
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
      success: true, pending_confirmation: true, user: { id: "1", username: "alice", email: "stale@example.com", pending_email: "pending@example.com" },
    }), { headers: { "Content-Type": "application/json" } }))
    await waitFor(() => expect(screen.getByLabelText("settings.email.current")).toHaveValue("replacement@example.com"))
    expect(screen.queryByText("settings.email.success")).not.toBeInTheDocument()
    expect(screen.queryByText("settings.email.confirmationSent")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("settings.email.confirmationCode")).not.toBeInTheDocument()
  })
})
