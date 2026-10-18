import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { LoginPage } from "./login"

const apiRequest = vi.hoisted(() => vi.fn())
const claimAuthLoginIntent = vi.hoisted(() => vi.fn())
const claimOidcAuthLoginIntent = vi.hoisted(() => vi.fn())
const createAuthSession = vi.hoisted(() => vi.fn())
const translations = vi.hoisted(() => ({ t: (key: string) => key, tDynamic: (_key: string, fallback: string) => fallback }))
const authMutationUnavailableTranslationKey = vi.hoisted(() => vi.fn((reason: string) => `login.alerts.${reason}`))

vi.mock("@/components/auth/auth-form-card", () => ({
  AuthFormCard: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
}))
vi.mock("@/components/auth/auth-page-shell", () => ({
  AuthPageShell: ({ children }: { children: React.ReactNode }) => <main>{children}</main>,
}))
vi.mock("@/components/ui/button", () => ({
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
}))
vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => translations,
}))
vi.mock("@/hooks/use-setup-status", () => ({
  useSetupStatus: () => ({ isLoading: false, registrationEnabled: false }),
}))
vi.mock("@/lib/api-wrapper", () => ({ apiRequest }))
vi.mock("@/lib/auth-cache", () => ({ claimAuthLoginIntent, claimOidcAuthLoginIntent, createAuthSession }))
vi.mock("@/lib/auth-pages", () => ({ authMutationUnavailableTranslationKey }))
vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent", logoPath: "/logo.svg", logoAlt: "Xagent", tagline: "Build agents" }),
}))
vi.mock("@/lib/utils", () => ({ getApiUrl: () => "https://api.example" }))
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}))

function captureNavigation() {
  const implementation = (window.location as unknown as Record<symbol, unknown>)[
    Object.getOwnPropertySymbols(window.location).find(symbol => String(symbol) === "Symbol(impl)")!
  ] as { _locationObjectSetterNavigate: (url: { path: string[] }) => void }
  const original = implementation._locationObjectSetterNavigate
  const targets: string[] = []
  implementation._locationObjectSetterNavigate = url => {
    targets.push(`/${url.path.join("/")}`)
  }
  return {
    targets,
    restore: () => { implementation._locationObjectSetterNavigate = original },
  }
}

function renderLogin() {
  render(<LoginPage />)
  fireEvent.change(screen.getByPlaceholderText("login.form.username_placeholder"), { target: { value: "alice" } })
  fireEvent.change(screen.getByPlaceholderText("login.form.password_placeholder"), { target: { value: "secret" } })
}

describe("LoginPage auth-session creation", () => {
  it.each(["storage_unavailable", "coordination_unavailable", "operation_failed"])("shows an auth-unavailable redirect reason: %s", async reason => {
    window.history.replaceState({}, "", `/login?auth_unavailable=${reason}`)
    try {
      render(<LoginPage />)
      expect(await screen.findByText(`login.alerts.${reason}`)).toBeInTheDocument()
    } finally { window.history.replaceState({}, "", "/") }
  })

  it("shows an OIDC redirect failure, then clears it when typing", async () => {
    window.history.replaceState({}, "", "/login?oidc_error=invalid_state")
    try {
      render(<LoginPage />)
      expect(await screen.findByText("login.alerts.google_failed")).toBeInTheDocument()
      fireEvent.change(screen.getByPlaceholderText("login.form.username_placeholder"), { target: { value: "alice" } })
      expect(screen.queryByText("login.alerts.google_failed")).not.toBeInTheDocument()
    } finally { window.history.replaceState({}, "", "/") }
  })

  it("lets users reveal and hide the password without changing it", () => {
    renderLogin()
    const password = screen.getByPlaceholderText("login.form.password_placeholder")
    fireEvent.click(screen.getByRole("button", { name: "Show password" }))
    expect(password).toHaveAttribute("type", "text")
    fireEvent.click(screen.getByRole("button", { name: "Hide password" }))
    expect(password).toHaveAttribute("type", "password")
    expect(password).toHaveValue("secret")
  })

  it("does not navigate to Google when login coordination is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ configured: true }))))
    claimOidcAuthLoginIntent.mockResolvedValue({ status: "unavailable", reason: "coordination_unavailable" })
    const navigation = captureNavigation()
    try {
      render(<LoginPage />)
      fireEvent.click(await screen.findByRole("button", { name: /login.google.continue/ }))
      expect(await screen.findByText("login.alerts.coordination_unavailable")).toBeInTheDocument()
      expect(claimOidcAuthLoginIntent).toHaveBeenCalledOnce()
      expect(navigation.targets).toEqual([])
    } finally { navigation.restore() }
  })

  it("fails closed when checking Google availability returns an error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503 })))
    render(<LoginPage />)
    expect(await screen.findByRole("button", { name: /login.google.unavailable/ })).toBeDisabled()
  })

  it("explains why a password change requires signing in again", async () => {
    window.history.replaceState({}, "", "/login?password_changed=1")
    try {
      render(<LoginPage />)
      expect(await screen.findByRole("status")).toHaveTextContent("settings.password.sign_in_again")
    } finally {
      window.history.replaceState({}, "", "/")
    }
  })
  beforeEach(() => {
    apiRequest.mockReset()
    claimAuthLoginIntent.mockReset()
    claimOidcAuthLoginIntent.mockReset()
    createAuthSession.mockReset()
    claimAuthLoginIntent.mockResolvedValue({ status: "claimed", intent: { id: "password-intent" } })
    claimOidcAuthLoginIntent.mockResolvedValue({ status: "claimed", intent: { id: "oidc-intent" } })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ configured: false }), {
      headers: { "Content-Type": "application/json" },
    })))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("waits for a created auth session before redirecting after a password login", async () => {
    let resolveSession!: (value: { status: "created"; projection: object }) => void
    createAuthSession.mockReturnValue(new Promise(resolve => { resolveSession = resolve }))
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      user: { id: "1", username: "alice" }, access_token: "access", refresh_token: "refresh",
    }), { headers: { "Content-Type": "application/json" } }))
    const navigation = captureNavigation()

    renderLogin()
    fireEvent.click(screen.getByRole("button", { name: "login.form.submit" }))

    await waitFor(() => expect(createAuthSession).toHaveBeenCalledWith(expect.objectContaining({
      user: { id: "1", username: "alice" }, access_token: "access", refresh_token: "refresh",
    }), { id: "password-intent" }))
    expect(navigation.targets).toEqual([])

    resolveSession({ status: "created", projection: {} })
    await waitFor(() => expect(navigation.targets).toEqual(["/"]))
    navigation.restore()
  })

  it("shows an auth failure and does not redirect when session creation is rejected", async () => {
    createAuthSession.mockResolvedValue({ status: "invalid" })
    apiRequest.mockResolvedValue(new Response(JSON.stringify({
      user: { id: "1", username: "alice" }, access_token: "access", refresh_token: "refresh",
    }), { headers: { "Content-Type": "application/json" } }))
    const navigation = captureNavigation()

    renderLogin()
    fireEvent.click(screen.getByRole("button", { name: "login.form.submit" }))

    expect(await screen.findByText("login.alerts.auth_failed")).toBeInTheDocument()
    expect(navigation.targets).toEqual([])
    navigation.restore()
  })
  it("uses the shared availability translation key when browser coordination is unavailable", async () => {
    claimAuthLoginIntent.mockResolvedValue({ status: "unavailable", reason: "coordination_unavailable" })

    renderLogin()
    fireEvent.click(screen.getByRole("button", { name: "login.form.submit" }))

    expect(await screen.findByText("login.alerts.coordination_unavailable")).toBeInTheDocument()
    expect(authMutationUnavailableTranslationKey).toHaveBeenCalledWith("coordination_unavailable")
    expect(apiRequest).not.toHaveBeenCalled()
  })
  it("maps password-login availability failures through the shared translation-key mapper", async () => {
    claimAuthLoginIntent.mockResolvedValue({ status: "unavailable", reason: "storage_unavailable" })

    renderLogin()
    fireEvent.click(screen.getByRole("button", { name: "login.form.submit" }))

    await waitFor(() => expect(authMutationUnavailableTranslationKey).toHaveBeenCalledWith("storage_unavailable"))
  })
})
