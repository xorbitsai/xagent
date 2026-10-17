import React, { StrictMode } from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"

const apiRequest = vi.hoisted(() => vi.fn())
vi.mock("@/lib/api-wrapper", () => ({ apiRequest }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
vi.mock("@/hooks/use-setup-status", () => ({ useSetupStatus: () => ({ passwordMinLength: 8 }) }))
vi.mock("@/components/auth/auth-page-shell", () => ({ AuthPageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock("@/components/auth/auth-form-card", () => ({ AuthFormCard: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))

import { ResetPasswordPage } from "./reset-password"

afterEach(() => { cleanup(); vi.clearAllMocks(); window.history.replaceState(null, "", "/") })

it.each([
  ["/reset-password#token=one-time-secret", "/reset-password"],
  ["/reset-password?token=one-time-secret&foo=1", "/reset-password?foo=1"],
  ["/reset-password?foo=1#token=one-time-secret", "/reset-password?foo=1"],
  ["/reset-password?token=old-secret&foo=1#token=one-time-secret", "/reset-password?foo=1"],
])("retains a token from %s across StrictMode effects and scrubs the URL", async (path, cleanPath) => {
  window.history.replaceState(null, "", path)
  apiRequest.mockResolvedValue(new Response(JSON.stringify({ success: false, message: "test response" })))
  render(<StrictMode><ResetPasswordPage /></StrictMode>)
  expect(window.location.hash).toBe("")
  expect(window.location.pathname + window.location.search).toBe(cleanPath)
  fireEvent.change(screen.getByPlaceholderText("resetPassword.form.password_placeholder"), { target: { value: "new-password-123" } })
  fireEvent.change(screen.getByPlaceholderText("resetPassword.form.confirm_password_placeholder"), { target: { value: "new-password-123" } })
  fireEvent.click(screen.getByRole("button", { name: "resetPassword.form.submit" }))
  await waitFor(() => expect(apiRequest).toHaveBeenCalledOnce())
  const [url, options] = apiRequest.mock.calls[0]
  expect(url).toMatch(/\/api\/auth\/reset-password$/)
  expect(JSON.parse(options.body)).toEqual({ token: "one-time-secret", new_password: "new-password-123" })
})

it.each(["/reset-password", "/reset-password#token=", "/reset-password?token="])("shows an invalid-link error for %s", (path) => {
  window.history.replaceState(null, "", path)
  render(<StrictMode><ResetPasswordPage /></StrictMode>)
  expect(screen.getByText("resetPassword.alerts.invalid_token")).toBeTruthy()
  expect((screen.getByRole("button", { name: "resetPassword.form.submit" }) as HTMLButtonElement).disabled).toBe(true)
  expect(window.location.href).not.toContain("token=")
  expect(apiRequest).not.toHaveBeenCalled()
})
