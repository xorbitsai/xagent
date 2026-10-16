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

it("retains the fragment token across StrictMode effects and sends it only in the POST body", async () => {
  window.history.replaceState(null, "", "/reset-password#token=one-time-secret")
  apiRequest.mockResolvedValue(new Response(JSON.stringify({ success: false, message: "test response" })))
  render(<StrictMode><ResetPasswordPage /></StrictMode>)
  expect(window.location.hash).toBe("")
  fireEvent.change(screen.getByPlaceholderText("resetPassword.form.password_placeholder"), { target: { value: "new-password-123" } })
  fireEvent.change(screen.getByPlaceholderText("resetPassword.form.confirm_password_placeholder"), { target: { value: "new-password-123" } })
  fireEvent.click(screen.getByRole("button", { name: "resetPassword.form.submit" }))
  await waitFor(() => expect(apiRequest).toHaveBeenCalledOnce())
  const [url, options] = apiRequest.mock.calls[0]
  expect(url).not.toContain("one-time-secret")
  expect(JSON.parse(options.body)).toEqual({ token: "one-time-secret", new_password: "new-password-123" })
})
