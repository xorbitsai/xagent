import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import UserManagement from "./user-management"

const apiRequestMock = vi.hoisted(() => vi.fn())
const authMock = vi.hoisted(() => ({ user: { id: "1", is_admin: true } }))
const toastMock = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: apiRequestMock }))
vi.mock("@/components/ui/sonner", () => ({ toast: toastMock }))
vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  getApiUrl: () => "http://api.test",
}))
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => authMock,
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en", t: (key: string) => key }),
}))
vi.mock("next/link", () => ({
  default: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a {...props}>{children}</a>
  ),
}))

describe("UserManagement account labels", () => {
  beforeEach(() => {
    vi.stubGlobal("React", React)
    apiRequestMock.mockReset()
    toastMock.success.mockReset()
    toastMock.error.mockReset()
    apiRequestMock.mockResolvedValue(userListResponse())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it("shows email instead of an opaque username", async () => {
    render(<UserManagement />)

    expect(await screen.findByText("managed@example.com")).toBeInTheDocument()
    expect(
      screen.queryByText("acct_0123456789abcdef0123456789abcdef"),
    ).not.toBeInTheDocument()
  })

  it("labels the mixed-identity column as an account", async () => {
    render(<UserManagement />)

    expect(
      await screen.findByRole("columnheader", {
        name: "userManagement.list.table.account",
      }),
    ).toBeInTheDocument()
  })

  it("shows successful deletion feedback in a toast", async () => {
    apiRequestMock.mockResolvedValueOnce(userListResponse())
    apiRequestMock.mockResolvedValueOnce({ ok: true })
    apiRequestMock.mockResolvedValueOnce(userListResponse())
    render(<UserManagement />)

    openDeleteConfirmation(await screen.findByText("managed@example.com"))
    fireEvent.click(screen.getByRole("button", {
      name: "userManagement.list.table.confirm_delete",
    }))

    await waitFor(() => {
      expect(toastMock.success).toHaveBeenCalledWith(
        "userManagement.list.alerts.delete_success_prefix" +
        "managed@example.com" +
        "userManagement.list.alerts.delete_success_suffix",
      )
    })
  })

  it("shows failed deletion feedback in a toast", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    apiRequestMock.mockResolvedValueOnce(userListResponse())
    apiRequestMock.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: "cannot delete" }),
    })
    render(<UserManagement />)

    openDeleteConfirmation(await screen.findByText("managed@example.com"))
    fireEvent.click(screen.getByRole("button", {
      name: "userManagement.list.table.confirm_delete",
    }))

    await waitFor(() => {
      expect(toastMock.error).toHaveBeenCalledWith(
        "userManagement.list.alerts.delete_failed_prefixcannot delete",
      )
    })
  })
})

function userListResponse() {
  return {
    ok: true,
    json: async () => ({
      users: [
        {
          id: 2,
          username: "acct_0123456789abcdef0123456789abcdef",
          email: "managed@example.com",
          is_admin: false,
          created_at: "2026-08-11T00:00:00Z",
          updated_at: "2026-08-11T00:00:00Z",
        },
      ],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    }),
  }
}

function openDeleteConfirmation(accountLabel: HTMLElement) {
  const row = accountLabel.closest("tr")
  const button = row?.querySelector("button")
  if (!button) throw new Error("expected account deletion button")
  fireEvent.click(button)
}
