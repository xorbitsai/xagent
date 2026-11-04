import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useOAuthAccountStep, type OAuthAccountRequest } from "./oauth-account-dialog"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

const zendeskInput = {
  label: "Zendesk subdomain",
  placeholder: "acme",
  help: "The part before .zendesk.com in your Zendesk address.",
  suffix: ".zendesk.com",
}

function Harness({
  onSubmit,
  requestKey = "zendesk",
  retry,
}: {
  onSubmit: (account: string) => void
  requestKey?: string
  retry?: boolean
}) {
  const { requestAccount, accountDialog } = useOAuthAccountStep()
  const request: OAuthAccountRequest = {
    key: requestKey,
    name: "Zendesk",
    input: zendeskInput,
    onSubmit,
  }
  return (
    <>
      <button type="button" onClick={() => requestAccount(request, { retry })}>
        start
      </button>
      {accountDialog}
    </>
  )
}

function open() {
  fireEvent.click(screen.getByRole("button", { name: "start" }))
  return screen.getByRole("textbox")
}

function submit() {
  fireEvent.click(screen.getByRole("button", { name: "tools.mcp.accountStep.continue" }))
}

describe("useOAuthAccountStep", () => {
  afterEach(() => {
    cleanup()
  })

  it("renders nothing until an account is requested", () => {
    render(<Harness onSubmit={vi.fn()} />)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("shows the backend-provided label, placeholder, help, and host suffix", () => {
    render(<Harness onSubmit={vi.fn()} />)
    const input = open()

    expect(screen.getByLabelText("Zendesk subdomain")).toBe(input)
    expect(input).toHaveAttribute("placeholder", "acme")
    expect(screen.getByText(zendeskInput.help)).toBeInTheDocument()
    expect(screen.getByText(".zendesk.com")).toBeInTheDocument()
  })

  it("blocks an invalid label without calling onSubmit", () => {
    const onSubmit = vi.fn()
    render(<Harness onSubmit={onSubmit} />)
    const input = open()

    fireEvent.change(input, { target: { value: "not a label" } })
    submit()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toHaveTextContent("tools.mcp.accountStep.invalid")
    expect(input).toHaveAttribute("aria-invalid", "true")
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("blocks an empty submission", () => {
    const onSubmit = vi.fn()
    render(<Harness onSubmit={onSubmit} />)
    open()

    submit()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("clears the format error once the user edits the value", () => {
    render(<Harness onSubmit={vi.fn()} />)
    const input = open()
    submit()
    expect(screen.getByRole("alert")).toBeInTheDocument()

    fireEvent.change(input, { target: { value: "acme" } })

    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("submits the normalized label synchronously and closes", () => {
    const onSubmit = vi.fn()
    render(<Harness onSubmit={onSubmit} />)
    const input = open()

    fireEvent.change(input, { target: { value: "https://Acme.zendesk.com/agent" } })
    submit()

    // Synchronous: a popup opened from onSubmit must still run inside the
    // click's user activation, or the browser blocks it.
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith("acme")
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("submits from the keyboard through the form", () => {
    const onSubmit = vi.fn()
    render(<Harness onSubmit={onSubmit} />)
    const input = open()

    fireEvent.change(input, { target: { value: "acme" } })
    fireEvent.submit(input.closest("form") as HTMLFormElement)

    expect(onSubmit).toHaveBeenCalledWith("acme")
  })

  it("cancels without calling onSubmit", () => {
    const onSubmit = vi.fn()
    render(<Harness onSubmit={onSubmit} />)
    const input = open()

    fireEvent.change(input, { target: { value: "acme" } })
    fireEvent.click(screen.getByRole("button", { name: "common.cancel" }))

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("starts empty again after a cancel", () => {
    render(<Harness onSubmit={vi.fn()} />)
    const input = open()
    fireEvent.change(input, { target: { value: "acme" } })
    fireEvent.click(screen.getByRole("button", { name: "common.cancel" }))

    expect(open()).toHaveValue("")
  })

  it("pre-fills the last submitted label and shows the retry hint on a retry", () => {
    const onSubmit = vi.fn()
    const { rerender } = render(<Harness onSubmit={onSubmit} />)
    const input = open()
    fireEvent.change(input, { target: { value: "acme" } })
    submit()
    expect(screen.queryByText("tools.mcp.accountStep.retryHint")).not.toBeInTheDocument()

    rerender(<Harness onSubmit={onSubmit} retry />)

    expect(open()).toHaveValue("acme")
    expect(screen.getByText("tools.mcp.accountStep.retryHint")).toBeInTheDocument()
  })

  it("does not carry a label over to a different connection", () => {
    const onSubmit = vi.fn()
    const { rerender } = render(<Harness onSubmit={onSubmit} />)
    fireEvent.change(open(), { target: { value: "acme" } })
    submit()

    rerender(<Harness onSubmit={onSubmit} requestKey="other" retry />)

    expect(open()).toHaveValue("")
  })
})
