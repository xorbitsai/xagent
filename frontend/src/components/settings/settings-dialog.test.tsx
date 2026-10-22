import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const settingsPageProps = vi.hoisted(() => vi.fn())

vi.mock("@/components/settings/settings-content", () => ({
  SettingsContent: () => {
    settingsPageProps()
    return <div data-testid="settings-content" />
  },
}))
vi.mock("@/lib/task-runtime-ui-extension", () => ({
  TaskRuntimeSettingsExtension: () => <div data-testid="computer-use-content" />,
}))
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { SettingsDialog } from "./settings-dialog"

describe("SettingsDialog", () => {
  afterEach(() => {
    settingsPageProps.mockReset()
    cleanup()
  })

  it("shows only General and Computer use and switches embedded content", () => {
    render(<SettingsDialog open onOpenChange={vi.fn()} />)

    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveClass(
      "w-[min(1120px,calc(100vw-48px))]",
      "sm:max-w-[1120px]",
    )
    expect(dialog).not.toHaveClass("sm:max-w-lg")
    const navigation = screen.getByRole("navigation", { name: "settings.title" })
    expect(navigation.querySelectorAll("button")).toHaveLength(2)
    expect(screen.getByRole("button", { name: "settings.tabs.general" })).toHaveAttribute("aria-current", "page")
    expect(settingsPageProps).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("settings-content").parentElement).toHaveClass(
      "[&_[data-slot=tabs]]:!mt-0",
      "[&_[role=tabpanel]]:!mt-0",
    )

    fireEvent.click(screen.getByRole("button", { name: "settings.tabs.computerAccess" }))
    expect(screen.getByRole("button", { name: "settings.tabs.computerAccess" })).toHaveAttribute("aria-current", "page")
    expect(screen.getByTestId("computer-use-content")).toBeInTheDocument()
  })

  it("opens a deep link on Computer use", () => {
    render(<SettingsDialog open onOpenChange={vi.fn()} initialTab="computer" targetId="desktop-relay" />)

    expect(screen.getByRole("button", { name: "settings.tabs.computerAccess" })).toHaveAttribute("aria-current", "page")
    expect(screen.getByTestId("computer-use-content")).toBeInTheDocument()
    expect(settingsPageProps).not.toHaveBeenCalled()
  })
})
