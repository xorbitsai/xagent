/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastSuccessMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) =>
    values.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { success: toastSuccessMock, error: toastErrorMock },
}))

vi.mock("@/contexts/i18n-context", async () => {
  const { resolveDynamicTranslation } = await import("@/i18n/translations")
  return {
    useI18n: () => ({
      t: (key: string) => resolveDynamicTranslation("en", key, key),
    }),
  }
})

import { translations } from "@/i18n/translations"
import {
  getAgentWidgetVoice,
  setAgentWidgetVoice,
  type WidgetVoiceSetting as VoiceSettingValue,
} from "@/lib/agent-widget-config"
import { WidgetVoiceSetting } from "./widget-voice-setting"

const copy = translations.en.widget_voice_setting

const OFF: VoiceSettingValue = { enabled: false, available: null, reason: null }
const ON_AVAILABLE: VoiceSettingValue = { enabled: true, available: true, reason: null }
const ON_NOT_CONFIGURED: VoiceSettingValue = {
  enabled: true,
  available: false,
  reason: "voice_models_not_configured",
}
const ON_UNAVAILABLE: VoiceSettingValue = {
  enabled: true,
  available: false,
  reason: "voice_models_unavailable",
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function getSwitch() {
  return screen.getByRole("switch", { name: copy.toggle_label })
}

describe("WidgetVoiceSetting", () => {
  beforeEach(() => {
    toastSuccessMock.mockReset()
    toastErrorMock.mockReset()
    vi.spyOn(console, "error").mockImplementation(() => {})
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it.each([
    ["off", OFF, null],
    ["on and available", ON_AVAILABLE, null],
    ["on, models not configured", ON_NOT_CONFIGURED, copy.unavailable_not_configured],
    ["on, service unavailable", ON_UNAVAILABLE, copy.unavailable_models],
  ])("shows loading then the stored state: %s", async (_name, stored, warning) => {
    const pending = deferred<VoiceSettingValue>()
    render(<WidgetVoiceSetting load={() => pending.promise} save={vi.fn()} />)

    expect(getSwitch()).toBeDisabled()
    expect(screen.getByText(copy.loading)).toBeInTheDocument()

    await act(async () => {
      pending.resolve(stored)
    })

    expect(screen.queryByText(copy.loading)).not.toBeInTheDocument()
    expect(getSwitch()).toBeEnabled()
    expect(getSwitch()).toHaveAttribute("aria-checked", String(stored.enabled))
    for (const message of [
      copy.unavailable_not_configured,
      copy.unavailable_models,
    ]) {
      if (message === warning) {
        expect(screen.getByText(message)).toBeInTheDocument()
      } else {
        expect(screen.queryByText(message)).not.toBeInTheDocument()
      }
    }
  })

  it("keeps the switch disabled when loading fails", async () => {
    const load = vi
      .fn<() => Promise<VoiceSettingValue>>()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(ON_AVAILABLE)
    render(<WidgetVoiceSetting load={load} save={vi.fn()} />)

    expect(await screen.findByText(copy.load_failed)).toBeInTheDocument()
    expect(getSwitch()).toBeDisabled()
    expect(getSwitch()).toHaveAttribute("aria-checked", "false")
    expect(screen.queryByText(copy.loading)).not.toBeInTheDocument()
    expect(load).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: copy.retry }))

    await waitFor(() => expect(getSwitch()).toBeEnabled())
    expect(load).toHaveBeenCalledTimes(2)
    expect(getSwitch()).toHaveAttribute("aria-checked", "true")
    expect(screen.queryByText(copy.load_failed)).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: copy.retry })).not.toBeInTheDocument()
  })

  it.each([
    ["off to on, server agrees", OFF, true, ON_AVAILABLE, "true"],
    ["on to off, server agrees", ON_AVAILABLE, false, OFF, "false"],
    ["off to on, server says off (concurrent change)", OFF, true, OFF, "false"],
    ["on to off, server says on (concurrent change)", ON_AVAILABLE, false, ON_AVAILABLE, "true"],
  ])("saves and follows the server response: %s", async (_name, stored, clicked, response, finalChecked) => {
    const pending = deferred<VoiceSettingValue>()
    const save = vi.fn().mockReturnValue(pending.promise)
    render(<WidgetVoiceSetting load={() => Promise.resolve(stored)} save={save} />)
    await waitFor(() => expect(getSwitch()).toBeEnabled())

    fireEvent.click(getSwitch())

    expect(save).toHaveBeenCalledWith(clicked)
    expect(getSwitch()).toBeDisabled()

    await act(async () => {
      pending.resolve(response)
    })

    expect(getSwitch()).toBeEnabled()
    expect(getSwitch()).toHaveAttribute("aria-checked", finalChecked)
    expect(toastSuccessMock).toHaveBeenCalledWith(copy.saved)
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    ["403", Object.assign(new Error("Access denied"), { status: 403 })],
    ["404", Object.assign(new Error("Agent not found"), { status: 404 })],
    ["500", Object.assign(new Error("Internal Server Error"), { status: 500 })],
    ["network error", new TypeError("Failed to fetch")],
  ])("rolls back when saving fails: %s", async (_name, failure) => {
    const save = vi.fn().mockRejectedValue(failure)
    render(<WidgetVoiceSetting load={() => Promise.resolve(OFF)} save={save} />)
    await waitFor(() => expect(getSwitch()).toBeEnabled())

    fireEvent.click(getSwitch())

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(copy.save_failed))
    expect(getSwitch()).toHaveAttribute("aria-checked", "false")
    expect(getSwitch()).toBeEnabled()
    expect(toastSuccessMock).not.toHaveBeenCalled()
  })

  it("ignores a failed read that finishes after unmount", async () => {
    const pending = deferred<VoiceSettingValue>()
    const { unmount } = render(
      <WidgetVoiceSetting load={() => pending.promise} save={vi.fn()} />,
    )
    unmount()

    await act(async () => {
      pending.reject(new Error("late failure"))
    })

    expect(console.error).not.toHaveBeenCalled()
  })

  it.each([
    ["succeeds", true],
    ["fails", false],
  ])("shows no notice when a save that %s finishes after unmount", async (_name, succeeds) => {
    const pending = deferred<VoiceSettingValue>()
    const save = vi.fn().mockReturnValue(pending.promise)
    const { unmount } = render(
      <WidgetVoiceSetting load={() => Promise.resolve(OFF)} save={save} />,
    )
    await waitFor(() => expect(getSwitch()).toBeEnabled())
    fireEvent.click(getSwitch())
    unmount()

    await act(async () => {
      if (succeeds) pending.resolve(ON_AVAILABLE)
      else pending.reject(new Error("late failure"))
    })

    expect(toastSuccessMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it.each([
    ["off", OFF],
    ["on and available", ON_AVAILABLE],
    ["on and unavailable", ON_UNAVAILABLE],
  ])("always shows the requirements and refresh notes: %s", async (_name, stored) => {
    render(<WidgetVoiceSetting load={() => Promise.resolve(stored)} save={vi.fn()} />)
    await waitFor(() => expect(getSwitch()).toBeEnabled())

    expect(screen.getByText(copy.requirements_note)).toBeInTheDocument()
    expect(screen.getByText(copy.refresh_note)).toBeInTheDocument()
  })

  it("shows the notes while the read is still pending and after it failed", async () => {
    const load = vi.fn().mockRejectedValue(new Error("boom"))
    render(<WidgetVoiceSetting load={load} save={vi.fn()} />)

    expect(screen.getByText(copy.requirements_note)).toBeInTheDocument()
    expect(screen.getByText(copy.refresh_note)).toBeInTheDocument()
    await screen.findByText(copy.load_failed)
    expect(screen.getByText(copy.requirements_note)).toBeInTheDocument()
    expect(screen.getByText(copy.refresh_note)).toBeInTheDocument()
  })
})

describe("agent widget voice requests", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("reads the setting with GET on the agent's widget-voice path", async () => {
    apiRequestMock.mockResolvedValueOnce(
      new Response(JSON.stringify(ON_AVAILABLE), { status: 200 }),
    )

    await expect(getAgentWidgetVoice(7)).resolves.toEqual(ON_AVAILABLE)

    expect(apiRequestMock).toHaveBeenCalledTimes(1)
    expect(apiRequestMock.mock.calls[0][0]).toBe(
      "http://api.local/api/agents/7/widget-voice",
    )
    expect(apiRequestMock.mock.calls[0][1]?.method ?? "GET").toBe("GET")
  })

  it.each([true, false])("writes %s with PUT and an enabled-only body", async (enabled) => {
    apiRequestMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ ...OFF, enabled }), { status: 200 }),
    )

    await expect(setAgentWidgetVoice("7", enabled)).resolves.toMatchObject({ enabled })

    expect(apiRequestMock).toHaveBeenCalledTimes(1)
    const [url, init] = apiRequestMock.mock.calls[0]
    expect(url).toBe("http://api.local/api/agents/7/widget-voice")
    expect(init.method).toBe("PUT")
    expect(JSON.parse(init.body)).toEqual({ enabled })
  })

  it("rejects with the server's message when reading fails", async () => {
    apiRequestMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Agent not found" }), { status: 404 }),
    )

    await expect(getAgentWidgetVoice(7)).rejects.toThrow("Agent not found")
  })

  it("rejects when writing fails", async () => {
    apiRequestMock.mockResolvedValueOnce(new Response("oops", { status: 500 }))

    await expect(setAgentWidgetVoice(7, true)).rejects.toThrow(
      "Failed to update widget voice setting",
    )
  })
})
