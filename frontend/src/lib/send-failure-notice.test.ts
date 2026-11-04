import { describe, expect, it, vi } from "vitest"
import type { Translate } from "@/contexts/i18n-context"

const toastMock = vi.hoisted(() => ({ error: vi.fn() }))
vi.mock("@/components/ui/sonner", () => ({ toast: toastMock }))

import { notifySendFailure } from "./send-failure-notice"

const t: Translate = (key) => key
const untranslated: Translate = () => ""

type Row = { name: string; error: unknown; suppress: boolean; translate?: Translate; toasts: unknown[][] }

const GENERIC = [["builds.list.chat.sendFailed"]]

const ROWS: Row[] = [
  { name: "shows nothing for a failure another surface already notified, even with an error code",
    error: Object.assign(new Error("x"), { notificationHandled: true, errorCode: "task_busy" }), suppress: false, toasts: [] },
  { name: "shows nothing for a lost connection the caller suppresses",
    error: Object.assign(new Error("x"), { connectionLost: true }), suppress: true, toasts: [] },
  { name: "shows the generic text for a lost connection the caller does not suppress",
    error: Object.assign(new Error("x"), { connectionLost: true }), suppress: false, toasts: GENERIC },
  { name: "still shows other failures while lost connections are suppressed",
    error: new Error("x"), suppress: true, toasts: GENERIC },
  { name: "prefers the error code's text to the server's text",
    error: Object.assign(new Error("provider token=secret"), { errorCode: "task_busy", userFacing: true }),
    suppress: false, toasts: [["clientErrors.taskBusy"]] },
  { name: "shows the server's text when there is no error code",
    error: Object.assign(new Error("Server said no"), { userFacing: true }), suppress: false, toasts: [["Server said no"]] },
  { name: "treats an unknown error code as no code",
    error: Object.assign(new Error("Server said no"), { errorCode: "not_a_code", userFacing: true }),
    suppress: false, toasts: [["Server said no"]] },
  { name: "shows the generic text for a userFacing value that is not an Error",
    error: { userFacing: true, message: "Server said no" }, suppress: false, toasts: GENERIC },
  { name: "shows the generic text, not the raw message, for an Error not marked userFacing",
    error: new Error("raw detail"), suppress: false, toasts: GENERIC },
  { name: "shows the generic text for a rejection with null", error: null, suppress: false, toasts: GENERIC },
  { name: "shows the generic text for a rejection with no value", error: undefined, suppress: false, toasts: GENERIC },
  { name: "falls back to English when the generic text has no translation",
    error: new Error("x"), suppress: false, translate: untranslated, toasts: [["Message was not sent"]] },
]

describe("notifySendFailure", () => {
  it.each(ROWS)("$name", ({ error, suppress, translate, toasts }) => {
    notifySendFailure(error, translate ?? t, { suppressConnectionLost: suppress })

    expect(toastMock.error.mock.calls).toEqual(toasts)
  })
})
