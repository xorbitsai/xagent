import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  READ_FAILURE_STATUSES,
  type ConnectorRuntimeConnector,
  type ConnectorRuntimeInput,
  type ConnectorRuntimeReport,
  type ConnectorRuntimeSection,
  type ConnectorRuntimeType,
} from "@/lib/connector-runtime-api"

const pathnameRef = vi.hoisted(() => ({ current: "/task/1" as string | null }))
const appStateRef = vi.hoisted(() => ({ taskId: 1 as number | null }))
const sendMessageMock = vi.hoisted(() =>
  vi.fn<(message: string, config?: { clientMessageId?: string; force?: boolean }, files?: File[]) => Promise<void>>(
    async () => {},
  ),
)
const authUserRef = vi.hoisted(() => ({ current: { id: "u1" } as { id: string } | null }))
const fetchMock = vi.hoisted(() => vi.fn())
const submitMock = vi.hoisted(() => vi.fn())
const toastMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({ usePathname: () => pathnameRef.current }))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ state: { taskId: appStateRef.taskId }, sendMessage: sendMessageMock }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserRef.current }),
}))

// Wrapped in a stable function so a spy installed on `toastMock` after the
// first render still intercepts every call, regardless of when the dialog
// component's own module-level import binding was evaluated.
vi.mock("@/components/ui/sonner", () => ({ toast: (...args: unknown[]) => toastMock(...args) }))

// A referentially stable value: the xagent frontend test convention this
// mirrors (i18n stubs must not return a new object every render) exists
// because a fresh object here would make any effect depending on `t`
// re-fire every render.
const i18nValue = {
  t: (key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}:${JSON.stringify(vars)}` : key,
}
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => i18nValue }))

vi.mock("@/lib/connector-runtime-api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/connector-runtime-api")>()
  return {
    ...actual,
    fetchTaskConnectorRuntimeRequirements: (...args: unknown[]) => fetchMock(...args),
    submitTaskConnectorRuntimeValues: (...args: unknown[]) => submitMock(...args),
  }
})

import { ConnectorRuntimeDialog } from "./connector-runtime-dialog"
import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogActions,
} from "@/contexts/connector-runtime-dialog-context"

const REF_A = { connector_type: "custom_api", connector_id: 1 }
const REF_B = { connector_type: "mcp", connector_id: 2 }

// Shared by the source-scanning test below instead of it re-reading twice.
const libSource = readFileSync(path.resolve(__dirname, "../../lib/connector-runtime-api.ts"), "utf8")
const dialogSource = readFileSync(path.resolve(__dirname, "./connector-runtime-dialog.tsx"), "utf8")

function input(
  overrides: Partial<ConnectorRuntimeInput> & { section: ConnectorRuntimeSection; key: string; type: ConnectorRuntimeType },
): ConnectorRuntimeInput {
  return { required: false, satisfied: false, expired: false, ...overrides }
}
function connector(ref: typeof REF_A, name: string, inputs: ConnectorRuntimeInput[]): ConnectorRuntimeConnector {
  return { connector_ref: ref, name, inputs }
}
function report(satisfied: boolean, connectors: ConnectorRuntimeConnector[]): ConnectorRuntimeReport {
  return { satisfied, secrets_expires_at: null, connectors }
}
function ok(r: ConnectorRuntimeReport) { return { ok: true as const, report: r } }

let latestActions: ConnectorRuntimeDialogActions
let latestState: { request: unknown; payload: unknown }

function Probe({ mounted = true }: { mounted?: boolean }) {
  latestActions = useConnectorRuntimeDialogActions()
  const { request, payload } = useConnectorRuntimeDialog()
  latestState = { request, payload }
  return mounted ? <ConnectorRuntimeDialog /> : null
}

function renderHarness() {
  return render(
    <ConnectorRuntimeDialogProvider>
      <Probe />
    </ConnectorRuntimeDialogProvider>,
  )
}

async function openForTask(taskId = 1) {
  await act(async () => {
    latestActions.openForTask(taskId)
  })
}

// Delivery and the later open are two separate updates in production (the
// delivered-turn stash is written long before any failure frame retargets
// it), so this mirrors that with two acts rather than one -- batching both
// into a single act would not exercise the same "read the already-committed
// stash" path openForTask's handoff depends on.
//
// recordDelivery only writes the stash for a clientMessageId it has a
// staged ticket for (production always stages before it ever records, in
// sendMessage), so every direct recordDelivery call in this file goes
// through stageThenRecord below instead of calling it bare.
async function stageThenRecord(
  delivery: { taskId: number; clientMessageId: string; text: string; files?: File[] },
) {
  await act(async () => {
    latestActions.stagePendingDelivery(delivery)
  })
  await act(async () => {
    latestActions.recordDelivery(delivery)
  })
}

async function recordThenOpen(
  delivery: { taskId: number; clientMessageId: string; text: string; files?: File[] },
  taskId = delivery.taskId,
) {
  await stageThenRecord(delivery)
  await act(async () => {
    latestActions.openForTask(taskId)
  })
}

beforeEach(() => {
  pathnameRef.current = "/task/1"
  appStateRef.taskId = 1
  authUserRef.current = { id: "u1" }
  sendMessageMock.mockReset()
  sendMessageMock.mockResolvedValue(undefined)
  fetchMock.mockReset()
  // A harmless default for any refresh this test does not explicitly stub
  // (a failed save whose disposition sets `refresh: true`): resolving to a
  // read failure just means the dialog keeps showing the report it already
  // had, and avoids an unhandled rejection from an un-mocked call.
  fetchMock.mockResolvedValue({ ok: false, kind: "malformed" })
  submitMock.mockReset()
  toastMock.mockReset()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("keeps read outcomes in three distinct tiers", () => {
  it("keeps read outcomes in three distinct tiers", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    // The full read-failure warn matrix: every status READ_FAILURE_STATUSES
    // lists, 418 standing in for "any other non-200", and the two non-HTTP
    // failure kinds. `.every` over the spy's calls would pass on an empty
    // array, so this pins down the exact call list instead.
    const expectedWarnCalls: unknown[][] = []

    for (const status of READ_FAILURE_STATUSES) {
      fetchMock.mockReset()
      fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status })
      renderHarness()
      await openForTask()
      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expectedWarnCalls.push(["[connector-runtime] requirements read failed", status])
      cleanup()
    }
    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 418 })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", 418])
    cleanup()

    fetchMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", "transport"])
    cleanup()

    fetchMock.mockResolvedValueOnce({ ok: false, kind: "malformed" })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expectedWarnCalls.push(["[connector-runtime] requirements read failed", "malformed"])
    cleanup()

    expect(warnSpy.mock.calls).toEqual(expectedWarnCalls)

    // Met: nothing to fill at all, and met with an unfilled optional key.
    fetchMock.mockResolvedValueOnce(ok(report(true, [])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    fetchMock.mockResolvedValueOnce(ok(report(true, [
      connector(REF_A, "A", [input({ section: "context", key: "optional", type: "string" })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    // Needs a fill: the dialog becomes visible.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
  })
})

describe("renders each row by section, type, satisfaction and outcome", () => {
  it("renders each row by section, type, satisfaction and outcome", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "textKey", type: "string", required: true }),
        input({ section: "context", key: "objectKey", type: "object", required: true }),
        input({ section: "context", key: "doneKey", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "secretKey", type: "string", required: false }),
        input({ section: "auth_selector", key: "authKey", type: "string", required: false }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    expect(screen.getByLabelText("textKey")).toBeInstanceOf(HTMLInputElement)
    expect(screen.getByLabelText("objectKey")).toBeInstanceOf(HTMLTextAreaElement)
    expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument()
    expect(screen.getByText("secretKey")).toBeInTheDocument()
    expect(screen.getByText("authKey")).toBeInTheDocument()
    expect(screen.getAllByText("connectorRuntime.unsupportedNote")).toHaveLength(2)
    expect(screen.queryByRole("textbox", { name: "secretKey" })).not.toBeInTheDocument()
    expect(screen.queryByRole("textbox", { name: "authKey" })).not.toBeInTheDocument()
    expect(document.querySelector('input[type="password"]')).not.toBeInTheDocument()
    cleanup()

    // unsupported_only: an unfilled context row (bad key name included)
    // renders no control and no "saved, cannot be changed" hint, but the
    // key-name warning still shows.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "bad key", type: "string", required: false }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.contextNote")).not.toBeInTheDocument()
    expect(screen.getByText("connectorRuntime.keyNameWarning")).toBeInTheDocument()
  })
})

describe("keeps drafts and the snapshot when re-requested while open", () => {
  it("keeps drafts and the snapshot when re-requested while open", async () => {
    // A second request for an already-open task keeps a draft for a key
    // still unsatisfied, but drops one the refreshed report now reports
    // satisfied.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "stillMissing", type: "string", required: true }),
        input({ section: "context", key: "getsFilled", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("stillMissing"), { target: { value: "draft-a" } })
    fireEvent.change(screen.getByLabelText("getsFilled"), { target: { value: "draft-b" } })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "stillMissing", type: "string", required: true }),
        input({ section: "context", key: "getsFilled", type: "string", required: true, satisfied: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    expect(screen.getByLabelText("stillMissing")).toHaveValue("draft-a")
    expect(screen.queryByLabelText("getsFilled")).not.toBeInTheDocument()
    cleanup()
    fetchMock.mockClear()

    // A read started earlier must not overwrite a fresher read's result if
    // it resolves later (dropped via the request's own seq).
    let resolveFirst: (v: unknown) => void = () => {}
    let resolveSecond: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFirst = res }))
    renderHarness()
    await openForTask()
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveSecond = res }))
    await openForTask()
    await act(async () => { resolveSecond(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "fromSecondRequest", type: "string", required: true })]),
    ]))) })
    await waitFor(() => expect(screen.getByLabelText("fromSecondRequest")).toBeInTheDocument())
    await act(async () => { resolveFirst(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "fromStaleFirstRequest", type: "string", required: true })]),
    ]))) })
    expect(screen.queryByLabelText("fromStaleFirstRequest")).not.toBeInTheDocument()
    expect(screen.getByLabelText("fromSecondRequest")).toBeInTheDocument()
    cleanup()
    fetchMock.mockClear()

    // A same-task re-request with no fresh stash in between keeps the resend
    // snapshot the first request already carried.
    const tokenReport = ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ]))
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-4", text: "keep me" })
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.saveAndResend")).toBeInTheDocument())
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // second request, no recordDelivery first
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][0]).toBe("keep me")
    cleanup()
    fetchMock.mockClear()

    // A save in flight when the task is re-requested must not leave the
    // dialog stuck once the stale save settles.
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    let resolveSave: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSave = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // retargets this same dialog instance mid-save
    await act(async () => { resolveSave(ok(report(true, []))) })
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    cleanup()
    fetchMock.mockClear()

    // The save half of a save-and-resend settles, then the resend it
    // triggers is still in flight when the task is re-requested: submitting
    // must still reset once that stale resend settles, or the dialog stays
    // stuck exactly like the save-only case above.
    fetchMock.mockResolvedValueOnce(tokenReport)
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-5", text: "keep me" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    let resolveSend: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveSend = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    fetchMock.mockResolvedValueOnce(tokenReport)
    await openForTask() // retargets this same dialog instance mid-resend
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await act(async () => { resolveSend() })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })

  it("does not let an invalid-object mark on a row the report now hides keep submission disabled", async () => {
    // A key the refreshed report reports satisfied loses its editable row,
    // so an invalid-object mark recorded against it while it was still
    // editable must not gate submission forever.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "object", required: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    const objField = screen.getByLabelText("objKey")
    fireEvent.change(objField, { target: { value: "{oops" } })
    fireEvent.blur(objField)
    expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()

    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "object", required: true, satisfied: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText("strKey"), { target: { value: "value" } })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
  })
})

describe("couples the invalid-object error to the submit gate", () => {
  it("keeps the save button disabled while a live invalid-object mark's error is shown", async () => {
    // Kills a predicate that always evaluates to false: with the error on
    // screen, the button must actually be unusable, not merely styled as if
    // it were.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "object", required: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText("strKey"), { target: { value: "value" } })
    const objField = screen.getByLabelText("objKey")
    fireEvent.change(objField, { target: { value: "{oops" } })
    fireEvent.blur(objField)
    expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()

    const saveButton = screen.getByText("connectorRuntime.actions.saveOnly")
    expect(saveButton).toBeDisabled()
    fireEvent.click(saveButton)
    expect(submitMock).not.toHaveBeenCalled()
  })

  it("closes the retry button on a draft turned invalid after a failed save", async () => {
    // The retry button a retryable failure offers is a third submit entry
    // point, and it used to be rendered off the failure alone. This sequence
    // is the one that made that matter: a save fails, the user edits an
    // object field into something unparsable, the save buttons go disabled,
    // and the retry button next to them stays usable. buildSubmitItems drops
    // an unparsable object draft rather than failing, so the retried batch
    // would have written the string field, silently lost the object field,
    // and closed -- with a stored context value immutable afterwards.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "object", required: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    const objField = screen.getByLabelText("objKey")
    fireEvent.change(objField, { target: { value: "{\"a\": 1}" } })
    fireEvent.blur(objField)
    fireEvent.change(screen.getByLabelText("strKey"), { target: { value: "value" } })

    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.retry")).toBeInTheDocument())
    expect(submitMock).toHaveBeenCalledTimes(1)

    fireEvent.change(objField, { target: { value: "{oops" } })
    fireEvent.blur(objField)
    expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()

    const retryButton = screen.getByText("connectorRuntime.actions.retry")
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeDisabled()
    expect(retryButton).toBeDisabled()
    fireEvent.click(retryButton)
    expect(submitMock).toHaveBeenCalledTimes(1)

    // And the other direction: repairing the draft restores all three.
    fireEvent.change(objField, { target: { value: "{\"a\": 2}" } })
    fireEvent.blur(objField)
    expect(retryButton).toBeEnabled()
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
  })

  it("drops a stale invalid-object error and mark once the key's declared type is no longer object", async () => {
    // Kills both dropping the `input.type === "object"` clause from the
    // shared predicate, and reverting the row's error paragraph back to
    // reading the raw `invalidDraftKeys` set directly.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "object", required: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    const objField = screen.getByLabelText("objKey")
    fireEvent.change(objField, { target: { value: "{oops" } })
    fireEvent.blur(objField)
    expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()

    // Same task, same key, but the connector now declares it a plain string
    // -- the shape the server itself can flip a key's declaration to between
    // two reads.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "objKey", type: "string", required: true }),
        input({ section: "context", key: "strKey", type: "string", required: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount
    await waitFor(() => expect(screen.getByLabelText("objKey").tagName).toBe("INPUT"))

    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
    // The draft is keyed by declared type along with connector and key name,
    // so the "{oops" text recorded while this row was object-typed lives
    // under a different key than the one this now-string row reads: it does
    // not resurface as a value the user never actually typed against the new
    // type.
    expect(screen.getByLabelText("objKey")).toHaveValue("")

    fireEvent.change(screen.getByLabelText("strKey"), { target: { value: "value" } })
    const saveButton = screen.getByText("connectorRuntime.actions.saveOnly")
    expect(saveButton).toBeEnabled()

    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(saveButton)
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(1))
    const items = submitMock.mock.calls[0][1] as Array<{ context: Record<string, unknown> }>
    expect(items).toHaveLength(1)
    // `objKey`'s draft never carried over (see above), so buildSubmitItems
    // has nothing recorded for it and only submits the field the user
    // actually filled in under the new report.
    expect(items[0].context).toEqual({ strKey: "value" })
  })

  it("drops a stale non-object draft once the key's declared type becomes object", async () => {
    // The reverse of the object -> string case above: a plain string typed
    // against a string-typed key must not resurface as that key's value once
    // the connector owner redeclares it object-typed -- a stray string is
    // not valid JSON either, and reusing it would either submit garbage or
    // (worse) parse by coincidence into something the user never intended.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: "plain text" } })

    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    await openForTask() // same task: a second request, not a remount
    await waitFor(() => expect(screen.getByLabelText("cfg").tagName).toBe("TEXTAREA"))

    expect(screen.getByLabelText("cfg")).toHaveValue("")
    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
    // Nothing submittable: the only context key has no draft under its new
    // type, so there is nothing for the save button to send.
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeDisabled()
  })
})

describe("re-reads the report on a type mismatch so a changed declaration takes over immediately", () => {
  it("clears the type-specific hint once a refresh shows the row's type changed from object to string", async () => {
    // The connector owner can flip a key's declared type between the report
    // this dialog read and the write it submits against; the server 400s
    // with type_mismatch. Refreshing here (rather than leaving the stale
    // report in place) means the row picks up the new type -- and because
    // the disposition's messageKey names a specific type ("typeObject"),
    // that hint no longer describes this row once the refresh shows it now
    // declares "string", so it must be cleared rather than reattached.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: '{"a":1}' } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.cfg", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "string", required: true })]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByLabelText("cfg").tagName).toBe("INPUT"))

    expect(screen.getByLabelText("cfg")).toHaveValue("")
    expect(screen.queryByText(/connectorRuntime\.errors\.typeObject/)).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: "plain text" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2))
    expect(submitMock.mock.calls[1][1]).toEqual([{ connector_ref: REF_A, context: { cfg: "plain text" } }])
  })

  it("clears the type-specific hint once a refresh shows the row's type changed from string to object", async () => {
    // Same defect, opposite direction: a "typeString" hint must not survive
    // a refresh that shows the row now declares "object".
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: "hello" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.cfg", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByLabelText("cfg").tagName).toBe("TEXTAREA"))

    expect(screen.queryByText(/connectorRuntime\.errors\.typeString/)).not.toBeInTheDocument()
  })

  it("keeps the type-specific hint when a refresh shows the row's declared type is unchanged", async () => {
    // Same failure, but the refresh reads back the same declared type: the
    // hint still describes the row, so it must stay rather than being
    // cleared on every refresh regardless of what changed.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: '{"a":1}' } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.cfg", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    expect(screen.getByText("connectorRuntime.errors.typeObject:{\"key\":\"cfg\"}")).toBeInTheDocument()
  })

  it("falls back to the whole dialog when a refresh's report drops the row the error was on", async () => {
    // Same failure as above, but the refresh reports a declaration that no
    // longer has this key at all (not just a different type): locateFieldError
    // cannot find any row to attach to, and must fall back to dialog scope
    // rather than rendering the rejection nowhere.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "cfg", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("cfg"), { target: { value: '{"a":1}' } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.cfg", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "other", type: "string", required: true })]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByLabelText("other")).toBeInTheDocument())

    // Whole-dialog scope renders without the {key} interpolation.
    expect(screen.getByText("connectorRuntime.errors.typeObject")).toBeInTheDocument()
    expect(screen.queryByText(/connectorRuntime\.errors\.typeObject:/)).not.toBeInTheDocument()
  })
})

describe("keeps distinct row identity for a key name legitimately reused across sections", () => {
  it("keeps distinct row identity for a key name legitimately reused across sections", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "shared", type: "string", required: true }),
        input({ section: "secrets", key: "shared", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    const duplicateKeyWarning = errorSpy.mock.calls.some(call =>
      typeof call[0] === "string" && call[0].includes("Encountered two children with the same key"),
    )
    expect(duplicateKeyWarning).toBe(false)
  })
})

describe("flags a non-object JSON draft on blur and does not submit it", () => {
  it("flags a non-object JSON draft on blur and does not submit it", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "config", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    const field = screen.getByLabelText("config")

    for (const badValue of ["[]", "1", '"x"', "{"]) {
      fireEvent.change(field, { target: { value: badValue } })
      fireEvent.blur(field)
      expect(screen.getByText("connectorRuntime.objectInvalid")).toBeInTheDocument()
      expect(submitMock).not.toHaveBeenCalled()
    }

    fireEvent.change(field, { target: { value: '{"a":1}' } })
    fireEvent.blur(field)
    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
  })
})

describe("flags an empty JSON object on blur as invalid and disables submit with a reason", () => {
  it("flags an empty JSON object on blur as invalid and disables submit with a reason", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "config", type: "object", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    const field = screen.getByLabelText("config")

    // `{}` is valid JSON and an object, but buildSubmitItems drops it as the
    // object-draft equivalent of a blank string, so this must mark the row
    // invalid too -- with a message distinct from "not valid JSON" -- rather
    // than leaving the row looking fine while the save button silently stays
    // disabled with no explanation on screen.
    fireEvent.change(field, { target: { value: "{}" } })
    fireEvent.blur(field)
    expect(screen.getByText("connectorRuntime.objectEmpty")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.objectInvalid")).not.toBeInTheDocument()
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeDisabled()

    fireEvent.change(field, { target: { value: '{"a":1}' } })
    fireEvent.blur(field)
    expect(screen.queryByText("connectorRuntime.objectEmpty")).not.toBeInTheDocument()
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).not.toBeDisabled()
  })
})

describe("refreshes after a conflict and drops newly satisfied keys", () => {
  it("refreshes after a conflict and drops newly satisfied keys", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true }),
        input({ section: "context", key: "b", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText("a"), { target: { value: "1" } })
    fireEvent.change(screen.getByLabelText("b"), { target: { value: "2" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
      reason: "conflict.context.a", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "b", type: "string", required: true }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())

    // The row the conflict was rejected on just collapsed into "already
    // filled" and no longer renders a field-level error slot at all; the
    // rejection must still be visible, re-located onto the whole dialog
    // instead of disappearing along with the row. At this scope there is no
    // row identity left to name, so it renders the placeholder-free variant
    // of the conflict message, not the keyed one the field-level scope uses
    // (see "renders a placeholder-free conflict message at dialog scope"
    // below for the real, interpolated text this stands in for).
    expect(screen.getByText("connectorRuntime.errors.conflictNoKey")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.errors.conflict")).not.toBeInTheDocument()

    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2))
    expect(submitMock.mock.calls[1][1]).toEqual([{ connector_ref: REF_A, context: { b: "2" } }])
  })
})

describe("renders a placeholder-free conflict message at dialog scope", () => {
  it("renders a placeholder-free conflict message at dialog scope", async () => {
    // Same shape as "refreshes after a conflict and drops newly satisfied
    // keys" above: a 409 conflict whose refresh collapses the named row
    // into "already filled", re-locating the error onto the whole dialog.
    // This file's own `t` stub always returns the key unchanged when no
    // vars are given, so it cannot tell a resolved string that still
    // carries a literal "{key}" apart from one that never had a
    // placeholder to begin with. Swap in the real translation resolver for
    // this one test so a regression here shows up as failing rendered
    // text, not a passing vacuous stub.
    const { resolveTranslation } = await import("@/i18n/translations")
    const originalT = i18nValue.t
    i18nValue.t = (key, vars) => resolveTranslation("en", key as never, vars)
    try {
      fetchMock.mockResolvedValueOnce(ok(report(false, [
        connector(REF_A, "A", [
          input({ section: "context", key: "a", type: "string", required: true }),
          input({ section: "context", key: "b", type: "string", required: true }),
        ]),
      ])))
      renderHarness()
      await openForTask()
      await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

      fireEvent.change(screen.getByLabelText("a"), { target: { value: "1" } })
      fireEvent.change(screen.getByLabelText("b"), { target: { value: "2" } })

      submitMock.mockResolvedValueOnce({
        ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
        reason: "conflict.context.a", connectorRef: REF_A,
      })
      fetchMock.mockResolvedValueOnce(ok(report(false, [
        connector(REF_A, "A", [
          input({ section: "context", key: "a", type: "string", required: true, satisfied: true }),
          input({ section: "context", key: "b", type: "string", required: true }),
        ]),
      ])))
      fireEvent.click(screen.getByText("Save only"))
      await waitFor(() => expect(screen.getByText("Already filled")).toBeInTheDocument())

      const alert = screen.getByRole("alert")
      expect(alert.textContent).not.toContain("{key}")
      expect(alert.textContent).toBe("Another window already filled in this value.")
    } finally {
      i18nValue.t = originalT
    }
  })
})

describe("re-locates an existing field error when an already-visible dialog re-reads a met report", () => {
  it("re-locates an existing field error when an already-visible dialog re-reads a met report", async () => {
    // Uses a disposition with refresh: false (empty_value) so the only
    // report swap in this test comes from the read effect's own same-task
    // re-request below, not from handleSave's own post-failure refresh --
    // isolating the "does an already-visible dialog re-derive a stale field
    // error" question this test is about.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    // A non-empty draft, so the submit gate lets the click through; the
    // server-side rejection this mocks below is what actually classifies as
    // empty_value, not a client-side empty draft (buildSubmitItems already
    // excludes those before a request is even sent).
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "empty_value.context.token", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.emptyValue:{\"key\":\"token\"}")).toBeInTheDocument())

    // A second terminal frame for the same task finds the report now met
    // while the field error above is still active.
    fetchMock.mockResolvedValueOnce(ok(report(true, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount

    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    // The stale field error re-derives against the fresh report instead of
    // continuing to point at a row identity the refresh folded away: it now
    // attaches to the whole dialog rather than rendering nowhere.
    expect(screen.getByText("connectorRuntime.errors.emptyValue")).toBeInTheDocument()
    expect(screen.queryByText(/connectorRuntime\.errors\.emptyValue:/)).not.toBeInTheDocument()
  })
})

describe("leaves a way out when a refresh reports the whole report satisfied", () => {
  it("leaves a way out when a refresh reports the whole report satisfied", async () => {
    // A failed save whose disposition asks for a refresh renders whatever
    // that refresh returns without closing -- a concurrent writer can have
    // satisfied everything between the read and the save. The requirements
    // read effect takes the same path when a same-task re-request finds
    // nothing missing while the dialog is already visible (see the
    // "already-visible" describe block below); the first read and a
    // successful save's optional resend are the two places that do close.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true }),
        input({ section: "context", key: "optional", type: "string" }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getByLabelText("a"), { target: { value: "1" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
      reason: "conflict.context.a", connectorRef: REF_A,
    })
    fetchMock.mockResolvedValueOnce(ok(report(true, [
      connector(REF_A, "A", [
        input({ section: "context", key: "a", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "optional", type: "string" }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())

    // The footer must not be empty: without a button the only way out of
    // this dialog is the window chrome's own close control.
    expect(screen.getByText("connectorRuntime.actions.acknowledge")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveOnly")).not.toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveAndResend")).not.toBeInTheDocument()
    // The still-unfilled optional key must lose its control too: an
    // editable field no button can submit is the same inconsistency read
    // from the other end.
    expect(screen.queryByLabelText("optional")).not.toBeInTheDocument()
    expect(screen.getByText("optional")).toBeInTheDocument()

    fireEvent.click(screen.getByText("connectorRuntime.actions.acknowledge"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    // Acknowledging must not resend the failed turn: the save it followed
    // was rejected, and a resend is a billed model call.
    expect(sendMessageMock).not.toHaveBeenCalled()
  })
})

describe("does not close a dialog the user is already looking at", () => {
  it("keeps it open, keeps the draft, and shows only Got it when a same-task re-request reads met", async () => {
    // A second terminal frame for the same task (a retry from another tab,
    // or the SDK/an external API) bumps request.seq while this dialog is
    // already visible. If the re-read finds nothing missing, the report
    // must be installed and rendered -- not discarded by closing the dialog
    // out from under a user who may still be mid-draft.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "half-typed-secret" } })

    fetchMock.mockResolvedValueOnce(ok(report(true, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
      ]),
    ])))
    await openForTask() // same task: a second request, not a remount

    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.getByText("connectorRuntime.actions.acknowledge")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveOnly")).not.toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveAndResend")).not.toBeInTheDocument()
  })

  it("keeps it open with the current report when a same-task re-request's read fails", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "half-typed-2" } })

    fetchMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    await openForTask() // same task: a second request, not a remount

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.getByLabelText("token")).toHaveValue("half-typed-2")
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeInTheDocument()
  })
})

describe("locates a field error by connector and key", () => {
  it("locates a field error by connector and key", async () => {
    const twoConnectors = report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      connector(REF_B, "B", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    fireEvent.change(screen.getAllByLabelText("token")[1], { target: { value: "y" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.token", connectorRef: REF_B,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.typeString:{\"key\":\"token\"}")).toBeInTheDocument())
    expect(screen.getAllByText(/connectorRuntime.errors.typeString/)).toHaveLength(1)
    cleanup()

    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 409, code: "runtime_context_immutable",
      reason: "conflict.context.token", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getAllByText(/connectorRuntime.errors.conflict/)).toHaveLength(1))
    cleanup()

    // No connector_ref at all: whole-dialog scope.
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 500 })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.contactAdmin")).toBeInTheDocument())
    cleanup()

    // A located key the current report no longer has: falls back to dialog scope.
    fetchMock.mockResolvedValueOnce(ok(twoConnectors))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getAllByLabelText("token")[0], { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.missing", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    // Falls back to whole-dialog scope, which renders without the {key} var.
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.typeString")).toBeInTheDocument())
  })

  it("locates a same-named key's error in the context row when a secrets row with that key comes first in the report", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "secrets", key: "shared", type: "string", required: false }),
        input({ section: "context", key: "shared", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    // The secrets row above renders as plain unsupported-note text, not a
    // labelled control, so this is the context row's own input.
    fireEvent.change(screen.getByLabelText("shared"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "type_mismatch.context.shared", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    // The secrets-section row shares the key but not the section; if the
    // locator matched on key alone it would bind to that row's draft key
    // instead, and this error would render nowhere.
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.typeString:{\"key\":\"shared\"}")).toBeInTheDocument())
  })
})

describe("resends the snapshot under a fresh id", () => {
  it("resends the snapshot under a fresh id", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-1", text: "hello" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][0]).toBe("hello")
    expect(sendMessageMock.mock.calls[0][1]?.clientMessageId).not.toBe("orig-1")
    // Matches every other programmatic resend call site in the app: without
    // this, a duplicate of this exact text still pending on this connection
    // throws instead of sending.
    expect(sendMessageMock.mock.calls[0][1]?.force).toBe(true)
    expect(sendMessageMock.mock.calls[0][2]).toEqual([])
    expect(submitMock).toHaveBeenCalledTimes(1)
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()

    // With attachments: the same File objects travel to sendMessage.
    const file = new File(["x"], "a.txt")
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-2", text: "hi", files: [file] })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    expect(sendMessageMock.mock.calls[0][2]).toEqual([file])
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())
    expect(submitMock).toHaveBeenCalledTimes(1)

    sendMessageMock.mockResolvedValueOnce(undefined)
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(2))
    expect(submitMock).toHaveBeenCalledTimes(1)
    const firstId = sendMessageMock.mock.calls[0][1]?.clientMessageId
    const secondId = sendMessageMock.mock.calls[1][1]?.clientMessageId
    expect(secondId).not.toBe(firstId)
    expect(secondId).not.toBe("orig-2")
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()

    // Saving the last context value while a required secret stays missing:
    // the turn would fail on that secret again, so the message is not
    // resent, the dialog says so and closes.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-3", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
      ]),
    ])))
    sendMessageMock.mockClear()
    toastMock.mockClear()
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(sendMessageMock).not.toHaveBeenCalled()
    expect(toastMock.mock.calls).toEqual([
      ['connectorRuntime.savedNotResentUnsupported:{"keys":"s1"}'],
    ])
    cleanup()
    sendMessageMock.mockClear()
    submitMock.mockClear()
    toastMock.mockClear()

    // A hung resend must not fire twice from a double click on the retry
    // button.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-4", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())
    sendMessageMock.mockClear()
    let resolveRetry: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveRetry = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    await act(async () => { resolveRetry() })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(sendMessageMock).toHaveBeenCalledTimes(1)
  })
})

describe("ignores a resend result superseded by a new request", () => {
  it("ignores a resend result superseded by a new request", async () => {
    // Get into "value already saved but message not sent": save-and-resend
    // succeeds on the save half and fails on the resend half.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    const closeSpy = vi.spyOn(latestActions, "close")
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-6", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())

    // Click retry with sendMessage hung, then a same-task new request (a new
    // seq) arrives while it is still in flight.
    sendMessageMock.mockClear()
    let resolveRetry: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveRetry = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    await openForTask() // retargets this same dialog instance mid-resend
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    // Now let the stale resend succeed: it must not close the new request's
    // dialog, and must not report "resent" for a turn that was never
    // resent under this request. It did go out, though -- the fresher
    // request may render its own resend button next, and without a toast
    // the user has no way to tell that clicking it would send this same
    // turn a second time.
    await act(async () => { resolveRetry() })
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(closeSpy).not.toHaveBeenCalledWith("resent")
    expect(toastMock.mock.calls).toEqual([["connectorRuntime.resendSupersededUnknown"]])
  })

  it("stays silent when the superseded retry failed, since nothing went out", async () => {
    // Same sequence, but the stale retry rejects. Telling the user the
    // message "was sent, do not resend it" here would be false and would
    // talk them out of the one resend they still need.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-7", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())

    sendMessageMock.mockClear()
    let rejectRetry: (e: Error) => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((_res, rej) => { rejectRetry = rej }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    await openForTask()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    toastMock.mockClear()

    await act(async () => { rejectRetry(new Error("closed")) })
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("reports a superseded save-and-resend whose save still landed", () => {
  it("reports a superseded save-and-resend whose save still landed", async () => {
    // A same-task terminal frame retargets this dialog instance (bumps seq)
    // while the save half of "save and resend" is still in flight, before
    // its own result comes back. Unlike the resend-only race above, the
    // save here does land server-side -- the other two "did not resend"
    // paths already say so with a toast, and this one must too.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-1", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })

    let resolveSubmit: (value: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))

    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    await openForTask() // retargets this same dialog instance mid-save
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    toastMock.mockClear()
    sendMessageMock.mockClear()
    await act(async () => { resolveSubmit(ok(report(true, []))) })

    expect(toastMock.mock.calls).toEqual([["connectorRuntime.savedNotResentSuperseded"]])
    expect(sendMessageMock).not.toHaveBeenCalled()
  })
})

describe("clears a rejected save's error once a later save lands", () => {
  it("does not show the first rejection after a save-and-resend succeeds and its resend fails", async () => {
    // First save is rejected (empty_value); the dialog stays open showing
    // that row's error and never resends. The user then fixes the value and
    // saves again -- this time the save lands (report comes back met) and
    // only the resend fails. The first rejection's error must not still be
    // on screen next to the send-failed panel for a save that succeeded.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await recordThenOpen({ taskId: 1, clientMessageId: "orig-1", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })

    submitMock.mockResolvedValueOnce({
      ok: false, kind: "coded", status: 400, code: "invalid_runtime_context",
      reason: "empty_value.context.token", connectorRef: REF_A,
    })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() =>
      expect(screen.getByText("connectorRuntime.errors.emptyValue:{\"key\":\"token\"}")).toBeInTheDocument(),
    )
    expect(sendMessageMock).not.toHaveBeenCalled()

    fireEvent.change(screen.getByLabelText("token"), { target: { value: "y" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())

    expect(screen.queryByText(/connectorRuntime\.errors\.emptyValue/)).not.toBeInTheDocument()
  })
})

async function openSimpleDialog(taskId = 1, withStash = true) {
  fetchMock.mockResolvedValueOnce(ok(report(false, [
    connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
  ])))
  renderHarness()
  if (withStash) await recordThenOpen({ taskId, clientMessageId: `orig-${taskId}`, text: "hi" })
  else await openForTask(taskId)
  await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
}

describe("clears the stash when the dialog closes", () => {
  it("clears the stash when the dialog closes", async () => {
    await openSimpleDialog()
    // A fresh delivery lands for this task while the dialog is already open
    // (its own stash was already claimed into the request on open, so this
    // is the only way to put a non-null payload in front of the close click
    // below -- without it this assertion would pass even if closing stopped
    // clearing the stash).
    await stageThenRecord({ taskId: 1, clientMessageId: "late-clean", text: "hi" })
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
    cleanup()

    // Read failure while a delivery lands mid-read: the stash it wrote survives.
    fetchMock.mockImplementationOnce(async () => {
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "late", text: "hi" })
      latestActions.recordDelivery({ taskId: 1, clientMessageId: "late", text: "hi" })
      return { ok: false, kind: "http", status: 500 }
    })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(latestState.request).toBeNull())
    expect((latestState.payload as { clientMessageId: string } | null)?.clientMessageId).toBe("late")
    cleanup()

    // Resent: the stash now holds the just-resent turn under its own fresh
    // id, not the id of the turn that failed. The stub writes the delivery
    // back the way production does (sendMessage -> addOptimisticUserMessage
    // -> recordDelivery); without that write-back there is no stash to keep
    // and this row could not tell a preserved stash from a cleared one.
    await openSimpleDialog()
    let resentId: string | undefined
    sendMessageMock.mockImplementationOnce(async (text, config) => {
      resentId = config?.clientMessageId
      latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: resentId as string, text })
      latestActions.recordDelivery({ taskId: 1, clientMessageId: resentId as string, text })
    })
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(resentId).toBeTruthy()
    expect(resentId).not.toBe("orig-1")
    expect(latestState.payload).toEqual({ taskId: 1, clientMessageId: resentId, text: "hi", files: [] })
    expect(latestState.request).toBeNull()
    cleanup()

    // Submitting in progress: X is ignored, nothing changes.
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    resolveSubmit(ok(report(true, [])))
  })
})

describe("clears the stash at the task-switch and reset cleanup points", () => {
  it.each([
    ["drops the stash and request of a task switched away from", async () => {
      await openSimpleDialog(1)
      // A fresh delivery for the same task after the original stash was
      // already claimed into the request on open -- without this, `payload`
      // would already be null before the switch below, and this assertion
      // would pass even if switching tasks stopped narrowing the stash.
      await stageThenRecord({ taskId: 1, clientMessageId: "still-here", text: "hi" })
      await act(async () => { latestActions.retainOnlyTask(2) })
      expect(latestState).toEqual({ request: null, payload: null })
    }],
    ["clears the stash on conversation reset", async () => {
      await openSimpleDialog(1)
      await act(async () => { latestActions.retainOnlyTask(null) })
      expect(latestState).toEqual({ request: null, payload: null })
      cleanup()

      // The dialog unmounts (AppProvider going away), a late delivery still
      // lands, then it remounts with no viewed task: the mount-time task
      // effect (not the unmount cleanup, which already ran) wipes it.
      appStateRef.taskId = null
      const toggle = render(providerTree())
      toggle.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
      await act(async () => {
        latestActions.stagePendingDelivery({ taskId: 1, clientMessageId: "x", text: "hi" })
        latestActions.recordDelivery({ taskId: 1, clientMessageId: "x", text: "hi" })
      })
      toggle.rerender(providerTree())
      await waitFor(() => expect(latestState.payload).toBeNull())
    }],
  ])("%s", async (_name, run) => { await run() })
})

function providerTree() {
  return <ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>
}

describe("clears the stash and request when the signed-in user changes", () => {
  it("clears the stash and request when the signed-in user changes", async () => {
    fetchMock.mockResolvedValue(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const { rerender } = render(providerTree())
    await recordThenOpen({ taskId: 1, clientMessageId: "x", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    // Same identity, re-rendered: a control row proving the effect only
    // fires on an actual id change, not on every render.
    rerender(providerTree())
    expect(screen.getByRole("dialog")).toBeInTheDocument()

    authUserRef.current = { id: "u2" }
    rerender(providerTree())
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
    cleanup()

    // Logout: id -> null clears the same way.
    authUserRef.current = { id: "u1" }
    const second = render(providerTree())
    await recordThenOpen({ taskId: 1, clientMessageId: "y", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    authUserRef.current = null
    second.rerender(providerTree())
    await waitFor(() => expect(latestState).toEqual({ request: null, payload: null }))
  })
})

describe("closes after save by the shared outcome", () => {
  it("closes after save by the shared outcome", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
        input({ section: "secrets", key: "s2", type: "string", required: false }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    // Opening-time annotation names only the required secret, not the optional one.
    expect(screen.getByText('connectorRuntime.stillMissingAfterSave:{"keys":"s1"}')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("k1"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "s1", type: "string", required: true }),
        input({ section: "secrets", key: "s2", type: "string", required: false }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(toastMock).toHaveBeenCalledTimes(1)
    expect(toastMock.mock.calls[0][0]).toBe('connectorRuntime.onlyUnsupportedRemaining:{"keys":"s1"}')
    cleanup()
    toastMock.mockClear()

    // Two required context keys, only one filled: stays open, no toast.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true }),
        input({ section: "context", key: "k2", type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("k1"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "k1", type: "string", required: true, satisfied: true }),
        input({ section: "context", key: "k2", type: "string", required: true }),
      ]),
    ])))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.filled")).toBeInTheDocument())
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("offers resend only with this request's snapshot", () => {
  it("offers resend only with this request's snapshot", async () => {
    await openSimpleDialog(1, false)
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeInTheDocument()
    expect(screen.queryByText("connectorRuntime.actions.saveAndResend")).not.toBeInTheDocument()
    cleanup()

    await openSimpleDialog(1, true)
    expect(screen.getByText("connectorRuntime.actions.saveAndResend")).toBeInTheDocument()
  })
})

describe("keeps the key-name hint out of the submit rule", () => {
  it("keeps the key-name hint out of the submit rule", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "bad key", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    expect(screen.getByText("connectorRuntime.keyNameWarning")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("bad key"), { target: { value: "x" } })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
  })
})

describe("keeps the dialog open and every draft on a failed save", () => {
  it("keeps the dialog open and every draft on a failed save", async () => {
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "kept-draft" } })

    for (const outcome of [
      { ok: false as const, kind: "transport" as const },
      { ok: false as const, kind: "coded" as const, status: 503, code: "connector_runtime_unavailable" },
      { ok: false as const, kind: "http" as const, status: 500 },
      { ok: false as const, kind: "coded" as const, status: 400, code: "invalid_runtime_context", reason: "runtime input key must match [A-Za-z0-9_-]+", connectorRef: REF_A },
    ]) {
      submitMock.mockResolvedValueOnce(outcome)
      fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
      await waitFor(() => expect(submitMock).toHaveBeenCalled())
      expect(screen.getByRole("dialog")).toBeInTheDocument()
      expect(screen.getByLabelText("token")).toHaveValue("kept-draft")
    }

    // The two retryable dispositions offer a retry button that re-POSTs the
    // same body.
    submitMock.mockClear()
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.retry")).toBeInTheDocument())
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.retry"))
    await waitFor(() => expect(submitMock).toHaveBeenCalledTimes(2))
    expect(submitMock.mock.calls[0][1]).toEqual(submitMock.mock.calls[1][1])
  })
})

describe("logs only a fixed prefix and a status", () => {
  it("logs only a fixed prefix and a status", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {})
    const infoSpy = vi.spyOn(console, "info").mockImplementation(() => {})

    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "s3cr3t-draft" } })
    submitMock.mockResolvedValueOnce({ ok: false, kind: "transport" })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.errors.network")).toBeInTheDocument())
    cleanup()

    fetchMock.mockReset()
    fetchMock.mockResolvedValueOnce({ ok: false, kind: "http", status: 500 })
    renderHarness()
    await openForTask()
    await waitFor(() => expect(warnSpy).toHaveBeenCalled())

    expect(warnSpy.mock.calls).toEqual([["[connector-runtime] requirements read failed", 500]])
    const leaked = JSON.stringify([errorSpy.mock.calls, logSpy.mock.calls, infoSpy.mock.calls])
    expect(leaked).not.toContain("s3cr3t-draft")
  })
})

describe("renders a hostile connector name and key as plain text, not markup", () => {
  it("renders a hostile connector name and key as plain text, not markup", async () => {
    // A connector name and a key name are the only server-controlled
    // strings this dialog renders (the key name reaches here even when the
    // per-turn gate would reject it, precisely so a rejected key still
    // renders instead of silently vanishing from the report --
    // schemas/connector_runtime.py's own docstring), so both are exercised
    // here with markup-shaped text: this is the observable rendering
    // contract, checked at the DOM level rather than by scanning source
    // text for sink spellings a comment or a helper could dodge either way.
    const hostileName = "<img src=x onerror=\"window.__connectorRuntimeHostile = true\">"
    const hostileKey = "<script>window.__connectorRuntimeHostile = true</script>"
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, hostileName, [
        input({ section: "context", key: hostileKey, type: "string", required: true }),
      ]),
    ])))
    renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())

    const dialog = screen.getByRole("dialog")
    expect(dialog.querySelector("img")).toBeNull()
    expect(dialog.querySelector("script")).toBeNull()
    expect(screen.getByText(hostileName)).toBeInTheDocument()
    expect(screen.getByText(hostileName).textContent).toBe(hostileName)
    const keyLabel = screen.getByText(hostileKey)
    expect(keyLabel.textContent).toBe(hostileKey)
    expect((window as unknown as { __connectorRuntimeHostile?: boolean }).__connectorRuntimeHostile)
      .toBeUndefined()
  })
})

function leafTranslationPaths(obj: Record<string, unknown>, prefix: string): string[] {
  return Object.entries(obj).flatMap(([k, v]) =>
    typeof v === "string" ? [`${prefix}${k}`] : leafTranslationPaths(v as Record<string, unknown>, `${prefix}${k}.`),
  )
}

describe("resolves every connectorRuntime key the dialog uses", () => {
  it("resolves every connectorRuntime key the dialog uses", async () => {
    const { translations, resolveTranslation } = await import("@/i18n/translations")
    expect(libSource).toContain("export function readConnectorRuntimeReport")
    expect(dialogSource).toContain("export function ConnectorRuntimeDialog")
    expect(dialogSource).not.toMatch(/connectorRuntime\.\$\{/)
    expect(libSource).not.toMatch(/connectorRuntime\.\$\{/)

    const usedKeys = new Set<string>()
    for (const source of [libSource, dialogSource]) {
      for (const match of source.matchAll(/"(connectorRuntime\.[a-zA-Z0-9_.]+)"/g)) usedKeys.add(match[1])
    }
    const definedKeys = new Set(
      leafTranslationPaths(translations.en.connectorRuntime as Record<string, unknown>, "connectorRuntime."),
    )
    expect(Array.from(usedKeys).sort()).toEqual(Array.from(definedKeys).sort())

    for (const locale of ["en", "zh"] as const) {
      for (const key of usedKeys) {
        expect(resolveTranslation(locale, key as never)).not.toBe(key)
      }
    }
    expect(translations.zh.connectorRuntime.keyNameWarning).toContain("在改名之前这个连接器每次都会失败")
  })
})

describe("does not open off the host routes", () => {
  it("does not open off the host routes", async () => {
    // Not a host route when the request arrives: no read, request cleared.
    pathnameRef.current = "/settings"
    renderHarness()
    await openForTask()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(latestState.request).toBeNull()
    cleanup()

    // A host route when requested, but the path changes before the read
    // resolves: still not shown.
    pathnameRef.current = "/task/1"
    let resolveFetch: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFetch = res }))
    renderHarness()
    await openForTask()
    pathnameRef.current = "/settings"
    await act(async () => {
      resolveFetch(ok(report(false, [
        connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      ])))
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(latestState.request).toBeNull()
    cleanup()

    // Already visible, then the path leaves the host routes: closes with
    // "left-host"; a mid-read delivery's stash survives.
    pathnameRef.current = "/task/1"
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const first = renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    await stageThenRecord({ taskId: 1, clientMessageId: "z", text: "hi" })
    // "not-shown"/"resent" also keep the stash, so only a spy on the actual
    // argument proves this passed "left-host".
    const closeSpy = vi.spyOn(latestActions, "close")
    pathnameRef.current = "/settings"
    first.rerender(providerTree())
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(closeSpy).toHaveBeenCalledWith("left-host")
    expect(latestState.request).toBeNull()
    expect((latestState.payload as { clientMessageId: string } | null)?.clientMessageId).toBe("z")
    cleanup()

    // Reverse control: moving between two host routes (a trailing slash on
    // the same workforce run page) does not close it.
    pathnameRef.current = "/workforces/7/run"
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    const second = renderHarness()
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    pathnameRef.current = "/workforces/7/run/"
    second.rerender(providerTree())
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(latestState.request).not.toBeNull()
  })
})

describe("does nothing after unmount when a request settles", () => {
  it("does nothing after unmount when a request settles", async () => {
    // POST in flight when the tree unmounts: resolving it afterward does not
    // touch state, does not resend, and does not close the dialog -- it
    // does, however, toast once, since the save landed server-side with no
    // way left to run the resend the user asked for (see the dedicated
    // describe block below for that toast and its two control cases). Uses
    // a resend snapshot and "save and resend" (not "save only") so a broken
    // alive-after-unmount guard would show up as a spurious sendMessage
    // call, not just a same-outcome no-op.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    const first = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "unmount-guard", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    first.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    sendMessageMock.mockClear()
    await act(async () => { resolveSubmit(ok(report(true, []))) })
    expect(sendMessageMock).not.toHaveBeenCalled()
    expect(toastMock.mock.calls).toEqual([["connectorRuntime.savedNotResentUnmounted"]])
    toastMock.mockClear()
    cleanup()

    // GET in flight when the task is switched away from: resolving it does nothing.
    let resolveFetch: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveFetch = res }))
    renderHarness()
    await openForTask()
    await act(async () => { latestActions.retainOnlyTask(2) })
    await act(async () => {
      resolveFetch(ok(report(false, [
        connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      ])))
    })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    cleanup()

    // Resend's sendMessage in flight when the tree unmounts: rejecting it
    // afterward touches no state, but the save has landed and the message
    // did not go out, and the send-failed panel that would normally say so
    // can never render -- so it is said once, globally.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let rejectSend: (e: Error) => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((_res, rej) => { rejectSend = rej }))
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    const third = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "x", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalledTimes(1))
    third.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    await act(async () => { rejectSend(new Error("closed")) })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(toastMock.mock.calls).toEqual([["connectorRuntime.sendFailed"]])
    cleanup()

    // The same unmount with a resend that succeeds stays silent: the message
    // arrives in the transcript on its own.
    toastMock.mockClear()
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let resolveSend: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise<void>((res) => { resolveSend = res }))
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    const fourth = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "y", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(sendMessageMock).toHaveBeenCalled())
    fourth.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    await act(async () => { resolveSend() })
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("toasts once for a landed save the tree is gone before it can resend", () => {
  it("does not toast when the save itself failed before the tree unmounted", async () => {
    // Same shape as the top of "does nothing after unmount when a request
    // settles" above, but the save POST resolves with a failure instead of
    // ok: nothing was written server-side, so there is no "saved but not
    // resent" fact to report, and the toast must stay silent.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    const first = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "unmount-guard-failed-save", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    first.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    await act(async () => { resolveSubmit({ ok: false, kind: "malformed" }) })
    expect(toastMock).not.toHaveBeenCalled()
  })

  it("does not toast when only 'save only' was requested before the tree unmounted", async () => {
    // Same shape again, but the click is "Save only": the user never asked
    // for a resend, so a save landing after unmount has nothing it failed
    // to deliver on.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    let resolveSubmit: (v: unknown) => void = () => {}
    submitMock.mockReturnValueOnce(new Promise((res) => { resolveSubmit = res }))
    const first = render(<ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>)
    await recordThenOpen({ taskId: 1, clientMessageId: "unmount-guard-save-only", text: "hi" })
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    first.rerender(<ConnectorRuntimeDialogProvider><Probe mounted={false} /></ConnectorRuntimeDialogProvider>)
    await act(async () => { resolveSubmit(ok(report(true, []))) })
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("stays usable when StrictMode remounts it", () => {
  it("stays usable when StrictMode remounts it", async () => {
    // Development builds run under StrictMode (next.config.mjs enables it),
    // where React mounts, cleans up and mounts again. Every other case in
    // this file renders without it, so none of them would notice an
    // unmount flag the interleaved cleanup left set on a dialog that is
    // still mounted -- which would short-circuit the read continuation and
    // leave the dialog permanently invisible in dev.
    fetchMock.mockResolvedValue(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    render(
      <React.StrictMode>
        <ConnectorRuntimeDialogProvider><Probe /></ConnectorRuntimeDialogProvider>
      </React.StrictMode>,
    )
    await openForTask()
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument())
    expect(screen.getByLabelText("token")).toBeInTheDocument()
  })
})

describe("warns when a resend fails", () => {
  it("warns when a resend fails", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())
    // The same fixed prefix the read path logs, and nothing else: the
    // rejection value is arbitrary and could carry message content.
    expect(warnSpy.mock.calls).toEqual([["[connector-runtime] resend failed"]])
  })
})

describe("keeps the save buttons disabled until a failure refresh settles", () => {
  it("keeps the save buttons disabled until a failure refresh settles", async () => {
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    // A failure whose disposition asks for a refresh, with the refresh GET
    // held open: that is the window in which a second submit would be built
    // from the report the refresh is about to replace.
    submitMock.mockResolvedValueOnce({ ok: false, kind: "malformed" })
    let resolveRefresh: (v: unknown) => void = () => {}
    fetchMock.mockReturnValueOnce(new Promise((res) => { resolveRefresh = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    const saveOnly = screen.getByText("connectorRuntime.actions.saveOnly")
    expect(saveOnly).toBeDisabled()
    fireEvent.click(saveOnly)
    expect(submitMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveRefresh(ok(report(false, [
        connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
      ])))
    })
    expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled()
  })
})

describe("holds the dialog open while a retry resend is in flight", () => {
  it("holds the dialog open while a retry resend is in flight", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {})
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())

    let resolveRetry: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveRetry = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    const beforeDismiss = latestState

    // Escape and the window's own close control both run the same guard.
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape", code: "Escape" })
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(latestState).toEqual(beforeDismiss)

    await act(async () => { resolveRetry() })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })
})

describe("keeps the submit gate closed while a retry resend is in flight, even after a same-task refresh reopens it", () => {
  it("keeps the submit gate closed while a retry resend is in flight, even after a same-task refresh reopens it", async () => {
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(ok(report(true, [])))
    sendMessageMock.mockRejectedValueOnce(new Error("closed"))
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.sendFailed")).toBeInTheDocument())

    // Click retry with the resend held open: `resending` stays true until
    // this resolves.
    sendMessageMock.mockClear()
    let resolveRetry: () => void = () => {}
    sendMessageMock.mockReturnValueOnce(new Promise((res) => { resolveRetry = res }))
    fireEvent.click(screen.getByText("connectorRuntime.actions.resend"))
    expect(sendMessageMock).toHaveBeenCalledTimes(1)

    // Same task, another terminal frame arrives while that resend is still
    // in flight -- e.g. a second tab's own broadcast of the same failure.
    // openForTask bumps the request's seq, the read effect re-fetches, and
    // installs a fresh fillable report: `sendFailed` clears and the
    // footer's save buttons come back.
    fetchMock.mockResolvedValueOnce(ok(report(false, [
      connector(REF_A, "A", [input({ section: "context", key: "token", type: "string", required: true })]),
    ])))
    await openForTask()
    await waitFor(() => expect(screen.queryByText("connectorRuntime.sendFailed")).not.toBeInTheDocument())
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "y" } })

    const saveOnly = screen.getByText("connectorRuntime.actions.saveOnly")
    const saveAndResend = screen.getByText("connectorRuntime.actions.saveAndResend")
    expect(saveOnly).toBeDisabled()
    expect(saveAndResend).toBeDisabled()

    // Even a click against the (disabled) button must not let a second send
    // go out under a fresh message id while the first retry is still
    // unaccounted for.
    fireEvent.click(saveAndResend)
    expect(sendMessageMock).toHaveBeenCalledTimes(1)
    expect(submitMock).toHaveBeenCalledTimes(1)

    await act(async () => { resolveRetry() })
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.saveOnly")).toBeEnabled())
  })
})

describe("tells the user when save and resend did not resend", () => {
  it("tells the user when save and resend did not resend", async () => {
    // Saved, but the server still reports the connector unavailable with
    // nothing left for this dialog to collect: the turn would fail on the
    // same gate, so it is not resent. Without the toast the primary button
    // would have done half of what it promised, silently.
    const stillUnavailable = () => ok(report(false, [
      connector(REF_A, "A", [
        input({ section: "context", key: "token", type: "string", required: true, satisfied: true }),
        input({ section: "secrets", key: "s1", type: "string", required: false }),
      ]),
    ]))

    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(stillUnavailable())
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveAndResend"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.acknowledge")).toBeInTheDocument())
    expect(sendMessageMock).not.toHaveBeenCalled()
    expect(toastMock.mock.calls).toEqual([["connectorRuntime.savedNotResentUnavailable"]])
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    cleanup()
    toastMock.mockClear()
    sendMessageMock.mockClear()

    // "Save only" on the same outcome promised no resend, so it says nothing.
    await openSimpleDialog()
    fireEvent.change(screen.getByLabelText("token"), { target: { value: "x" } })
    submitMock.mockResolvedValueOnce(stillUnavailable())
    fireEvent.click(screen.getByText("connectorRuntime.actions.saveOnly"))
    await waitFor(() => expect(screen.getByText("connectorRuntime.actions.acknowledge")).toBeInTheDocument())
    expect(toastMock).not.toHaveBeenCalled()
  })
})

describe("stays quiet when mounted with no provider", () => {
  it("does not warn when the widget/share shape (no ConnectorRuntimeDialogProvider) mounts and unmounts", async () => {
    // A widget or share page mounts ConnectorRuntimeDialog with no
    // ConnectorRuntimeDialogProvider above it by design (see the component's
    // own docstring); its mount and unmount effects must not call the
    // no-op default's actions, since doing so would trip the dev-only
    // "called outside provider" warning that exists to catch an actual
    // wiring mistake -- not this expected shape. This asserts the console
    // stays silent through a full mount-then-unmount cycle. A caller that
    // genuinely reaches an action from outside a provider by some other
    // path still warns (layout.test.tsx's provider-boundary test covers
    // that case for openForTask).
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    const { unmount } = render(<ConnectorRuntimeDialog />)
    unmount()
    expect(warn).not.toHaveBeenCalled()
    warn.mockRestore()
  })
})
