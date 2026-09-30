import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { I18nProvider } from "@/contexts/i18n-context"
import type { AppState } from "@/contexts/app-context-chat"

// The task page's session-check trigger. Kept apart from page-client.test.tsx
// because that file's app-context mock does not expose `sameTaskReconnects`,
// which every reconnect case here depends on.

const navigation = vi.hoisted(() => ({ params: { id: "1" } as { id: string } }))
vi.mock("next/navigation", () => ({
  useParams: () => navigation.params,
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="conversation-panel" />,
}))

vi.mock("@/contexts/auth-context", () => ({ useAuth: () => ({ user: { id: "u1" } }) }))

const app = vi.hoisted(() => ({
  taskId: 1 as number | null,
  sameTaskReconnects: 0,
  setTaskId: vi.fn(),
  closeFilePreview: vi.fn(),
}))
vi.mock("@/contexts/app-context-chat", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/contexts/app-context-chat")>()
  return {
    ...actual,
    useApp: () => ({
      state: { taskId: app.taskId, currentTask: null, dagExecution: null, steps: [] } as Partial<AppState>,
      sameTaskReconnects: app.sameTaskReconnects,
      setTaskId: app.setTaskId,
      closeFilePreview: app.closeFilePreview,
    }),
  }
})

import TaskDetailPage, {
  INITIAL_SESSION_CHECK_WATCH,
  nextSessionCheck,
  type SessionCheckWatch,
} from "./page-client"
import {
  ConnectorRuntimeDialogProvider,
  useConnectorRuntimeDialog,
  useConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogActions,
  type ConnectorRuntimeDialogValue,
  type SessionCheckCause,
} from "@/contexts/connector-runtime-dialog-context"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  navigation.params = { id: "1" }
  app.taskId = 1
  app.sameTaskReconnects = 0
})

// One observation per effect run: [the URL's task id, the state's task id,
// the app's sameTaskReconnects] and the check it must ask for, if any.
type Step = [number, number | null, number, [number, SessionCheckCause] | null]
const N = Number.NaN

function replay(steps: Step[]): Array<[number, SessionCheckCause] | null> {
  let watch: SessionCheckWatch = INITIAL_SESSION_CHECK_WATCH
  return steps.map(([urlTaskId, stateTaskId, reconnects]) => {
    const { next, check } = nextSessionCheck(watch, { urlTaskId, stateTaskId, reconnects })
    watch = next
    return check === null ? null : [check.taskId, check.cause]
  })
}

describe("nextSessionCheck", () => {
  it.each<[string, Step[]]>([
    ["E1 mount, then the first connect (not counted), then a reconnect", [
      [5, 5, 0, [5, "opened"]], [5, 5, 0, null], [5, 5, 1, [5, "reconnected"]],
    ]],
    ["E2 mount after earlier reconnects of the same task: only later ones check", [
      [5, 5, 3, [5, "opened"]], [5, 5, 3, null], [5, 5, 4, [5, "reconnected"]],
    ]],
    ["E3 arrive from another task: nothing while the URL and the state disagree", [
      [5, 3, 2, null], [5, 5, 2, [5, "opened"]], [5, 5, 2, null], [5, 5, 3, [5, "reconnected"]],
    ]],
    ["E4 soft navigation from one task to another", [
      [3, 3, 0, [3, "opened"]], [3, 3, 1, [3, "reconnected"]],
      [5, 3, 1, null], [5, 5, 1, [5, "opened"]], [5, 5, 2, [5, "reconnected"]],
    ]],
    ["E5 back from the new-conversation page: the app counts that socket open as a same-task reconnect", [
      [3, null, 0, null], [3, 3, 0, [3, "opened"]], [3, 3, 1, [3, "reconnected"]],
    ]],
    ["E6 a URL id that is not a number", [[N, 5, 0, null], [N, 5, 1, null]]],
    ["E7 no task in the app state", [[5, null, 0, null], [5, null, 1, null]]],
    ["E8 the same observation twice (strict mode)", [[5, 5, 0, [5, "opened"]], [5, 5, 0, null]]],
    ["E9 two reconnects between two observations check once", [
      [5, 5, 0, [5, "opened"]], [5, 5, 2, [5, "reconnected"]], [5, 5, 2, null],
    ]],
    ["E16 the reconnect after a switch is checked however the switch's renders were merged", [
      [3, 3, 0, [3, "opened"]], [5, 5, 0, [5, "opened"]], [5, 5, 1, [5, "reconnected"]],
    ]],
    ["E19 the URL briefly disagrees, then shows the same task again: still the same view", [
      [5, 5, 0, [5, "opened"]], [7, 5, 0, null], [5, 5, 0, null], [5, 5, 1, [5, "reconnected"]],
    ]],
    ["E20 a reconnect while the URL disagrees is checked for the task the view is on", [
      [5, 5, 0, [5, "opened"]], [7, 5, 1, [5, "reconnected"]], [5, 5, 1, null],
    ]],
    ["E21 the app state leaves the task and comes back: a new view", [
      [5, 5, 0, [5, "opened"]], [5, 7, 0, null], [5, 5, 0, [5, "opened"]],
    ]],
  ])("%s", (_name, steps) => {
    expect(replay(steps)).toEqual(steps.map(step => step[3]))
  })
})

describe("the task page asks for a session check", () => {
  let actions: ConnectorRuntimeDialogActions | undefined
  let value: ConnectorRuntimeDialogValue | undefined
  function Probe() {
    actions = useConnectorRuntimeDialogActions()
    value = useConnectorRuntimeDialog()
    return null
  }
  function tree(page: boolean, strict = false) {
    const body = (
      <I18nProvider initialLocale="en">
        <ConnectorRuntimeDialogProvider>
          <Probe />
          {page && <TaskDetailPage />}
        </ConnectorRuntimeDialogProvider>
      </I18nProvider>
    )
    return strict ? <React.StrictMode>{body}</React.StrictMode> : body
  }
  // Mounts the provider first so the page reads the spied action.
  function mountPage(strict = false) {
    const view = render(tree(false, strict))
    const spy = vi.spyOn(actions as ConnectorRuntimeDialogActions, "openSessionCheck")
    view.rerender(tree(true, strict))
    return { view, spy, update: () => view.rerender(tree(true, strict)) }
  }

  it("checks once, as opened, when it mounts on the viewed task", () => {
    const { spy } = mountPage()
    expect(spy.mock.calls).toEqual([[1, "opened"]])
    expect(value?.request).toMatchObject({ taskId: 1, trigger: "session_open", resendPayload: null })
  })

  it("does not check while the URL and the app state disagree", () => {
    navigation.params = { id: "2" }
    const { spy, update } = mountPage()
    expect(spy).not.toHaveBeenCalled()
    app.taskId = 2
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[2, "opened"]])
  })

  it("checks once more after a soft navigation to another task", () => {
    const { spy, update } = mountPage()
    navigation.params = { id: "2" }
    act(() => { update() })
    app.taskId = 2
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[1, "opened"], [2, "opened"]])
  })

  it("checks once when the app state fills in the task after mount", () => {
    navigation.params = { id: "3" }
    app.taskId = null
    const { spy, update } = mountPage()
    app.taskId = 3
    act(() => { update() })
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[3, "opened"]])
  })

  it("checks again, as reconnected, when the app reports a same-task reconnect", () => {
    const { spy, update } = mountPage()
    app.sameTaskReconnects = 1
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[1, "opened"], [1, "reconnected"]])
  })

  it("does not bring back a dismissed check when the URL briefly disagrees", () => {
    const { spy, update } = mountPage()
    act(() => { actions?.close("dismissed", 1) })
    expect(value?.request).toBeNull()
    navigation.params = { id: "2" }
    act(() => { update() })
    navigation.params = { id: "1" }
    act(() => { update() })
    expect(spy.mock.calls).toEqual([[1, "opened"]])
    expect(value?.request).toBeNull()
  })

  it("checks once under StrictMode", () => {
    const { spy } = mountPage(true)
    expect(spy.mock.calls).toEqual([[1, "opened"]])
  })
})
