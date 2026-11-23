import { createLinkSessionWorkerCore } from "./link-session-worker-core"

// Entry file of the link session Worker: wiring only. All behaviour lives in
// link-session-worker-core.ts.

interface WorkerScope {
  addEventListener: (type: string, listener: (event: Event) => void) => void
  postMessage: (message: unknown) => void
}

const scope = self as unknown as WorkerScope

// An exception that is not handled inside the Worker would be re-fired as an
// ErrorEvent on the Worker object in the page, carrying its message text. Mark
// both kinds as handled so no error text leaves the Worker.
scope.addEventListener("error", (event) => event.preventDefault())
scope.addEventListener("unhandledrejection", (event) => event.preventDefault())

const core = createLinkSessionWorkerCore({
  fetch: (input, init) => fetch(input, init),
  post: (message) => scope.postMessage(message),
  now: () => Date.now(),
  random: () => Math.random(),
  setTimer: (callback, ms) => setTimeout(callback, ms),
  clearTimer: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
})

scope.addEventListener("message", (event) => core.handleMessage((event as MessageEvent).data))
core.announce()
