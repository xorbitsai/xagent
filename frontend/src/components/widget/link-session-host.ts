import {
  LINK_CODE_PATTERN,
  MAX_RAPID_RECOVERIES,
  SESSION_IFRAME_PATH,
  SESSION_REFRESH_THRESHOLD_MS,
  STABLE_CONNECTION_MS,
  WORKER_HELLO_TIMEOUT_MS,
  WORKER_REDEEMING_TIMEOUT_MS,
  endedUrl,
  isNonBlankString,
  isRecord,
  isTimestamp,
  isWorkerTerminalReason,
  type EndReason,
  type LinkSessionFields,
} from "./link-session-constants"

// Main-thread logic of the link host page. No React and no network calls: the
// redeem and renew requests belong to the Worker (link-session-worker-core.ts),
// which is the only holder of the reconnect token. This module
//
//  - erases the link code from the address bar, hands it to the Worker and
//    forgets it,
//  - watches that the Worker is alive and fails closed when it is not,
//  - speaks the postMessage protocol with the chat iframe,
//  - counts rapid recoveries (the circuit breaker, copied from widget.js
//    runRecoveryFlow), and
//  - turns every way the session can end into one `location.replace` to the end
//    page, whose `reason` is the contract with the embedding app.

export interface LinkWorkerLike {
  postMessage: (message: unknown) => void
  terminate: () => void
  addEventListener: (
    type: "message" | "error" | "messageerror",
    listener: (event: { data?: unknown; preventDefault?: () => void }) => void,
  ) => void
}

export interface LinkHostSnapshot {
  phase: "starting" | "active" | "terminated"
  // Set once the first session arrived; the page creates the iframe from it.
  iframeSrc: string | null
  closeButtonVisible: boolean
}

export interface LinkSessionHostDeps {
  readHash: () => string
  readPathname: () => string
  readSearch: () => string
  readOrigin: () => string
  replaceState: (url: string) => void
  navigate: (url: string) => void
  createWorker: () => LinkWorkerLike
  // The iframe's contentWindow, or null while there is no iframe.
  getIframeWindow: () => unknown
  // Posts to the iframe with its own origin as target.
  postToIframe: (message: Record<string, unknown>) => void
  now: () => number
  setTimer: (callback: () => void, ms: number) => unknown
  clearTimer: (handle: unknown) => void
  onChange: (snapshot: LinkHostSnapshot) => void
}

// Deliberately holds no link code and no reconnect token.
export interface LinkHostState {
  phase: LinkHostSnapshot["phase"]
  session: LinkSessionFields | null
  iframeReady: boolean
  chromeReady: boolean
  renewInFlight: boolean
  recoverableCode: "network_unavailable" | null
  rapidRecoveries: number
  deliverySequence: number
  // The id of the session currently held; every new session gets one id.
  currentDeliveryId: string | null
  awaitedDeliveryId: string | null
  invalidatedDeliveryId: string | null
  stableDeliveryId: string | null
}

export interface LinkSessionHost {
  start: () => void
  handleMessageEvent: (event: { source: unknown; origin: string; data: unknown }) => void
  handlePageShow: (event: { persisted: boolean }) => void
  // The network came back or the page became visible again.
  wake: () => void
  // The page's own close button.
  close: () => void
  // Stops listening and timers without navigating (the page unmounted).
  dispose: () => void
  getState: () => LinkHostState
}

export function extractLinkCode(hash: string): string | null {
  const code = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash).get("code")
  return code !== null && LINK_CODE_PATTERN.test(code) ? code : null
}

// The chat iframe URL. `timezone` is passed through the way widget.js
// withTimezone does it; `voice` is never added.
export function buildIframeSrc(search: string): string {
  const timezone = new URLSearchParams(search).get("timezone")?.trim()
  if (!timezone) return SESSION_IFRAME_PATH
  return `${SESSION_IFRAME_PATH}?timezone=${encodeURIComponent(timezone)}`
}

function parseSessionFields(value: unknown): LinkSessionFields | null {
  if (
    !isRecord(value)
    || !isNonBlankString(value.session_token)
    || !isTimestamp(value.session_token_expires_at)
    || !isTimestamp(value.absolute_expires_at)
    || !isRecord(value.agent)
  ) {
    return null
  }
  const fields: LinkSessionFields = {
    session_token: value.session_token,
    session_token_expires_at: value.session_token_expires_at,
    absolute_expires_at: value.absolute_expires_at,
    agent: value.agent,
  }
  if (isRecord(value.capabilities)) fields.capabilities = value.capabilities
  return fields
}

export function createLinkSessionHost(deps: LinkSessionHostDeps): LinkSessionHost {
  const state: LinkHostState = {
    phase: "starting",
    session: null,
    iframeReady: false,
    chromeReady: false,
    renewInFlight: false,
    recoverableCode: null,
    rapidRecoveries: 0,
    deliverySequence: 0,
    currentDeliveryId: null,
    awaitedDeliveryId: null,
    invalidatedDeliveryId: null,
    stableDeliveryId: null,
  }
  let worker: LinkWorkerLike | null = null
  let iframeSrc: string | null = null
  let search = ""
  let disposed = false
  let redeemingSeen = false
  let helloTimer: unknown = null
  let redeemingTimer: unknown = null
  let expiryTimer: unknown = null
  let expiryFor: string | null = null
  let stabilityTimer: unknown = null

  function notify(): void {
    deps.onChange({
      phase: state.phase,
      iframeSrc: state.phase === "terminated" ? null : iframeSrc,
      closeButtonVisible: !state.chromeReady,
    })
  }

  function clearTimer(handle: unknown): null {
    if (handle !== null) deps.clearTimer(handle)
    return null
  }

  function clearWatchdogs(): void {
    helloTimer = clearTimer(helloTimer)
    redeemingTimer = clearTimer(redeemingTimer)
  }

  function clearAllTimers(): void {
    clearWatchdogs()
    expiryTimer = clearTimer(expiryTimer)
    stabilityTimer = clearTimer(stabilityTimer)
  }

  function stopWorker(): void {
    const current = worker
    worker = null
    try {
      current?.terminate()
    } catch {
      // The page is leaving anyway.
    }
  }

  function terminate(reason: EndReason): void {
    if (state.phase === "terminated" || disposed) return
    state.phase = "terminated"
    state.session = null
    state.renewInFlight = false
    clearAllTimers()
    stopWorker()
    notify()
    deps.navigate(endedUrl(reason))
  }

  // A Worker that failed after it had given us a session cannot be recovered
  // here: the embedding app requests a new link and the conversation resumes.
  function onWorkerFailure(): void {
    terminate(state.session ? "session_ended" : "unavailable")
  }

  function sendToIframe(message: Record<string, unknown>): void {
    deps.postToIframe({ xagent: true, v: 1, ...message })
  }

  function mintDeliveryId(): string {
    state.deliverySequence += 1
    return `link-delivery-${deps.now().toString(36)}-${state.deliverySequence}`
  }

  function isStale(expiresAt: string): boolean {
    return Date.parse(expiresAt) - deps.now() < SESSION_REFRESH_THRESHOLD_MS
  }

  // Level-triggered: whatever the current state is gets (re)sent, and only the
  // latest state is ever sent.
  function flush(): void {
    if (state.phase !== "active" || !state.iframeReady) return
    if (state.recoverableCode) {
      sendToIframe({ type: "session_degraded", code: state.recoverableCode })
      return
    }
    const session = state.session
    if (!session) return
    if (isStale(session.session_token_expires_at)) {
      requestRenew()
      return
    }
    // A renew is already on its way; the new session is delivered when it lands.
    if (state.renewInFlight) return
    const deliveryId = state.currentDeliveryId
    if (!deliveryId) return
    if (deliveryId !== state.stableDeliveryId && deliveryId !== state.invalidatedDeliveryId) {
      state.awaitedDeliveryId = deliveryId
    }
    const update: Record<string, unknown> = {
      type: "session_update",
      session_delivery_id: deliveryId,
      session_token: session.session_token,
      session_token_expires_at: session.session_token_expires_at,
      absolute_expires_at: session.absolute_expires_at,
      agent: session.agent,
    }
    if (session.capabilities) update.capabilities = session.capabilities
    sendToIframe(update)
  }

  function invalidateStability(): void {
    stabilityTimer = clearTimer(stabilityTimer)
    if (state.awaitedDeliveryId) {
      state.invalidatedDeliveryId = state.awaitedDeliveryId
      state.awaitedDeliveryId = null
    }
  }

  function beginStability(deliveryId: string): void {
    if (deliveryId !== state.awaitedDeliveryId || deliveryId === state.invalidatedDeliveryId) return
    if (stabilityTimer !== null || state.stableDeliveryId === deliveryId) return
    const timer = deps.setTimer(() => {
      if (stabilityTimer !== timer) return
      stabilityTimer = null
      if (state.awaitedDeliveryId !== deliveryId) return
      state.awaitedDeliveryId = null
      state.stableDeliveryId = deliveryId
      state.rapidRecoveries = 0
    }, STABLE_CONNECTION_MS)
    stabilityTimer = timer
  }

  // Circuit breaker, copied from widget.js runRecoveryFlow (widget.js:1535-1561):
  // a renew already running absorbs the new request without counting it; every
  // other request first voids the unfinished stability timer, then ends the
  // session if three rapid recoveries already happened, and otherwise counts.
  function requestRenew(): void {
    if (state.phase !== "active" || !state.session || !worker) return
    if (state.renewInFlight) return
    invalidateStability()
    if (state.rapidRecoveries >= MAX_RAPID_RECOVERIES) {
      terminate("network")
      return
    }
    state.rapidRecoveries += 1
    state.renewInFlight = true
    try {
      worker.postMessage({ type: "renew" })
    } catch {
      terminate("session_ended")
    }
  }

  function scheduleExpiry(absoluteExpiresAt: string): void {
    if (expiryFor === absoluteExpiresAt) return
    expiryTimer = clearTimer(expiryTimer)
    expiryFor = absoluteExpiresAt
    const delay = Math.max(0, Date.parse(absoluteExpiresAt) - deps.now())
    expiryTimer = deps.setTimer(() => {
      expiryTimer = null
      terminate("session_ended")
    }, delay)
  }

  function onSession(value: unknown): void {
    const fields = parseSessionFields(value)
    if (!fields) {
      terminate("unavailable")
      return
    }
    clearWatchdogs()
    const first = iframeSrc === null
    // A new session replaces the delivery being timed; a repeated ready from
    // the iframe re-sends this same id.
    invalidateStability()
    state.currentDeliveryId = mintDeliveryId()
    state.session = fields
    state.renewInFlight = false
    state.recoverableCode = null
    state.phase = "active"
    scheduleExpiry(fields.absolute_expires_at)
    if (first) {
      iframeSrc = buildIframeSrc(search)
      notify()
    }
    flush()
  }

  function handleWorkerMessage(data: unknown): void {
    if (state.phase === "terminated" || disposed || !isRecord(data)) return
    switch (data.type) {
      case "hello":
        helloTimer = clearTimer(helloTimer)
        if (!redeemingSeen && redeemingTimer === null) {
          redeemingTimer = deps.setTimer(() => {
            redeemingTimer = null
            terminate("unavailable")
          }, WORKER_REDEEMING_TIMEOUT_MS)
        }
        return
      case "redeeming":
        redeemingSeen = true
        clearWatchdogs()
        return
      case "session":
        onSession(data.session)
        return
      case "degraded":
        if (data.code !== "network_unavailable") return
        state.recoverableCode = "network_unavailable"
        flush()
        return
      case "terminal":
        terminate(isWorkerTerminalReason(data.reason) ? data.reason : "unavailable")
        return
      default:
        return
    }
  }

  function start(): void {
    search = deps.readSearch()
    const code = extractLinkCode(deps.readHash())
    // The fragment is removed before the Worker is created; a failing
    // replaceState does not stop the redeem, since a code can be used only once.
    deps.replaceState(deps.readPathname() + search)
    if (code === null) {
      terminate("link_invalid")
      return
    }
    try {
      const created = deps.createWorker()
      worker = created
      created.addEventListener("message", (event) => handleWorkerMessage(event.data))
      created.addEventListener("error", (event) => {
        event.preventDefault?.()
        onWorkerFailure()
      })
      created.addEventListener("messageerror", () => onWorkerFailure())
      helloTimer = deps.setTimer(() => {
        helloTimer = null
        terminate("unavailable")
      }, WORKER_HELLO_TIMEOUT_MS)
      created.postMessage({ type: "start", code })
    } catch {
      terminate("unavailable")
    }
    notify()
  }

  function handleMessageEvent(event: { source: unknown; origin: string; data: unknown }): void {
    if (state.phase === "terminated" || disposed) return
    const iframeWindow = deps.getIframeWindow()
    if (!iframeWindow || event.source !== iframeWindow) return
    if (event.origin !== deps.readOrigin()) return
    const data = event.data
    if (!isRecord(data) || data.xagent !== true || data.v !== 1 || typeof data.type !== "string") return
    switch (data.type) {
      case "ready":
        state.iframeReady = true
        flush()
        return
      case "reconnect_request":
        requestRenew()
        return
      case "session_connection_open":
        if (isNonBlankString(data.session_delivery_id)) beginStability(data.session_delivery_id)
        return
      case "widget_close":
        terminate("closed")
        return
      case "widget_expand":
        sendToIframe({ type: "widget_expand_rejected" })
        return
      case "widget_chrome_ready":
        state.chromeReady = true
        sendToIframe({ type: "widget_expand_state", expanded: false })
        notify()
        return
      case "widget_chrome_not_ready":
        state.chromeReady = false
        notify()
        return
      default:
        return
    }
  }

  return {
    start,
    handleMessageEvent,
    // A restored page cannot be trusted to have a running Worker, and whether
    // it does is not observable. End the session either way; the embedding app
    // requests a new link and the conversation resumes.
    handlePageShow: (event) => {
      if (event.persisted) terminate("session_ended")
    },
    wake: () => {
      if (state.phase === "terminated" || disposed || !worker) return
      try {
        worker.postMessage({ type: "wake" })
      } catch {
        // A Worker that cannot be reached is handled by its error events.
      }
    },
    close: () => terminate("closed"),
    dispose: () => {
      if (disposed) return
      disposed = true
      clearAllTimers()
      stopWorker()
    },
    getState: () => ({ ...state }),
  }
}
