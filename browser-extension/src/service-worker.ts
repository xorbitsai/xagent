import {
  PROTOCOL_VERSION,
  isRecord,
  normalizeRelayUrl,
  parsePairingSetup,
  parseServerMessage,
  reconnectDelayMs,
  type RelayCommand,
  type RelayPublicStatus,
} from "./protocol.js"

const STORAGE = {
  relayUrl: "relayUrl",
  sessionToken: "sessionToken",
  clientId: "clientId",
  clientName: "clientName",
} as const
const SESSION = {
  attachedTabId: "attachedTabId",
} as const
const RECONNECT_ALARM = "browser-relay-reconnect"

let socket: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let keepaliveTimer: ReturnType<typeof setInterval> | null = null
let attachedTabId: number | null = null
let lastFrameId: string | null = null
let lastError: string | null = null
let connecting = false
let reconnectAttempt = 0
let nextRetryAt: number | null = null

void restoreRuntimeState()

chrome.runtime.onStartup.addListener(() => {
  void restoreRuntimeState()
})

chrome.runtime.onInstalled.addListener(() => {
  void ensureClientIdentity()
})

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM) {
    void reconnectStoredSession()
  }
})

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  if (!isRecord(message) || typeof message.type !== "string") {
    return false
  }
  if (message.type === "get_status") {
    void publicStatus().then(sendResponse)
    return true
  }
  if (message.type === "connect") {
    void connectFromUserInput(message)
      .then(publicStatus)
      .then(sendResponse)
      .catch((error: unknown) => sendResponse(errorStatus(error)))
    return true
  }
  if (message.type === "disconnect") {
    void forgetRelaySession()
      .then(publicStatus)
      .then(sendResponse)
    return true
  }
  if (message.type === "attach_active_tab") {
    void attachActiveTab()
      .then(publicStatus)
      .then(sendResponse)
      .catch((error: unknown) => sendResponse(errorStatus(error)))
    return true
  }
  if (message.type === "detach_tab") {
    void detachApprovedTab()
      .then(publicStatus)
      .then(sendResponse)
    return true
  }
  return false
})

chrome.debugger.onDetach.addListener((source) => {
  if (source.tabId === attachedTabId) {
    void clearAttachedTab()
  }
})

chrome.debugger.onEvent.addListener((source, method) => {
  if (
    source.tabId === attachedTabId &&
    (method === "Page.frameNavigated" ||
      method === "Page.loadEventFired" ||
      method === "DOM.documentUpdated")
  ) {
    lastFrameId = null
  }
})

async function restoreRuntimeState(): Promise<void> {
  await ensureClientIdentity()
  const session = await chrome.storage.session.get(SESSION.attachedTabId)
  const candidate = session[SESSION.attachedTabId]
  if (typeof candidate === "number") {
    const targets = await chrome.debugger.getTargets()
    if (targets.some((target) => target.tabId === candidate && target.attached)) {
      attachedTabId = candidate
    } else {
      await chrome.storage.session.remove(SESSION.attachedTabId)
    }
  }
  await updateBadge()
  const stored = await chrome.storage.local.get([
    STORAGE.relayUrl,
    STORAGE.sessionToken,
  ])
  const relayUrl = stored[STORAGE.relayUrl]
  const sessionToken = stored[STORAGE.sessionToken]
  if (typeof relayUrl === "string" && typeof sessionToken === "string") {
    connectSocket({
      relayUrl,
      sessionToken,
      reconnecting: true,
    })
  }
}

async function ensureClientIdentity(): Promise<{
  clientId: string
  clientName: string
}> {
  const stored = await chrome.storage.local.get([
    STORAGE.clientId,
    STORAGE.clientName,
  ])
  const storedClientId = stored[STORAGE.clientId]
  const storedClientName = stored[STORAGE.clientName]
  const clientId =
    typeof storedClientId === "string" ? storedClientId : crypto.randomUUID()
  const clientName =
    typeof storedClientName === "string" ? storedClientName : "Chrome"
  await chrome.storage.local.set({
    [STORAGE.clientId]: clientId,
    [STORAGE.clientName]: clientName,
  })
  return { clientId, clientName }
}

async function connectFromUserInput(
  message: Record<string, unknown>,
): Promise<void> {
  const setupCode = String(message.setupCode ?? "").trim()
  const setup = setupCode ? parsePairingSetup(setupCode) : null
  const relayUrl = setup
    ? setup.relayUrl
    : normalizeRelayUrl(String(message.relayUrl ?? ""))
  const pairingToken = setup
    ? setup.pairingToken
    : String(message.pairingToken ?? "").trim()
  const stored = await chrome.storage.local.get(STORAGE.sessionToken)
  const storedSessionToken = stored[STORAGE.sessionToken]
  const sessionToken =
    typeof storedSessionToken === "string" ? storedSessionToken : ""
  if (!pairingToken && !sessionToken) {
    throw new Error("Enter a fresh pairing token.")
  }
  await chrome.storage.local.set({ [STORAGE.relayUrl]: relayUrl })
  reconnectAttempt = 0
  nextRetryAt = null
  connectSocket({
    relayUrl,
    pairingToken: pairingToken || undefined,
    sessionToken: pairingToken ? undefined : sessionToken,
    reconnecting: false,
  })
}

function connectSocket(options: {
  relayUrl: string
  pairingToken?: string
  sessionToken?: string
  reconnecting: boolean
}): void {
  clearReconnectSchedule()
  if (socket) {
    socket.onclose = null
    socket.close()
  }
  connecting = true
  if (options.reconnecting) {
    reconnectAttempt = Math.max(1, reconnectAttempt)
  }
  lastError = null
  socket = new WebSocket(options.relayUrl)
  void updateBadge()
  socket.onopen = () => {
    void (async () => {
      try {
        const identity = await ensureClientIdentity()
        send({
          type: "hello",
          protocol_version: PROTOCOL_VERSION,
          client_id: identity.clientId,
          client_name: identity.clientName,
          ...(options.pairingToken
            ? { pairing_token: options.pairingToken }
            : { session_token: options.sessionToken }),
        })
        startKeepalive()
      } catch (error) {
        lastError = errorMessage(error)
        socket?.close()
      }
    })()
  }
  socket.onmessage = (event) => {
    void handleServerMessage(String(event.data))
  }
  socket.onerror = () => {
    lastError = "Could not connect to the Xagent relay."
    void updateBadge()
  }
  socket.onclose = () => {
    socket = null
    connecting = false
    lastFrameId = null
    stopKeepalive()
    void updateBadge()
    void scheduleReconnect()
  }
}

async function handleServerMessage(raw: string): Promise<void> {
  try {
    const message = parseServerMessage(raw)
    if (message.type === "ready") {
      connecting = false
      reconnectAttempt = 0
      nextRetryAt = null
      lastError = null
      if (message.session_token) {
        await chrome.storage.local.set({
          [STORAGE.sessionToken]: message.session_token,
        })
      }
      await sendAttachedStatus()
      await updateBadge()
      return
    }
    if (message.type === "command") {
      await handleCommand(message)
      return
    }
    if (message.type === "error") {
      lastError = message.error
      if (isAuthenticationError(message.error)) {
        await invalidateRelaySession(message.error)
      }
    }
  } catch (error) {
    lastError = errorMessage(error)
  }
}

async function handleCommand(message: RelayCommand): Promise<void> {
  try {
    if (attachedTabId === null) {
      throw new Error(
        "No tab is attached. Open the extension and approve the current tab.",
      )
    }
    let observation: BrowserObservation
    if (message.command === "observe") {
      const frameId = requiredString(message.payload.frame_id, "frame_id")
      observation = await captureObservation(attachedTabId)
      lastFrameId = frameId
    } else if (message.command === "act") {
      const expectedFrameId = requiredString(
        message.payload.expected_frame_id,
        "expected_frame_id",
      )
      const frameId = requiredString(message.payload.frame_id, "frame_id")
      if (lastFrameId !== expectedFrameId) {
        throw new Error(
          "The approved tab changed after the last screenshot. Request a fresh screenshot.",
        )
      }
      if (!isRecord(message.payload.action)) {
        throw new Error("Action payload is invalid.")
      }
      await performAction(attachedTabId, message.payload.action)
      const actionType = String(message.payload.action.type ?? "")
      if (actionType !== "navigate" && actionType !== "wait") {
        await delay(250)
      }
      observation = await captureObservation(attachedTabId)
      lastFrameId = frameId
    } else {
      throw new Error(`Unsupported relay command: ${String(message.command)}`)
    }
    send({
      type: "response",
      protocol_version: PROTOCOL_VERSION,
      request_id: message.request_id,
      success: true,
      result: { observation },
    })
  } catch (error) {
    send({
      type: "response",
      protocol_version: PROTOCOL_VERSION,
      request_id: message.request_id,
      success: false,
      error: errorMessage(error),
    })
  }
}

async function attachActiveTab(): Promise<void> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  if (typeof tab?.id !== "number") {
    throw new Error("No active browser tab was found.")
  }
  // Chrome may withhold the URL until activeTab is granted by the toolbar
  // gesture. The request still originates from our own popup and always
  // targets the current active tab; reject known privileged URLs, while CDP
  // remains the final authority when the URL is hidden.
  if (tab.url && !isSupportedUrl(tab.url)) {
    throw new Error("Only http, https, and about:blank tabs can be approved.")
  }
  if (attachedTabId !== null && attachedTabId !== tab.id) {
    await detachApprovedTab()
  }
  if (attachedTabId !== tab.id) {
    await chrome.debugger.attach({ tabId: tab.id }, "1.3")
    attachedTabId = tab.id
    await chrome.storage.session.set({ [SESSION.attachedTabId]: tab.id })
    await chrome.debugger.sendCommand({ tabId: tab.id }, "Page.enable")
    await chrome.debugger.sendCommand({ tabId: tab.id }, "DOM.enable")
  }
  lastFrameId = null
  lastError = null
  await updateBadge()
  await sendAttachedStatus()
}

async function detachApprovedTab(): Promise<void> {
  const tabId = attachedTabId
  await clearAttachedTab()
  if (tabId !== null) {
    try {
      await chrome.debugger.detach({ tabId })
    } catch {
      // The user may have already closed the tab or detached DevTools.
    }
  }
}

async function clearAttachedTab(): Promise<void> {
  attachedTabId = null
  lastFrameId = null
  await chrome.storage.session.remove(SESSION.attachedTabId)
  await updateBadge()
  await sendAttachedStatus()
}

async function sendAttachedStatus(): Promise<void> {
  let tab: chrome.tabs.Tab | null = null
  if (attachedTabId !== null) {
    try {
      tab = await chrome.tabs.get(attachedTabId)
    } catch {
      attachedTabId = null
      lastFrameId = null
      await chrome.storage.session.remove(SESSION.attachedTabId)
      await updateBadge()
    }
  }
  send({
    type: "status",
    protocol_version: PROTOCOL_VERSION,
    attached: attachedTabId !== null,
    tab_id: attachedTabId,
    title: tab?.title ?? null,
    url: tab?.url ?? null,
  })
}

async function forgetRelaySession(): Promise<void> {
  await clearReconnectSchedule()
  stopKeepalive()
  if (socket) {
    socket.onclose = null
    socket.close()
    socket = null
  }
  connecting = false
  reconnectAttempt = 0
  nextRetryAt = null
  lastError = null
  await detachApprovedTab()
  await chrome.storage.local.remove([
    STORAGE.relayUrl,
    STORAGE.sessionToken,
  ])
  await updateBadge()
}

async function scheduleReconnect(): Promise<void> {
  const stored = await chrome.storage.local.get([
    STORAGE.relayUrl,
    STORAGE.sessionToken,
  ])
  if (
    typeof stored[STORAGE.relayUrl] !== "string" ||
    typeof stored[STORAGE.sessionToken] !== "string"
  ) {
    reconnectAttempt = 0
    nextRetryAt = null
    return
  }
  await clearReconnectSchedule()
  reconnectAttempt += 1
  const delayMs = reconnectDelayMs(reconnectAttempt)
  nextRetryAt = Date.now() + delayMs
  await chrome.alarms.create(RECONNECT_ALARM, { when: nextRetryAt })
  reconnectTimer = setTimeout(() => {
    void reconnectStoredSession()
  }, delayMs)
}

async function reconnectStoredSession(): Promise<void> {
  if (socket || connecting) return
  const stored = await chrome.storage.local.get([
    STORAGE.relayUrl,
    STORAGE.sessionToken,
  ])
  const relayUrl = stored[STORAGE.relayUrl]
  const sessionToken = stored[STORAGE.sessionToken]
  if (typeof relayUrl !== "string" || typeof sessionToken !== "string") {
    await clearReconnectSchedule()
    return
  }
  connectSocket({
    relayUrl,
    sessionToken,
    reconnecting: true,
  })
}

function startKeepalive(): void {
  stopKeepalive()
  keepaliveTimer = setInterval(() => {
    send({ type: "ping", protocol_version: PROTOCOL_VERSION })
  }, 20_000)
}

function stopKeepalive(): void {
  if (keepaliveTimer !== null) {
    clearInterval(keepaliveTimer)
    keepaliveTimer = null
  }
}

async function clearReconnectSchedule(): Promise<void> {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  nextRetryAt = null
  await chrome.alarms.clear(RECONNECT_ALARM)
}

function send(message: Record<string, unknown>): void {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message))
  }
}

async function publicStatus(): Promise<RelayPublicStatus> {
  const stored = await chrome.storage.local.get(STORAGE.sessionToken)
  const hasSession = typeof stored[STORAGE.sessionToken] === "string"
  let tab: chrome.tabs.Tab | null = null
  if (attachedTabId !== null) {
    try {
      tab = await chrome.tabs.get(attachedTabId)
    } catch {
      await clearAttachedTab()
    }
  }
  const connected = socket?.readyState === WebSocket.OPEN && !connecting
  return {
    connected,
    connecting,
    connectionState: connected
      ? "connected"
      : connecting && reconnectAttempt > 0
        ? "reconnecting"
        : connecting
          ? "connecting"
          : hasSession
            ? "offline"
            : "unpaired",
    hasSession,
    reconnectAttempt,
    nextRetryAt,
    attached: attachedTabId !== null,
    tabId: attachedTabId,
    title: tab?.title ?? null,
    url: tab?.url ?? null,
    error: lastError,
  }
}

async function updateBadge(): Promise<void> {
  const connected = socket?.readyState === WebSocket.OPEN && !connecting
  const text = connecting
    ? "..."
    : connected && attachedTabId !== null
      ? "ON"
      : !connected && attachedTabId !== null
        ? "!"
        : ""
  await chrome.action.setBadgeBackgroundColor({
    color: connecting ? "#d97706" : connected ? "#16a34a" : "#dc2626",
  })
  await chrome.action.setBadgeText({ text })
}

interface BrowserObservation {
  screenshot_base64: string
  viewport: {
    width: number
    height: number
    device_pixel_ratio: number
  }
  elements: unknown[]
  active_url: string | null
  title: string | null
}

async function captureObservation(tabId: number): Promise<BrowserObservation> {
  const target = { tabId }
  const [screenshot, viewportResult, elementsResult] = await Promise.all([
    chrome.debugger.sendCommand(target, "Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
    }),
    chrome.debugger.sendCommand(target, "Runtime.evaluate", {
      expression:
        "({width: Math.max(1, innerWidth), height: Math.max(1, innerHeight), " +
        "device_pixel_ratio: Math.max(0.1, devicePixelRatio || 1), " +
        "active_url: location.href, title: document.title})",
      returnByValue: true,
    }),
    chrome.debugger.sendCommand(target, "Runtime.evaluate", {
      expression: `(${collectInteractiveElements.toString()})()`,
      returnByValue: true,
    }),
  ])
  const screenshotData = readCommandValue(screenshot, "data")
  const viewport = readRuntimeValue(viewportResult)
  const elements = readRuntimeValue(elementsResult)
  if (typeof screenshotData !== "string" || !isRecord(viewport)) {
    throw new Error("Chrome returned an invalid screenshot observation.")
  }
  return {
    screenshot_base64: screenshotData,
    viewport: {
      width: finiteInteger(viewport.width, "viewport width"),
      height: finiteInteger(viewport.height, "viewport height"),
      device_pixel_ratio: finiteNumber(
        viewport.device_pixel_ratio,
        "device pixel ratio",
      ),
    },
    elements: Array.isArray(elements) ? elements.slice(0, 100) : [],
    active_url:
      typeof viewport.active_url === "string" ? viewport.active_url : null,
    title: typeof viewport.title === "string" ? viewport.title : null,
  }
}

async function performAction(
  tabId: number,
  action: Record<string, unknown>,
): Promise<void> {
  const type = requiredString(action.type, "action type")
  const target = { tabId }
  if (type === "navigate") {
    const url = requiredString(action.url, "navigation URL")
    if (!isSupportedUrl(url)) {
      throw new Error("Navigation URL is not supported.")
    }
    await chrome.debugger.sendCommand(target, "Page.navigate", { url })
    await waitForTabReady(tabId, 15_000)
    return
  }
  if (type === "wait") {
    await delay(clampInteger(action.duration_ms, 0, 30_000, 1_000))
    return
  }
  if (type === "keypress") {
    const keys = Array.isArray(action.keys)
      ? action.keys.filter((key): key is string => typeof key === "string")
      : []
    await dispatchKeypress(tabId, keys)
    return
  }
  if (type === "type") {
    if (isRecord(action.target)) {
      const point = await pointInPixels(tabId, action.target)
      await dispatchClick(tabId, point.x, point.y, 1)
    }
    await chrome.debugger.sendCommand(target, "Input.insertText", {
      text: String(action.text ?? ""),
    })
    return
  }
  if (type === "scroll") {
    const viewport = await readViewport(tabId)
    const point = isRecord(action.target)
      ? await pointInPixels(tabId, action.target)
      : { x: viewport.width / 2, y: viewport.height / 2 }
    await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
      type: "mouseWheel",
      x: point.x,
      y: point.y,
      deltaX: finiteNumber(action.delta_x ?? 0, "delta_x") * viewport.width,
      deltaY: finiteNumber(action.delta_y ?? 0, "delta_y") * viewport.height,
    })
    return
  }
  if (type === "drag") {
    if (!isRecord(action.start) || !isRecord(action.end)) {
      throw new Error("Drag requires start and end points.")
    }
    const start = await pointInPixels(tabId, action.start)
    const end = await pointInPixels(tabId, action.end)
    await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
      type: "mousePressed",
      x: start.x,
      y: start.y,
      button: "left",
      buttons: 1,
      clickCount: 1,
    })
    const steps = 8
    for (let step = 1; step <= steps; step += 1) {
      await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x: start.x + ((end.x - start.x) * step) / steps,
        y: start.y + ((end.y - start.y) * step) / steps,
        button: "left",
        buttons: 1,
      })
    }
    await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: end.x,
      y: end.y,
      button: "left",
      buttons: 0,
      clickCount: 1,
    })
    return
  }
  if (!isRecord(action.target)) {
    throw new Error(`${type} requires a target point.`)
  }
  const point = await pointInPixels(tabId, action.target)
  if (type === "click" || type === "double_click") {
    await dispatchClick(tabId, point.x, point.y, type === "double_click" ? 2 : 1)
  } else if (type === "move") {
    await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: point.x,
      y: point.y,
    })
  } else {
    throw new Error(`Unsupported computer action: ${type}`)
  }
}

async function dispatchClick(
  tabId: number,
  x: number,
  y: number,
  clickCount: number,
): Promise<void> {
  const target = { tabId }
  await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
    type: "mousePressed",
    x,
    y,
    button: "left",
    buttons: 1,
    clickCount,
  })
  await chrome.debugger.sendCommand(target, "Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x,
    y,
    button: "left",
    buttons: 0,
    clickCount,
  })
}

async function dispatchKeypress(tabId: number, keys: string[]): Promise<void> {
  if (keys.length === 0) {
    throw new Error("Keypress requires at least one key.")
  }
  const upper = keys.map((key) => key.trim().toUpperCase())
  let modifiers = 0
  if (upper.some((key) => key === "ALT")) modifiers |= 1
  if (upper.some((key) => key === "CTRL" || key === "CONTROL")) modifiers |= 2
  if (
    upper.some(
      (key) => key === "META" || key === "CMD" || key === "COMMAND",
    )
  ) {
    modifiers |= 4
  }
  if (upper.some((key) => key === "SHIFT")) modifiers |= 8
  const modifierNames = new Set([
    "ALT",
    "CTRL",
    "CONTROL",
    "META",
    "CMD",
    "COMMAND",
    "SHIFT",
  ])
  const rawKey = upper.find((key) => !modifierNames.has(key))
  if (!rawKey) {
    throw new Error("Keypress requires a non-modifier key.")
  }
  const key = normalizeKey(rawKey)
  const params = {
    key,
    code: keyCode(key),
    modifiers,
    windowsVirtualKeyCode: virtualKeyCode(key),
  }
  const target = { tabId }
  await chrome.debugger.sendCommand(target, "Input.dispatchKeyEvent", {
    type: "keyDown",
    ...params,
  })
  await chrome.debugger.sendCommand(target, "Input.dispatchKeyEvent", {
    type: "keyUp",
    ...params,
  })
}

function collectInteractiveElements(): unknown[] {
  const selector = [
    "a[href]",
    "button",
    "input",
    "textarea",
    "select",
    "summary",
    "[role='button']",
    "[role='link']",
    "[role='checkbox']",
    "[role='radio']",
    "[role='tab']",
    "[role='menuitem']",
    "[onclick]",
    "[tabindex]",
  ].join(",")
  const width = Math.max(1, window.innerWidth)
  const height = Math.max(1, window.innerHeight)
  const elements: unknown[] = []
  for (const [index, node] of Array.from(
    document.querySelectorAll<HTMLElement>(selector),
  ).entries()) {
    const rect = node.getBoundingClientRect()
    const style = window.getComputedStyle(node)
    if (
      style.visibility === "hidden" ||
      style.display === "none" ||
      rect.width < 2 ||
      rect.height < 2 ||
      rect.right <= 0 ||
      rect.bottom <= 0 ||
      rect.left >= width ||
      rect.top >= height
    ) {
      continue
    }
    const input = node as HTMLInputElement
    const inputType = String(node.getAttribute("type") ?? "").toLowerCase()
    const autocomplete = String(
      node.getAttribute("autocomplete") ?? "",
    ).toLowerCase()
    const sensitive =
      inputType === "password" ||
      inputType === "hidden" ||
      autocomplete === "current-password" ||
      autocomplete === "new-password" ||
      autocomplete === "one-time-code" ||
      autocomplete === "webauthn" ||
      autocomplete.startsWith("cc-")
    const safeValue = sensitive ? "" : String(input.value ?? "")
    const text = String(node.innerText || safeValue).trim().slice(0, 240)
    const label = String(
      node.getAttribute("aria-label") ||
        node.getAttribute("title") ||
        node.getAttribute("placeholder") ||
        text,
    )
      .trim()
      .slice(0, 240)
    const left = Math.max(0, rect.left)
    const top = Math.max(0, rect.top)
    const right = Math.min(width, rect.right)
    const bottom = Math.min(height, rect.bottom)
    elements.push({
      element_id: `dom-${index + 1}`,
      bounds: {
        x: left / width,
        y: top / height,
        width: Math.max(0.000001, (right - left) / width),
        height: Math.max(0.000001, (bottom - top) / height),
      },
      label: label || null,
      role: node.getAttribute("role") || node.tagName.toLowerCase(),
      text: text || null,
      metadata: {
        tag: node.tagName.toLowerCase(),
        input_type: inputType || null,
        autocomplete: autocomplete || null,
        sensitive,
        disabled: Boolean(input.disabled),
        focused: node === document.activeElement,
      },
    })
    if (elements.length >= 100) break
  }
  return elements
}

async function readViewport(
  tabId: number,
): Promise<{ width: number; height: number }> {
  const result = await chrome.debugger.sendCommand(
    { tabId },
    "Runtime.evaluate",
    {
      expression:
        "({width: Math.max(1, innerWidth), height: Math.max(1, innerHeight)})",
      returnByValue: true,
    },
  )
  const value = readRuntimeValue(result)
  if (!isRecord(value)) throw new Error("Could not read browser viewport.")
  return {
    width: finiteInteger(value.width, "viewport width"),
    height: finiteInteger(value.height, "viewport height"),
  }
}

async function pointInPixels(
  tabId: number,
  point: Record<string, unknown>,
): Promise<{ x: number; y: number }> {
  const viewport = await readViewport(tabId)
  return {
    x: clampNumber(point.x, 0, 1) * viewport.width,
    y: clampNumber(point.y, 0, 1) * viewport.height,
  }
}

async function waitForTabReady(tabId: number, timeoutMs: number): Promise<void> {
  const current = await chrome.tabs.get(tabId)
  if (current.status === "complete") {
    await delay(250)
    return
  }
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      chrome.tabs.onUpdated.removeListener(listener)
      resolve()
    }, timeoutMs)
    const listener = (
      updatedTabId: number,
      changeInfo: { status?: string },
    ) => {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        clearTimeout(timer)
        chrome.tabs.onUpdated.removeListener(listener)
        resolve()
      }
    }
    chrome.tabs.onUpdated.addListener(listener)
  })
  await delay(250)
}

function readRuntimeValue(result: object | undefined): unknown {
  if (!isRecord(result) || !isRecord(result.result)) return undefined
  return result.result.value
}

function readCommandValue(
  result: object | undefined,
  key: string,
): unknown {
  return isRecord(result) ? result[key] : undefined
}

function isSupportedUrl(raw: string): boolean {
  if (raw === "about:blank") return true
  try {
    const url = new URL(raw)
    return url.protocol === "http:" || url.protocol === "https:"
  } catch {
    return false
  }
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${field} must be a non-empty string.`)
  }
  return value.trim()
}

function finiteNumber(value: unknown, field: string): number {
  const number = Number(value)
  if (!Number.isFinite(number)) throw new Error(`${field} must be finite.`)
  return number
}

function finiteInteger(value: unknown, field: string): number {
  const number = Math.round(finiteNumber(value, field))
  if (number <= 0 || number > 20_000) {
    throw new Error(`${field} is outside the supported range.`)
  }
  return number
}

function clampNumber(value: unknown, min: number, max: number): number {
  return Math.min(max, Math.max(min, finiteNumber(value, "number")))
}

function clampInteger(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  if (value === undefined || value === null) return fallback
  return Math.round(clampNumber(value, min, max))
}

function normalizeKey(key: string): string {
  const aliases: Record<string, string> = {
    ARROWDOWN: "ArrowDown",
    ARROWLEFT: "ArrowLeft",
    ARROWRIGHT: "ArrowRight",
    ARROWUP: "ArrowUp",
    BACKSPACE: "Backspace",
    DELETE: "Delete",
    END: "End",
    ENTER: "Enter",
    ESC: "Escape",
    ESCAPE: "Escape",
    HOME: "Home",
    PAGEDOWN: "PageDown",
    PAGEUP: "PageUp",
    SPACE: " ",
    TAB: "Tab",
  }
  return aliases[key] ?? (key.length === 1 ? key.toLowerCase() : key)
}

function keyCode(key: string): string {
  if (/^[a-z]$/.test(key)) return `Key${key.toUpperCase()}`
  if (/^[0-9]$/.test(key)) return `Digit${key}`
  if (key === " ") return "Space"
  return key
}

function virtualKeyCode(key: string): number {
  if (key.length === 1) return key.toUpperCase().charCodeAt(0)
  const codes: Record<string, number> = {
    Backspace: 8,
    Tab: 9,
    Enter: 13,
    Escape: 27,
    " ": 32,
    PageUp: 33,
    PageDown: 34,
    End: 35,
    Home: 36,
    ArrowLeft: 37,
    ArrowUp: 38,
    ArrowRight: 39,
    ArrowDown: 40,
    Delete: 46,
  }
  return codes[key] ?? 0
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, durationMs))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message.slice(0, 2_000) : String(error)
}

function errorStatus(error: unknown): RelayPublicStatus {
  lastError = errorMessage(error)
  return {
    connected: false,
    connecting,
    connectionState: "offline",
    hasSession: false,
    reconnectAttempt,
    nextRetryAt,
    attached: attachedTabId !== null,
    tabId: attachedTabId,
    title: null,
    url: null,
    error: lastError,
  }
}

function isAuthenticationError(message: string): boolean {
  const normalized = message.toLowerCase()
  return (
    normalized.includes("invalid") ||
    normalized.includes("expired") ||
    normalized.includes("revoked") ||
    normalized.includes("authentication") ||
    normalized.includes("no longer exists")
  )
}

async function invalidateRelaySession(reason: string): Promise<void> {
  await clearReconnectSchedule()
  stopKeepalive()
  if (socket) {
    socket.onclose = null
    socket.close()
    socket = null
  }
  connecting = false
  reconnectAttempt = 0
  nextRetryAt = null
  lastError = reason
  await chrome.storage.local.remove(STORAGE.sessionToken)
  await detachApprovedTab()
  await updateBadge()
}
