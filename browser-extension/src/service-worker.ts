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
import {
  waitForPageStable,
  type PageStabilityState,
} from "./page-stability.js"

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
const OFFSCREEN_DOCUMENT_PATH = "offscreen.html"
const OFFSCREEN_MESSAGE_TARGET = "xagent-media-offscreen"

let socket: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let keepaliveTimer: ReturnType<typeof setInterval> | null = null
let attachedTabId: number | null = null
let lastFrameId: string | null = null
let lastObservationContextId: number | null = null
let lastError: string | null = null
let connecting = false
let reconnectAttempt = 0
let nextRetryAt: number | null = null
let creatingOffscreenDocument: Promise<void> | null = null

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

interface NavigationPolicy {
  allowlist: string[]
  denylist: string[]
}

interface ActiveNavigationGuard {
  tabId: number
  policy: NavigationPolicy
  blockedReason: string | null
}

let activeNavigationGuard: ActiveNavigationGuard | null = null

chrome.debugger.onEvent.addListener((source, method, params) => {
  if (method === "Fetch.requestPaused") {
    void handlePausedNavigation(source, params)
    return
  }
  if (
    source.tabId === attachedTabId &&
    (method === "Page.frameNavigated" ||
      method === "Page.loadEventFired" ||
      method === "DOM.documentUpdated")
  ) {
    lastFrameId = null
    lastObservationContextId = null
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
    lastObservationContextId = null
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
      observation = await captureObservation(attachedTabId, frameId)
      lastFrameId = frameId
    } else if (message.command === "capture_media") {
      const expectedFrameId = requiredString(
        message.payload.expected_frame_id,
        "expected_frame_id",
      )
      if (lastFrameId !== expectedFrameId) {
        throw new Error(
          "The approved tab changed after the last screenshot. Request a fresh screenshot.",
        )
      }
      if (!isRecord(message.payload.action)) {
        throw new Error("Action payload is invalid.")
      }
      const transferId = requiredString(
        message.payload.transfer_id,
        "transfer_id",
      )
      const artifact = await captureApprovedTabMedia(
        attachedTabId,
        message.payload.action,
      )
      const manifest = await sendMediaArtifactChunks(
        message.request_id,
        transferId,
        artifact,
      )
      send({
        type: "response",
        protocol_version: PROTOCOL_VERSION,
        request_id: message.request_id,
        success: true,
        result: { artifact: manifest },
      })
      return
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
      const action = message.payload.action
      const actionType = String(action.type ?? "")
      const navigationPolicy = parseNavigationPolicy(
        message.payload.navigation_policy,
      )
      observation = await withNavigationGuard(
        attachedTabId,
        navigationPolicy,
        async () => {
          const baseline =
            actionType === "wait" || actionType === "move"
              ? null
              : await readPageStabilityState(attachedTabId as number).catch(
                  () => null,
                )
          await performAction(
            attachedTabId as number,
            action,
            expectedFrameId,
          )
          if (actionType !== "wait" && actionType !== "move") {
            await waitForPageStable(
              () => readPageStabilityState(attachedTabId as number),
              { baseline },
            )
          }
          return captureObservation(attachedTabId as number, frameId)
        },
      )
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
    // Refresh authorization state before the failure response. The server can
    // then distinguish a closed/detached tab (retriable user action) from an
    // ordinary invalid or blocked command.
    try {
      await sendAttachedStatus()
    } catch {
      // Status refresh is best-effort; always return the command failure.
    }
    send({
      type: "response",
      protocol_version: PROTOCOL_VERSION,
      request_id: message.request_id,
      success: false,
      error: errorMessage(error),
    })
  }
}

const MEDIA_CHUNK_BYTES = 256 * 1024

async function sendMediaArtifactChunks(
  requestId: string,
  transferId: string,
  artifact: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const dataBase64 = requiredString(artifact.data_base64, "media data")
  const bytes = base64ToBytes(dataBase64)
  const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(bytes))
  let chunkCount = 0
  for (let offset = 0; offset < bytes.length; offset += MEDIA_CHUNK_BYTES) {
    await waitForSocketCapacity()
    sendRequired({
      type: "media_chunk",
      protocol_version: PROTOCOL_VERSION,
      request_id: requestId,
      transfer_id: transferId,
      chunk_index: chunkCount,
      data_base64: bytesToBase64(
        bytes.subarray(offset, offset + MEDIA_CHUNK_BYTES),
      ),
    })
    chunkCount += 1
  }
  return {
    transfer_id: transferId,
    mime_type: requiredString(artifact.mime_type, "mime_type"),
    media_kind: requiredString(artifact.media_kind, "media_kind"),
    duration_ms: artifact.duration_ms,
    chunk_count: chunkCount,
    size_bytes: bytes.length,
    sha256: bytesToHex(new Uint8Array(digest)),
  }
}

async function waitForSocketCapacity(): Promise<void> {
  while (
    socket?.readyState === WebSocket.OPEN &&
    socket.bufferedAmount > 1024 * 1024
  ) {
    await delay(10)
  }
  if (socket?.readyState !== WebSocket.OPEN) {
    throw new Error("The Xagent relay disconnected during media transfer.")
  }
}

function sendRequired(message: Record<string, unknown>): void {
  if (socket?.readyState !== WebSocket.OPEN) {
    throw new Error("The Xagent relay is not connected.")
  }
  socket.send(JSON.stringify(message))
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

function bytesToBase64(bytes: Uint8Array): string {
  const parts: string[] = []
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    parts.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)),
    )
  }
  return btoa(parts.join(""))
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join(
    "",
  )
}

async function captureApprovedTabMedia(
  tabId: number,
  action: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  if (requiredString(action.type, "action type") !== "capture_media") {
    throw new Error("Media capture requires a capture_media action.")
  }
  const mediaKind = requiredString(action.media_kind, "media_kind")
  if (mediaKind !== "audio" && mediaKind !== "video") {
    throw new Error("media_kind must be audio or video.")
  }
  const durationMs = clampInteger(
    action.duration_ms,
    1_000,
    30_000,
    10_000,
  )
  await ensureOffscreenDocument()
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tabId,
  })
  const response: unknown = await chrome.runtime.sendMessage({
    target: OFFSCREEN_MESSAGE_TARGET,
    type: "capture_media",
    streamId,
    mediaKind,
    durationMs,
  })
  if (!isRecord(response) || response.success !== true) {
    const error =
      isRecord(response) && typeof response.error === "string"
        ? response.error
        : "The browser media recorder did not return a result."
    throw new Error(error)
  }
  if (!isRecord(response.artifact)) {
    throw new Error("The browser media recorder returned an invalid artifact.")
  }
  return response.artifact
}

async function ensureOffscreenDocument(): Promise<void> {
  if (await chrome.offscreen.hasDocument()) return
  if (creatingOffscreenDocument === null) {
    creatingOffscreenDocument = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_DOCUMENT_PATH,
        reasons: [chrome.offscreen.Reason.USER_MEDIA],
        justification:
          "Record bounded audio or video from the user-approved browser tab.",
      })
      .finally(() => {
        creatingOffscreenDocument = null
      })
  }
  await creatingOffscreenDocument
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
  lastObservationContextId = null
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
  lastObservationContextId = null
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

const MAX_OBSERVATION_ELEMENTS = 100
const ISOLATED_WORLD_NAME = "xagent-computer"

type ComputerInputPlatform =
  | "macos"
  | "windows"
  | "linux"
  | "chromeos"
  | "android"
  | "unknown"

interface BrowserObservation {
  screenshot_base64: string
  viewport: {
    width: number
    height: number
    device_pixel_ratio: number
  }
  elements: unknown[]
  elements_truncated: boolean
  element_extraction_failed: boolean
  element_extraction_incomplete: boolean
  active_url: string | null
  title: string | null
  platform: ComputerInputPlatform
  supported_actions: string[]
}

let inputPlatformPromise: Promise<ComputerInputPlatform> | null = null

function computerInputPlatform(): Promise<ComputerInputPlatform> {
  if (!inputPlatformPromise) {
    inputPlatformPromise = chrome.runtime
      .getPlatformInfo()
      .then((info): ComputerInputPlatform => {
        switch (info.os) {
          case "mac":
            return "macos"
          case "win":
            return "windows"
          case "linux":
          case "openbsd":
            return "linux"
          case "cros":
            return "chromeos"
          case "android":
            return "android"
          default:
            return "unknown"
        }
      })
      .catch(() => "unknown")
  }
  return inputPlatformPromise
}

async function captureObservation(
  tabId: number,
  frameToken: string,
): Promise<BrowserObservation> {
  const target = { tabId }
  const frameTreeResult = await chrome.debugger.sendCommand(
    target,
    "Page.getFrameTree",
  )
  const rawFrameTree = readCommandValue(frameTreeResult, "frameTree")
  const frameTree = isRecord(rawFrameTree)
    ? rawFrameTree
    : null
  const mainFrame = frameTree && isRecord(frameTree.frame)
    ? frameTree.frame
    : null
  const mainFrameId =
    mainFrame && typeof mainFrame.id === "string" ? mainFrame.id : ""
  if (!mainFrameId) {
    throw new Error("Chrome returned no main frame for the approved tab.")
  }
  const isolatedWorld = await chrome.debugger.sendCommand(
    target,
    "Page.createIsolatedWorld",
    {
      frameId: mainFrameId,
      worldName: ISOLATED_WORLD_NAME,
      grantUniveralAccess: false,
    },
  )
  const executionContextId = Math.round(
    finiteNumber(
      readCommandValue(isolatedWorld, "executionContextId"),
      "isolated execution context",
    ),
  )
  if (executionContextId <= 0) {
    throw new Error("Chrome returned an invalid isolated execution context.")
  }
  const collectOptions = JSON.stringify({
    frameToken,
    limit: MAX_OBSERVATION_ELEMENTS,
  })
  const [screenshot, viewportResult, elementsResult, platform] = await Promise.all([
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
    chrome.debugger
      .sendCommand(target, "Runtime.evaluate", {
        expression: `(${collectInteractiveElements.toString()})(${collectOptions})`,
        returnByValue: true,
        contextId: executionContextId,
      })
      .catch(() => null),
    computerInputPlatform(),
  ])
  const screenshotData = readCommandValue(screenshot, "data")
  const viewport = readRuntimeValue(viewportResult)
  const extraction = readRuntimeValue(elementsResult)
  if (typeof screenshotData !== "string" || !isRecord(viewport)) {
    throw new Error("Chrome returned an invalid screenshot observation.")
  }
  // A screenshot without element hints is still usable. Preserve the
  // extraction status so the backend can avoid pretending the list is
  // exhaustive while treating coordinate-only targets as ordinary content.
  const extracted = isRecord(extraction) ? extraction : null
  const elements = extracted && Array.isArray(extracted.elements)
    ? extracted.elements
    : []
  lastObservationContextId = executionContextId
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
    elements: elements.slice(0, MAX_OBSERVATION_ELEMENTS),
    elements_truncated: Boolean(extracted?.truncated),
    element_extraction_failed: extracted === null,
    element_extraction_incomplete:
      Boolean(extracted?.incomplete) || frameTreeHasChildren(frameTree),
    active_url:
      typeof viewport.active_url === "string" ? viewport.active_url : null,
    title: typeof viewport.title === "string" ? viewport.title : null,
    platform,
    supported_actions: [
      "screenshot",
      "capture_media",
      "navigate",
      "click",
      "double_click",
      "move",
      "scroll",
      "type",
      "replace_text",
      "keypress",
      "drag",
      "wait",
    ],
  }
}

async function verifyHitTarget(
  tabId: number,
  action: Record<string, unknown>,
  expectedFrameToken: string,
  x: number,
  y: number,
): Promise<void> {
  const elementId = action.target_element_id
  if (typeof elementId !== "string" || !elementId) {
    return
  }
  if (lastObservationContextId === null) {
    throw new Error(
      `Cannot verify ${elementId} because its browser frame is no longer current.`,
    )
  }
  const options = JSON.stringify({
    x,
    y,
    frameToken: expectedFrameToken,
    elementId,
  })
  let result: object | undefined
  try {
    result = await chrome.debugger.sendCommand(
      { tabId },
      "Runtime.evaluate",
      {
        expression: `(${hitTestTarget.toString()})(${options})`,
        contextId: lastObservationContextId,
        returnByValue: true,
      },
    )
  } catch (error) {
    throw new Error(
      `Could not verify ${elementId} before clicking. Request a fresh screenshot.`,
      { cause: error },
    )
  }
  const hit = readRuntimeValue(result)
  if (isRecord(hit) && hit.found === true && hit.matches === true) return
  const obstruction =
    isRecord(hit) && typeof hit.tag === "string" && hit.tag
      ? hit.tag
      : "an unknown element"
  throw new Error(
    `${elementId} is covered by ${obstruction} at the clicked position. ` +
      "Take a fresh screenshot, then dismiss the overlay, scroll the target " +
      "into the clear, or choose a different element.",
  )
}

async function performAction(
  tabId: number,
  action: Record<string, unknown>,
  expectedFrameToken: string,
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
  if (type === "type" || type === "replace_text") {
    if (type === "replace_text" && !isRecord(action.target)) {
      throw new Error("replace_text requires a target.")
    }
    if (isRecord(action.target)) {
      const point = await pointInPixels(tabId, action.target)
      await verifyHitTarget(tabId, action, expectedFrameToken, point.x, point.y)
      await dispatchClick(tabId, point.x, point.y, 1)
    }
    if (type === "replace_text") {
      await selectEditableTargetText(
        tabId,
        action,
        expectedFrameToken,
      )
    }
    const text = String(action.text ?? "")
    await chrome.debugger.sendCommand(target, "Input.insertText", {
      text,
    })
    if (type === "replace_text") {
      await assertEditableTargetText(
        tabId,
        action,
        expectedFrameToken,
        text,
      )
    }
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
    await verifyHitTarget(tabId, action, expectedFrameToken, point.x, point.y)
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

async function selectEditableTargetText(
  tabId: number,
  action: Record<string, unknown>,
  expectedFrameToken: string,
): Promise<void> {
  const elementId = requiredString(
    action.target_element_id,
    "replace_text target element ID",
  )
  if (lastObservationContextId === null) {
    throw new Error(
      `Cannot select ${elementId} because its browser frame is no longer current.`,
    )
  }
  const options = JSON.stringify({
    frameToken: expectedFrameToken,
    elementId,
  })
  const result = await chrome.debugger.sendCommand(
    { tabId },
    "Runtime.evaluate",
    {
      expression: `(${selectEditableTarget.toString()})(${options})`,
      contextId: lastObservationContextId,
      returnByValue: true,
    },
  )
  const selection = readRuntimeValue(result)
  if (!isRecord(selection) || selection.selected !== true) {
    throw new Error(
      `replace_text target ${elementId} is not a selectable text field. ` +
        "Request a fresh screenshot and choose a verified editable element.",
    )
  }
}

async function assertEditableTargetText(
  tabId: number,
  action: Record<string, unknown>,
  expectedFrameToken: string,
  expectedText: string,
): Promise<void> {
  const elementId = requiredString(
    action.target_element_id,
    "replace_text target element ID",
  )
  if (lastObservationContextId === null) {
    throw new Error(
      `Cannot verify ${elementId} because its browser frame is no longer current.`,
    )
  }
  await delay(50)
  const options = JSON.stringify({
    frameToken: expectedFrameToken,
    elementId,
  })
  const result = await chrome.debugger.sendCommand(
    { tabId },
    "Runtime.evaluate",
    {
      expression: `(${readEditableTarget.toString()})(${options})`,
      contextId: lastObservationContextId,
      returnByValue: true,
    },
  )
  const current = readRuntimeValue(result)
  if (
    !isRecord(current) ||
    current.readable !== true ||
    current.text !== expectedText
  ) {
    throw new Error(
      `replace_text did not produce the requested value in ${elementId}. ` +
        "The field may use a custom editor; request a fresh screenshot before " +
        "choosing another action.",
    )
  }
}

function collectInteractiveElements(options: {
  frameToken: string
  limit: number
}): { elements: unknown[]; truncated: boolean; incomplete: boolean } {
  const { frameToken, limit } = options
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
    "[role='textbox']",
    "[onclick]",
    "[tabindex]",
    "[contenteditable='true']",
  ].join(",")
  const width = Math.max(1, window.innerWidth)
  const height = Math.max(1, window.innerHeight)
  const elements: unknown[] = []
  const targets = new Map<string, HTMLElement>()
  let truncated = false
  let incomplete = false

  // Open shadow trees are invisible to a document-level querySelectorAll, so
  // they are walked explicitly to keep controls inside web components usable.
  const collect = (root: Document | ShadowRoot, out: HTMLElement[]): void => {
    try {
      for (const node of Array.from(
        root.querySelectorAll<HTMLElement>(selector),
      )) {
        out.push(node)
      }
      for (const host of Array.from(root.querySelectorAll<HTMLElement>("*"))) {
        if (host.shadowRoot) {
          collect(host.shadowRoot, out)
        } else if (host.tagName.includes("-")) {
          // A custom element without an open shadow root may contain a closed
          // tree, which cannot be inspected even from an isolated world.
          incomplete = true
        }
      }
    } catch {
      // A malformed selector context is skipped rather than failing the frame.
    }
  }
  const candidates: HTMLElement[] = []
  collect(document, candidates)

  for (const node of candidates) {
    if (elements.length >= limit) {
      truncated = true
      break
    }
    const rect = node.getBoundingClientRect()
    const style = window.getComputedStyle(node)
    if (
      style.visibility === "hidden" ||
      style.visibility === "collapse" ||
      style.display === "none" ||
      Number(style.opacity) === 0 ||
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
    const elementId = `dom-${elements.length + 1}`
    targets.set(elementId, node)
    elements.push({
      element_id: elementId,
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
  }
  const state = globalThis as typeof globalThis & {
    __xagentComputerTargets?: Map<string, Map<string, HTMLElement>>
  }
  // Keep only the newest frame. A stale frame cannot be acted on, and bounding
  // this map also releases remote DOM nodes promptly.
  state.__xagentComputerTargets = new Map([[frameToken, targets]])
  return { elements, truncated, incomplete }
}

function selectEditableTarget(options: {
  frameToken: string
  elementId: string
}): { selected: boolean } {
  const state = globalThis as typeof globalThis & {
    __xagentComputerTargets?: Map<string, Map<string, HTMLElement>>
  }
  const target = state.__xagentComputerTargets
    ?.get(options.frameToken)
    ?.get(options.elementId)
  if (!target) return { selected: false }
  target.focus()
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement
  ) {
    target.select()
    return {
      selected:
        target.selectionStart === 0 &&
        target.selectionEnd === target.value.length,
    }
  }
  if (target.isContentEditable) {
    const selection = window.getSelection()
    if (!selection) return { selected: false }
    const range = document.createRange()
    range.selectNodeContents(target)
    selection.removeAllRanges()
    selection.addRange(range)
    return { selected: selection.rangeCount === 1 }
  }
  return { selected: false }
}

function readEditableTarget(options: {
  frameToken: string
  elementId: string
}): { readable: boolean; text?: string } {
  const state = globalThis as typeof globalThis & {
    __xagentComputerTargets?: Map<string, Map<string, HTMLElement>>
  }
  const target = state.__xagentComputerTargets
    ?.get(options.frameToken)
    ?.get(options.elementId)
  if (!target) return { readable: false }
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement
  ) {
    return { readable: true, text: target.value }
  }
  if (target.isContentEditable) {
    return { readable: true, text: target.textContent ?? "" }
  }
  return { readable: false }
}

function hitTestTarget(options: {
  x: number
  y: number
  frameToken: string
  elementId: string
}): { matches: boolean; tag: string | null; found: boolean } {
  const { x, y, frameToken, elementId } = options
  const state = globalThis as typeof globalThis & {
    __xagentComputerTargets?: Map<string, Map<string, HTMLElement>>
  }
  const target = state.__xagentComputerTargets
    ?.get(frameToken)
    ?.get(elementId)
  if (!target) return { matches: false, tag: null, found: false }
  const deepestElementFromPoint = (
    root: Document | ShadowRoot,
  ): Element | null => {
    let hit = root.elementFromPoint(x, y)
    while (hit?.shadowRoot) {
      const nested = hit.shadowRoot.elementFromPoint(x, y)
      if (!nested || nested === hit) break
      hit = nested
    }
    return hit
  }
  const node = deepestElementFromPoint(document)
  if (node === null) return { matches: false, tag: null, found: false }
  const tag = node.tagName ? node.tagName.toLowerCase() : null
  return {
    matches: node === target || target.contains(node),
    tag,
    found: true,
  }
}

function parseNavigationPolicy(value: unknown): NavigationPolicy {
  const policy = isRecord(value) ? value : {}
  return {
    allowlist: normalizeHostPatterns(policy.allowlist),
    denylist: normalizeHostPatterns(policy.denylist),
  }
}

function normalizeHostPatterns(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return Array.from(
    new Set(
      value
        .filter((item): item is string => typeof item === "string")
        .map((item) => item.trim().toLowerCase().replace(/^\.+/, ""))
        .filter(Boolean),
    ),
  )
}

function hostMatches(host: string, patterns: string[]): boolean {
  const candidate = host.trim().toLowerCase().replace(/\.+$/, "")
  return patterns.some(
    (pattern) =>
      candidate === pattern || candidate.endsWith(`.${pattern}`),
  )
}

function navigationBlockReason(
  rawUrl: string,
  policy: NavigationPolicy,
): string | null {
  if (rawUrl === "about:blank") return null
  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    return null
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null
  if (hostMatches(url.hostname, policy.denylist)) {
    return `Navigation to ${url.hostname} is blocked by the configured policy.`
  }
  if (
    policy.allowlist.length > 0 &&
    !hostMatches(url.hostname, policy.allowlist)
  ) {
    return `Navigation to ${url.hostname} is outside the configured allowlist.`
  }
  return null
}

async function withNavigationGuard<T>(
  tabId: number,
  policy: NavigationPolicy,
  operation: () => Promise<T>,
): Promise<T> {
  if (activeNavigationGuard !== null) {
    throw new Error("Another guarded browser action is already running.")
  }
  const guard: ActiveNavigationGuard = {
    tabId,
    policy,
    blockedReason: null,
  }
  activeNavigationGuard = guard
  await chrome.debugger.sendCommand({ tabId }, "Fetch.enable", {
    patterns: [
      {
        urlPattern: "*",
        resourceType: "Document",
        requestStage: "Request",
      },
    ],
  })
  try {
    const result = await operation()
    if (guard.blockedReason) {
      throw new Error(guard.blockedReason)
    }
    return result
  } finally {
    activeNavigationGuard = null
    try {
      await chrome.debugger.sendCommand({ tabId }, "Fetch.disable")
    } catch {
      // Detaching or closing the approved tab also tears down interception.
    }
  }
}

async function handlePausedNavigation(
  source: chrome.debugger.Debuggee,
  rawParams: object | undefined,
): Promise<void> {
  if (typeof source.tabId !== "number" || !isRecord(rawParams)) return
  const requestId = rawParams.requestId
  const request = rawParams.request
  if (typeof requestId !== "string" || !isRecord(request)) return
  const guard = activeNavigationGuard
  const reason =
    guard && guard.tabId === source.tabId
      ? navigationBlockReason(String(request.url ?? ""), guard.policy)
      : null
  if (reason !== null && guard !== null) {
    guard.blockedReason = reason
    // A failed top-level request commits Chrome's error page and destroys the
    // user's current tab state. A synthetic 204 cancels the navigation while
    // leaving the approved page in place; the command still returns the policy
    // error to Xagent below.
    await chrome.debugger.sendCommand(source, "Fetch.fulfillRequest", {
      requestId,
      responseCode: 204,
      responsePhrase: "Blocked by Xagent policy",
      responseHeaders: [
        {
          name: "X-Xagent-Blocked-Navigation",
          value: "1",
        },
      ],
    })
    return
  }
  await chrome.debugger.sendCommand(source, "Fetch.continueRequest", {
    requestId,
  })
}

function frameTreeHasChildren(frameTree: Record<string, unknown> | null): boolean {
  return Boolean(
    frameTree &&
      Array.isArray(frameTree.childFrames) &&
      frameTree.childFrames.length > 0,
  )
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

async function readPageStabilityState(
  tabId: number,
): Promise<PageStabilityState> {
  const result = await chrome.debugger.sendCommand(
    { tabId },
    "Runtime.evaluate",
    {
      expression: `(() => {
        const body = document.body;
        return {
          url: location.href,
          title: document.title,
          readyState: document.readyState,
          domSize: Math.max(0, body?.getElementsByTagName("*").length ?? 0),
          interactiveSize: Math.max(
            0,
            document.querySelectorAll(
              'a[href],button,input,textarea,select,[role="button"],[role="link"],[contenteditable="true"]'
            ).length
          ),
          textSize: Math.max(0, body?.textContent?.length ?? 0),
        };
      })()`,
      returnByValue: true,
    },
  )
  const value = readRuntimeValue(result)
  if (!isRecord(value)) {
    throw new Error("Could not read browser page stability.")
  }
  return {
    url: typeof value.url === "string" ? value.url : "",
    title: typeof value.title === "string" ? value.title : "",
    readyState:
      typeof value.readyState === "string" ? value.readyState : "loading",
    domSize: nonNegativeInteger(value.domSize),
    interactiveSize: nonNegativeInteger(value.interactiveSize),
    textSize: nonNegativeInteger(value.textSize),
  }
}

function readRuntimeValue(result: object | null | undefined): unknown {
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

function nonNegativeInteger(value: unknown): number {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? Math.round(number) : 0
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
