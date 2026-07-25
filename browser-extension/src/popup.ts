import type { RelayPublicStatus } from "./protocol.js"

const setupCodeInput = element<HTMLTextAreaElement>("setup-code")
const relayUrlInput = element<HTMLInputElement>("relay-url")
const pairingTokenInput = element<HTMLInputElement>("pairing-token")
const pairingSection = element<HTMLElement>("pairing-section")
const connectButton = element<HTMLButtonElement>("connect")
const disconnectButton = element<HTMLButtonElement>("disconnect")
const attachButton = element<HTMLButtonElement>("attach")
const detachButton = element<HTMLButtonElement>("detach")
const statusBadge = element<HTMLSpanElement>("status-badge")
const connectionTitle = element<HTMLElement>("connection-title")
const connectionDetail = element<HTMLParagraphElement>("connection-detail")
const retryDetail = element<HTMLParagraphElement>("retry-detail")
const tabStatus = element<HTMLDivElement>("tab-status")
const tabUrl = element<HTMLDivElement>("tab-url")
const tabIndicator = element<HTMLSpanElement>("tab-indicator")
const errorText = element<HTMLParagraphElement>("error")
const steps = [
  element<HTMLLIElement>("step-pair"),
  element<HTMLLIElement>("step-connect"),
  element<HTMLLIElement>("step-approve"),
]
let busy = false

void initialize()

connectButton.addEventListener("click", () => {
  void run(async () => {
    const status = await send<RelayPublicStatus>({
      type: "connect",
      setupCode: setupCodeInput.value,
      relayUrl: relayUrlInput.value,
      pairingToken: pairingTokenInput.value,
    })
    pairingTokenInput.value = ""
    render(status)
  })
})

disconnectButton.addEventListener("click", () => {
  void run(async () => render(await send({ type: "disconnect" })))
})

attachButton.addEventListener("click", () => {
  void run(async () => render(await send({ type: "attach_active_tab" })))
})

detachButton.addEventListener("click", () => {
  void run(async () => render(await send({ type: "detach_tab" })))
})

async function initialize(): Promise<void> {
  const stored = await chrome.storage.local.get("relayUrl")
  if (typeof stored.relayUrl === "string") {
    relayUrlInput.value = stored.relayUrl
  }
  render(await send({ type: "get_status" }))
  window.setInterval(() => {
    void send<RelayPublicStatus>({ type: "get_status" }).then(render)
  }, 500)
}

function render(status: RelayPublicStatus): void {
  const state = status.connectionState
  errorText.textContent = status.error ?? ""
  statusBadge.className = "badge"
  if (state === "connected") {
    statusBadge.textContent = "Connected"
    statusBadge.classList.add("online")
    connectionTitle.textContent = "Relay ready"
    connectionDetail.textContent = status.attached
      ? "Xagent can use the approved tab."
      : "Approve the browser tab Xagent should use."
  } else if (state === "connecting" || state === "reconnecting") {
    statusBadge.textContent =
      state === "reconnecting" ? "Reconnecting" : "Connecting"
    statusBadge.classList.add("connecting")
    connectionTitle.textContent =
      state === "reconnecting" ? "Restoring relay" : "Connecting relay"
    connectionDetail.textContent = "The saved session will reconnect automatically."
  } else if (state === "offline") {
    statusBadge.textContent = "Offline"
    statusBadge.classList.add("offline")
    connectionTitle.textContent = "Relay temporarily offline"
    connectionDetail.textContent =
      "Your session is saved. Reconnect now or wait for the automatic retry."
  } else {
    statusBadge.textContent = "Not paired"
    connectionTitle.textContent = "Pairing required"
    connectionDetail.textContent =
      "Create a pairing setup in Xagent Settings, then paste it below."
  }

  const retrySeconds =
    status.nextRetryAt === null
      ? null
      : Math.max(0, Math.ceil((status.nextRetryAt - Date.now()) / 1_000))
  retryDetail.hidden = retrySeconds === null
  retryDetail.textContent =
    retrySeconds === null
      ? ""
      : `Automatic retry ${status.reconnectAttempt} in ${retrySeconds}s`

  if (status.hasSession) {
    setupCodeInput.value = ""
  }
  pairingSection.hidden = status.hasSession
  connectButton.textContent = status.hasSession ? "Reconnect now" : "Pair and connect"
  connectButton.hidden = state === "connected"
  disconnectButton.hidden =
    !status.hasSession && !status.connected && !status.connecting && !status.attached

  tabStatus.textContent = status.attached
    ? status.title || `Tab ${status.tabId}`
    : "No tab approved"
  tabUrl.textContent = status.attached ? status.url ?? "" : ""
  tabIndicator.textContent = status.attached ? "ON" : "OFF"
  tabIndicator.className = status.attached
    ? "tab-indicator online"
    : "tab-indicator"

  attachButton.disabled = busy || !status.connected
  detachButton.disabled = busy || !status.attached
  detachButton.hidden = !status.attached
  connectButton.disabled = busy || status.connecting
  disconnectButton.disabled = busy

  renderSteps(status)
}

function renderSteps(status: RelayPublicStatus): void {
  const completed = [
    status.hasSession,
    status.connected,
    status.connected && status.attached,
  ]
  const activeIndex = completed.findIndex((value) => !value)
  steps.forEach((step, index) => {
    step.classList.toggle("complete", completed[index] ?? false)
    step.classList.toggle(
      "active",
      activeIndex === index || (activeIndex === -1 && index === steps.length - 1),
    )
  })
}

async function run(operation: () => Promise<void>): Promise<void> {
  errorText.textContent = ""
  busy = true
  setButtonsDisabled(true)
  try {
    await operation()
  } catch (error) {
    errorText.textContent =
      error instanceof Error ? error.message : String(error)
  } finally {
    busy = false
    render(await send({ type: "get_status" }))
  }
}

function setButtonsDisabled(disabled: boolean): void {
  for (const button of [
    connectButton,
    disconnectButton,
    attachButton,
    detachButton,
  ]) {
    button.disabled = disabled
  }
}

function send<T = RelayPublicStatus>(
  message: Record<string, unknown>,
): Promise<T> {
  return chrome.runtime.sendMessage(message) as Promise<T>
}

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id)
  if (!value) throw new Error(`Missing popup element: ${id}`)
  return value as T
}
