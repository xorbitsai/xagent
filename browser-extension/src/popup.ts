import type { RelayPublicStatus } from "./protocol.js"

const relayUrlInput = element<HTMLInputElement>("relay-url")
const pairingTokenInput = element<HTMLInputElement>("pairing-token")
const connectButton = element<HTMLButtonElement>("connect")
const disconnectButton = element<HTMLButtonElement>("disconnect")
const attachButton = element<HTMLButtonElement>("attach")
const detachButton = element<HTMLButtonElement>("detach")
const statusBadge = element<HTMLSpanElement>("status-badge")
const tabStatus = element<HTMLDivElement>("tab-status")
const errorText = element<HTMLParagraphElement>("error")

void initialize()

connectButton.addEventListener("click", () => {
  void run(async () => {
    const status = await send<RelayPublicStatus>({
      type: "connect",
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
  }, 1_000)
}

function render(status: RelayPublicStatus): void {
  errorText.textContent = status.error ?? ""
  statusBadge.className = "badge"
  if (status.connecting) {
    statusBadge.textContent = "Connecting"
    statusBadge.classList.add("connecting")
  } else if (status.connected) {
    statusBadge.textContent = "Connected"
    statusBadge.classList.add("online")
  } else {
    statusBadge.textContent = "Offline"
  }
  tabStatus.textContent = status.attached
    ? `Approved: ${status.title || status.url || `tab ${status.tabId}`}`
    : "No tab approved"
  attachButton.disabled = !status.connected
  detachButton.disabled = !status.attached
}

async function run(operation: () => Promise<void>): Promise<void> {
  errorText.textContent = ""
  for (const button of [
    connectButton,
    disconnectButton,
    attachButton,
    detachButton,
  ]) {
    button.disabled = true
  }
  try {
    await operation()
  } catch (error) {
    errorText.textContent =
      error instanceof Error ? error.message : String(error)
  } finally {
    render(await send({ type: "get_status" }))
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
