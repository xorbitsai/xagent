/**
 * Copy text to the clipboard, tolerating non-secure contexts (plain HTTP on a
 * non-localhost host) and older browsers where ``navigator.clipboard`` is
 * undefined -- accessing it directly there throws. Falls back to a hidden
 * textarea + ``execCommand('copy')`` and reports success so callers can toast
 * accordingly. Never throws.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Secure-context API unavailable or blocked; fall through to the legacy path.
  }
  if (typeof document === "undefined") return false
  try {
    const textarea = document.createElement("textarea")
    textarea.value = text
    textarea.style.position = "fixed"
    textarea.style.opacity = "0"
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.select()
    const ok = document.execCommand("copy")
    document.body.removeChild(textarea)
    return ok
  } catch {
    return false
  }
}
