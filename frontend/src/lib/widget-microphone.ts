// Whether the page this code runs in can use the microphone at all. The chat
// pages ask it before offering a voice call, so the call button is not shown
// where starting a call could only fail.
//
// A page that is not a secure context never gets the microphone. Where the
// browser exposes a permissions policy query (document.permissionsPolicy or
// the older document.featurePolicy), a policy that denies "microphone" for
// this page, for example a parent frame that did not delegate it, also hides
// the button. Browsers without that query are not blocked here: the call then
// reports that the microphone could not be used. The visitor's own earlier
// choice in the browser permission prompt is deliberately not read, so a
// visitor who once refused still sees the button and can start a call.
export function pageMayUseMicrophone(): boolean {
  if (typeof window === "undefined" || window.isSecureContext !== true) return false
  const doc = document as Document & {
    permissionsPolicy?: { allowsFeature?: (feature: string) => boolean }
    featurePolicy?: { allowsFeature?: (feature: string) => boolean }
  }
  const policy = doc.permissionsPolicy ?? doc.featurePolicy
  if (!policy || typeof policy.allowsFeature !== "function") return true
  try {
    return policy.allowsFeature("microphone") === true
  } catch {
    return true
  }
}
