export function getBrowserLocationOrigin(): string {
  if (typeof window === "undefined") {
    return ""
  }

  try {
    return window.location?.origin ?? ""
  } catch {
    return ""
  }
}

export function getBrowserLocationHostname(): string {
  if (typeof window === "undefined") {
    return ""
  }

  try {
    return window.location?.hostname ?? ""
  } catch {
    return ""
  }
}
