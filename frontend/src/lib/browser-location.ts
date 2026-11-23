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

export function getBrowserLocationHash(): string {
  if (typeof window === "undefined") {
    return ""
  }

  try {
    return window.location?.hash ?? ""
  } catch {
    return ""
  }
}

export function getBrowserLocationPathname(): string {
  if (typeof window === "undefined") {
    return ""
  }

  try {
    return window.location?.pathname ?? ""
  } catch {
    return ""
  }
}

export function getBrowserLocationSearch(): string {
  if (typeof window === "undefined") {
    return ""
  }

  try {
    return window.location?.search ?? ""
  } catch {
    return ""
  }
}

// Navigates without adding a history entry, so Back does not return to the
// page being left.
export function replaceBrowserLocation(url: string): void {
  if (typeof window === "undefined") {
    return
  }

  window.location.replace(url)
}
