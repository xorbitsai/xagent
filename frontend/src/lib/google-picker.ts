export interface SanitizedGooglePickerDocument {
  id: string
  name?: string
  mimeType?: string
  resourceKey?: string
  sizeBytes?: number
}

export function sanitizeGooglePickerDocuments(
  value: unknown,
): SanitizedGooglePickerDocument[] {
  if (!Array.isArray(value)) return []

  return value.flatMap(item => {
    if (!item || typeof item !== "object") return []
    const document = item as Record<string, unknown>
    if (typeof document.id !== "string" || document.id.trim() === "") return []

    return [{
      id: document.id,
      name: typeof document.name === "string" ? document.name : undefined,
      mimeType: typeof document.mimeType === "string" ? document.mimeType : undefined,
      resourceKey: typeof document.resourceKey === "string" ? document.resourceKey : undefined,
      sizeBytes: typeof document.sizeBytes === "number"
        && Number.isFinite(document.sizeBytes)
        && document.sizeBytes >= 0
        ? document.sizeBytes
        : undefined,
    }]
  })
}

let googlePickerScriptPromise: Promise<void> | null = null
const GOOGLE_PICKER_LOAD_TIMEOUT_MS = 15_000

export function loadGooglePicker(): Promise<void> {
  if (googlePickerScriptPromise) return googlePickerScriptPromise

  googlePickerScriptPromise = new Promise<void>((resolve, reject) => {
    let settled = false
    const script = document.createElement("script")
    const timeoutId = setTimeout(
      () => fail(new Error("Google Picker failed to load")),
      GOOGLE_PICKER_LOAD_TIMEOUT_MS,
    )
    const fail = (error: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timeoutId)
      if (script.parentNode) script.parentNode.removeChild(script)
      reject(error)
    }
    const succeed = () => {
      if (settled) return
      settled = true
      clearTimeout(timeoutId)
      resolve()
    }
    const loadPicker = () => {
      if (!window.gapi) {
        fail(new Error("Google Picker API is unavailable"))
        return
      }
      window.gapi.load("picker", {
        callback: succeed,
        onerror: () => fail(new Error("Google Picker failed to load")),
      })
    }

    if (window.gapi) {
      loadPicker()
      return
    }

    script.src = "https://apis.google.com/js/api.js"
    script.async = true
    script.onload = loadPicker
    script.onerror = () => fail(new Error("Google Picker failed to load"))
    document.head.appendChild(script)
  }).catch(error => {
    googlePickerScriptPromise = null
    throw error
  })

  return googlePickerScriptPromise
}
