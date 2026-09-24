export interface GooglePickerDocument {
  id: string
  name?: string
  mimeType?: string
  sizeBytes?: string
  modifiedDate?: string
  resourceKey?: string
}

interface GooglePickerData {
  action?: string
  docs?: GooglePickerDocument[]
}

interface GooglePickerBuilder {
  setAppId(appId: string): GooglePickerBuilder
  setDeveloperKey(apiKey: string): GooglePickerBuilder
  setOAuthToken(token: string): GooglePickerBuilder
  setOrigin(origin: string): GooglePickerBuilder
  addView(view: unknown): GooglePickerBuilder
  enableFeature(feature: unknown): GooglePickerBuilder
  setCallback(callback: (data: GooglePickerData) => void): GooglePickerBuilder
  build(): { setVisible(visible: boolean): void }
}

interface GooglePickerNamespace {
  Action: { PICKED: string; CANCEL: string; ERROR: string }
  Feature: { MULTISELECT_ENABLED: string }
  Response: { ACTION: string; DOCUMENTS: string }
  ViewId: { DOCS: string }
  DocsView: new (viewId: string) => {
    setIncludeFolders(include: boolean): unknown
    setSelectFolderEnabled(enabled: boolean): unknown
  }
  PickerBuilder: new () => GooglePickerBuilder
}

interface GoogleApiLoader {
  load(
    api: string,
    options: { callback: () => void; onerror?: () => void },
  ): void
}

declare global {
  interface Window {
    gapi?: GoogleApiLoader
    google?: { picker?: GooglePickerNamespace }
  }
}

const GOOGLE_API_SCRIPT = "https://apis.google.com/js/api.js"
let scriptPromise: Promise<void> | undefined
let pickerPromise: Promise<GooglePickerNamespace> | undefined

function loadGoogleApiScript(): Promise<void> {
  if (scriptPromise) return scriptPromise

  scriptPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(
      `script[src="${GOOGLE_API_SCRIPT}"]`,
    )
    if (existingScript) {
      if (window.gapi) {
        resolve()
      } else {
        existingScript.addEventListener("load", () => resolve(), { once: true })
        existingScript.addEventListener("error", () => reject(new Error("Google API script failed to load")), { once: true })
      }
      return
    }

    const script = document.createElement("script")
    script.src = GOOGLE_API_SCRIPT
    script.async = true
    script.defer = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error("Google API script failed to load"))
    document.head.appendChild(script)
  })

  return scriptPromise
}

async function loadPicker(): Promise<GooglePickerNamespace> {
  if (pickerPromise) return pickerPromise

  pickerPromise = loadGoogleApiScript().then(
    () =>
      new Promise<GooglePickerNamespace>((resolve, reject) => {
        if (!window.gapi) {
          reject(new Error("Google API loader is unavailable"))
          return
        }

        window.gapi.load("picker", {
          callback: () => {
            const picker = window.google?.picker
            if (picker) resolve(picker)
            else reject(new Error("Google Picker API is unavailable"))
          },
          onerror: () => reject(new Error("Google Picker API failed to load")),
        })
      }),
  )

  try {
    return await pickerPromise
  } catch (error) {
    pickerPromise = undefined
    throw error
  }
}

export async function openGoogleDrivePicker(options: {
  apiKey: string
  appId: string
  accessToken: string
}): Promise<GooglePickerDocument[]> {
  const picker = await loadPicker()

  return new Promise((resolve, reject) => {
    const view = new picker.DocsView(picker.ViewId.DOCS)
    view.setIncludeFolders(false)
    view.setSelectFolderEnabled(false)

    const builder = new picker.PickerBuilder()
      .setAppId(options.appId)
      .setDeveloperKey(options.apiKey)
      .setOAuthToken(options.accessToken)
      .setOrigin(window.location.protocol + "//" + window.location.host)
      .addView(view)
      .enableFeature(picker.Feature.MULTISELECT_ENABLED)
      .setCallback((data) => {
        const action = data.action
        if (action === picker.Action.PICKED) {
          resolve(data.docs || [])
        } else if (action === picker.Action.ERROR) {
          reject(new Error("Google Picker returned an error"))
        } else if (action === picker.Action.CANCEL) {
          resolve([])
        }
      })

    builder.build().setVisible(true)
  })
}
