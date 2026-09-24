export {}

interface GooglePickerDocument {
  id?: string
  name?: string
  mimeType?: string
  resourceKey?: string
  sizeBytes?: number
  lastEditedUtc?: string
}

interface GooglePickerData {
  action?: string
  docs?: GooglePickerDocument[]
}

interface GooglePickerView {
  setIncludeFolders(includeFolders: boolean): GooglePickerView
  setSelectFolderEnabled(selectFolderEnabled: boolean): GooglePickerView
}

interface GooglePickerBuilder {
  setDeveloperKey(developerKey: string): GooglePickerBuilder
  setAppId(appId: string): GooglePickerBuilder
  setOAuthToken(accessToken: string): GooglePickerBuilder
  addView(view: GooglePickerView): GooglePickerBuilder
  enableFeature(feature: string): GooglePickerBuilder
  setCallback(callback: (data: GooglePickerData) => void): GooglePickerBuilder
  build(): { setVisible(visible: boolean): void }
}

interface GooglePickerNamespace {
  Action: { PICKED: string }
  Feature: { MULTISELECT_ENABLED: string }
  ViewId: { DOCS: string }
  DocsView: new (viewId: string) => GooglePickerView
  PickerBuilder: new () => GooglePickerBuilder
}

interface GoogleApiNamespace {
  load(
    library: string,
    options: { callback: () => void; onerror?: () => void },
  ): void
}

declare global {
  interface Window {
    gapi?: GoogleApiNamespace
    google?: { picker?: GooglePickerNamespace }
  }
}
