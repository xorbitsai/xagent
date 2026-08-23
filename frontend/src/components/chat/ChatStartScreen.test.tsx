import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({ openFilePreview: vi.fn() }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: null }),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    fileMentionsEnabled: false,
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: vi.fn(),
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    hasAsrModel: false,
    startRecording: vi.fn(),
    status: "idle",
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  generateClientMessageId: () => "client-message-id",
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://upload.local",
}))

import { ChatStartScreen } from "@/components/chat/ChatStartScreen"

const AGENTS_SECTION = "chatPage.sections.chatWithAgents"

// The agents block only renders inside the prompts branch, so every case needs
// a non-empty prompts list to reach it at all.
const renderScreen = (agents?: React.ComponentProps<typeof ChatStartScreen>["agents"]) =>
  render(
    <ChatStartScreen
      hideConfig
      voiceInputEnabled={false}
      title="Support Agent"
      prompts={["Summarize this page"]}
      agents={agents}
      onSend={vi.fn()}
    />
  )

afterEach(() => {
  cleanup()
})

describe("ChatStartScreen agents section", () => {
  it("renders nothing when no agent list is passed", () => {
    renderScreen(undefined)

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
  })

  it("renders nothing for an empty agent list", () => {
    renderScreen([])

    expect(screen.queryByText(AGENTS_SECTION)).toBeNull()
  })

  it("renders the agents that were passed", () => {
    renderScreen([{ id: 7, name: "Billing Agent" }])

    expect(screen.getByText(AGENTS_SECTION)).toBeTruthy()
    expect(screen.getByText("Billing Agent")).toBeTruthy()
  })
})

describe("ChatStartScreen file capability", () => {
  it("forwards the disabled file capability to ChatInput", () => {
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    const { container } = render(
      <ChatStartScreen
        files={[file]}
        filesDisabled
        hideConfig
        voiceInputEnabled={false}
        onSend={vi.fn()}
        title="Session Agent"
      />
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(screen.queryByText("secret.txt")).not.toBeInTheDocument()
  })

  it("owns selected files internally and removes them when no owner is supplied", async () => {
    const { container } = render(
      <ChatStartScreen
        deferFileUpload
        hideConfig
        voiceInputEnabled={false}
        onSend={vi.fn()}
        title="Uncontrolled Agent"
      />
    )
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["draft"], "draft.txt", { type: "text/plain" })

    fireEvent.change(fileInput, { target: { files: [file] } })
    await waitFor(() => expect(screen.getByText("draft.txt")).toBeInTheDocument())

    fireEvent.click(screen.getByTitle("common.remove"))
    await waitFor(() => expect(screen.queryByText("draft.txt")).not.toBeInTheDocument())
  })

  it("preserves externally controlled files and change callback semantics", async () => {
    const observedChanges: File[][] = []

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      const handleFilesChange = (nextFiles: File[]) => {
        observedChanges.push(nextFiles)
        setFiles(nextFiles)
      }
      return (
        <ChatStartScreen
          deferFileUpload
          files={files}
          hideConfig
          voiceInputEnabled={false}
          onFilesChange={handleFilesChange}
          onSend={vi.fn()}
          title="Controlled Agent"
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["owned"], "owned.txt", { type: "text/plain" })

    fireEvent.change(fileInput, { target: { files: [file] } })
    await waitFor(() => expect(screen.getByText("owned.txt")).toBeInTheDocument())
    expect(observedChanges).toEqual([[file]])

    fireEvent.click(screen.getByTitle("common.remove"))
    await waitFor(() => expect(screen.queryByText("owned.txt")).not.toBeInTheDocument())
    expect(observedChanges).toEqual([[file], []])
  })
})
