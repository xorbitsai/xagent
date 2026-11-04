import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"
import OnboardingPage from "./page"

const routerPush = vi.hoisted(() => vi.fn())
const routerReplace = vi.hoisted(() => vi.fn())
const authUser = vi.hoisted(() => ({ id: "user-a", username: "Shulei" as string | undefined }))
// Lets exactly one test diverge readAuthSessionSnapshot()'s userId from
// authUser.id (which every other test keeps them equal to by default, so
// none of them can tell whether wizardUserIdRef's initial capture actually
// reads readAuthSessionSnapshot() at mount vs. user?.id). undefined (reset
// every beforeEach) means "no override, fall back to authUser.id" - the
// same behavior every other test already relies on.
const sessionUserIdOverride = vi.hoisted(() => ({ current: undefined as string | null | undefined }))
const apiRequestMock = vi.hoisted(() => vi.fn())
const updateUserPreferencesMock = vi.hoisted(() => vi.fn())
const hireAgentFromTemplateMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const markOnboardingSaveEscapedMock = vi.hoisted(() => vi.fn())
const i18nState = vi.hoisted(() => ({ locale: "en" as "en" | "zh" }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
}))

vi.mock("sonner", () => ({
  toast: { error: toastErrorMock },
}))

// Spreads into a NEW object on every call, rather than returning `authUser`
// itself - the real AuthProvider replaces its `user` value via React state
// (setProjection in auth-context.tsx) on an identity swap, it doesn't mutate
// an existing object in place. Returning the same object reference here
// would let a stale closure's `user.id` read "live" through that shared
// reference, silently hiding bugs where a closure captured from an earlier
// render can't actually observe a later identity change.
vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: { ...authUser } }),
}))

// readAuthSessionSnapshot() is called directly (not via a React hook) at
// every identity-swap guard checkpoint, precisely so those checks read live
// storage instead of a value that can only update while this page stays
// mounted - see wizardUserIdRef's/currentSessionUserId's comments in
// page.tsx. Backed by the SAME `authUser` object as the useAuth() mock
// above (read fresh on every call, not spread into a snapshot at mock-setup
// time) so mutating `authUser.id` mid-test - including after this
// component has unmounted - is visible here too.
vi.mock("@/lib/auth-cache", () => ({
  readAuthSessionSnapshot: () => ({
    userId: sessionUserIdOverride.current !== undefined ? sessionUserIdOverride.current : authUser.id,
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation(i18nState.locale, key as TranslationKey, vars),
    locale: i18nState.locale,
  }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/user-preferences", () => ({
  updateUserPreferences: updateUserPreferencesMock,
  fetchUserPreferences: vi.fn(),
  markOnboardingSaveEscaped: markOnboardingSaveEscapedMock,
}))

vi.mock("@/lib/hire-agent", () => ({
  hireAgentFromTemplate: hireAgentFromTemplateMock,
}))

const TEMPLATES = [
  {
    id: "marketing-social-media-content-manager",
    name: "Maya",
    category: "Marketing",
    description: "Turns briefs into posts.",
    features: [],
    persona: { name: "Maya", role: "Social Media Content Manager", intro: "Hi", kickoff_questions: [] },
    connections: [{ name: "LinkedIn" }, { name: "Facebook Pages" }, { name: "Instagram" }, { name: "Google Drive" }],
    setup_time: "5 min",
    tags: [],
    author: "xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 0,
  },
  {
    id: "support-inbox-manager",
    name: "Ellie",
    category: "Support",
    description: "Keeps your inbox triaged.",
    features: [],
    persona: { name: "Ellie", role: "Inbox Manager", intro: "Hi", kickoff_questions: [] },
    connections: [],
    setup_time: "5 min",
    tags: [],
    author: "xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 0,
  },
  {
    id: "sales-meeting-agent",
    name: "Kevin",
    category: "Sales",
    description: "Writes up your meetings.",
    features: [],
    persona: { name: "Kevin", role: "Meeting Agent", intro: "Hi", kickoff_questions: [] },
    connections: [],
    setup_time: "5 min",
    tags: [],
    author: "xagent",
    version: "1.0",
    views: 0,
    likes: 0,
    used_count: 0,
  },
]

async function startOnboarding() {
  render(<OnboardingPage />)
  await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())
}

describe("OnboardingPage", () => {
  beforeEach(() => {
    routerPush.mockClear()
    routerReplace.mockClear()
    i18nState.locale = "en"
    authUser.id = "user-a"
    authUser.username = "Shulei"
    sessionUserIdOverride.current = undefined
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => TEMPLATES })
    updateUserPreferencesMock.mockReset()
    updateUserPreferencesMock.mockResolvedValue({ ok: true })
    hireAgentFromTemplateMock.mockReset()
    hireAgentFromTemplateMock.mockResolvedValue({ taskId: 42 })
    toastErrorMock.mockReset()
    markOnboardingSaveEscapedMock.mockReset()
  })

  afterEach(cleanup)

  it("starts with needs and reaches hiring in two steps, with inline voice settings", async () => {
    await startOnboarding()
    expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled()
    expect(screen.queryByText("Let's go")).not.toBeInTheDocument()
    const need = screen.getByRole("button", { name: "What you need" })
    expect(need).toHaveAttribute("aria-current", "step")
    expect(screen.getByRole("button", { name: "Your team" })).toBeDisabled()
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Write up my meetings"))
    fireEvent.click(screen.getByText("Continue"))
    const heading = await screen.findByRole("heading", { name: "Meet your AI team." })
    expect(heading).toHaveFocus()
    expect(screen.getByRole("button", { name: "Your team" })).toHaveAttribute("aria-current", "step")
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start with Maya" })).toBeEnabled()
    expect(screen.getByText(/Maya can connect to/)).toHaveTextContent("Instagram")
    fireEvent.click(screen.getByRole("button", { name: /^Kevin Meeting Agent/ }))
    expect(screen.getByRole("button", { name: "Start with Kevin" })).toBeEnabled()
    expect(screen.getByText("Kevin needs no connected accounts to start.")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Voice: Professional" }))
    fireEvent.click(screen.getByRole("button", { name: "Friendly" }))
    expect(screen.getByRole("button", { name: "Voice: Friendly" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("Friendly sounds like")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Start with Kevin" }))
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task/42"))
    expect(updateUserPreferencesMock).toHaveBeenCalledWith(expect.objectContaining({
      department: "marketing", goals: ["social", "meetings"], voice: "friendly",
    }))
    expect(hireAgentFromTemplateMock).toHaveBeenCalledWith(expect.objectContaining({
      templateId: "sales-meeting-agent",
      welcome: expect.objectContaining({ description: "Writes up your meetings." }),
    }))
  })

  it.each(["network", "http", "malformed"])("lets users retry a %s catalog failure instead of spinning forever", async (failure) => {
    if (failure === "network") apiRequestMock.mockRejectedValueOnce(new Error("offline"))
    if (failure === "http") apiRequestMock.mockResolvedValueOnce({ ok: false })
    if (failure === "malformed") apiRequestMock.mockResolvedValueOnce({ ok: true, json: async () => ({}) })
    await startOnboarding()
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load your teammates")
    expect(screen.getByRole("button", { name: /^Start with/ })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByRole("button", { name: "Start with Maya" })).toBeEnabled()
  })

  it("handles an empty catalog with a disabled hire button and a way out", async () => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => [] })
    await startOnboarding()
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    expect(await screen.findByText(/No recommended teammates/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^Start with/ })).toBeDisabled()
    fireEvent.click(screen.getByText("Take me to the catalogue instead"))
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/templates"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
  })

  it("keeps goals and the selected voice when navigating back with the progress rail", async () => {
    await startOnboarding()
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    fireEvent.click(screen.getByRole("button", { name: "Voice: Professional" }))
    fireEvent.click(screen.getByRole("button", { name: "Concise" }))
    fireEvent.click(screen.getByRole("button", { name: "What you need" }))
    expect(screen.getByRole("heading", { name: /take off your plate/ })).toHaveFocus()
    expect(screen.getByText("Post on social media")).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(screen.getByText("Continue"))
    expect(screen.getByRole("button", { name: "Voice: Concise" })).toBeInTheDocument()
  })

  it("localizes the two-step flow and inline voice controls in Chinese", async () => {
    i18nState.locale = "zh"
    render(<OnboardingPage />)
    fireEvent.click(screen.getByText("发布社交媒体内容"))
    fireEvent.click(screen.getByText("继续"))
    expect(await screen.findByRole("button", { name: "语气: 专业" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "你的团队" })).toHaveAttribute("aria-current", "step")
  })

  it("greets the real authenticated user by username, not a hardcoded placeholder", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getAllByText("Shulei").length).toBeGreaterThan(0))
    expect(screen.queryByText(/Gerard Santos/)).not.toBeInTheDocument()
  })

  it("renders the account badge's initial from the first Unicode code point, not the first UTF-16 unit", async () => {
    authUser.username = "\u{1F600}unicode"

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getAllByText(/unicode/).length).toBeGreaterThan(0))

    const badge = document.querySelector(".ob-acct span")
    expect(badge).toHaveTextContent("\u{1F600}")
  })

  it("falls back to a generic name when the user has no username", async () => {
    authUser.username = undefined
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())
    expect(screen.getAllByText("there").length).toBeGreaterThan(0)
  })

  it("omits industry entirely (neither a value nor a clear) when the department was never chosen", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalled())
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).not.toHaveProperty("industry")
    expect(call).not.toHaveProperty("department")
  })

  it("waits for the preferences save to resolve before navigating away on Skip setup", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(expect.objectContaining({ onboarded: true }))
    // The save is still pending - navigation must not have happened yet.
    expect(routerReplace).not.toHaveBeenCalled()

    await act(async () => {
      resolveSave({ ok: true })
    })

    expect(routerReplace).toHaveBeenCalledWith("/templates")
  })

  it("shows a Saving… label and aria-busy on the header exit button while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    const button = (await screen.findAllByText("Saving…")).find((node) => node.classList.contains("ob-exit"))!
    expect(button.closest("button")).toHaveAttribute("aria-busy", "true")
    expect(screen.queryByText("Skip setup")).not.toBeInTheDocument()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("shows a Saving… label and aria-busy on the goals-step skip button while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    // Grabbed by reference before the click, not re-queried by text after -
    // the header exit button shows the same "Saving…" label at the same
    // time (both share the `launching` state), so a text query afterward
    // would match 2 elements.
    const button = screen.getByText("Skip — show me everyone").closest("button")!
    fireEvent.click(button)

    await waitFor(() => expect(button).toHaveTextContent("Saving…"))
    expect(button).toHaveAttribute("aria-busy", "true")

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("shows a Saving… label and aria-busy on the done-step skip button while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // Grabbed by reference before the click - see the goals-step test's
    // comment for why (the header exit button shares the same label at
    // the same time).
    const button = screen.getByText("Take me to the catalogue instead").closest("button")!
    fireEvent.click(button)

    await waitFor(() => expect(button).toHaveTextContent("Saving…"))
    expect(button).toHaveAttribute("aria-busy", "true")

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("ignores a second click on Skip setup while the first save is still in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    const skipButton = screen.getByText("Skip setup")
    fireEvent.click(skipButton)
    fireEvent.click(skipButton)
    fireEvent.click(skipButton)

    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)
    expect(skipButton.closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("disables the goal chips while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))

    fireEvent.click(screen.getByText("Skip — show me everyone"))

    expect(screen.getByText("Post on social media").closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("disables the voice chips while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: /Voice:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Playful" }))

    fireEvent.click(screen.getByText("Skip setup"))

    expect(screen.getByRole("button", { name: "Playful" }).closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("does not mark the save-escape flag if the component unmounts before the 2nd failure's navigation would fire", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: false })
    let resolveSecondSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValueOnce(
      new Promise((resolve) => { resolveSecondSave = resolve })
    )

    const { unmount } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText("Skip setup"))
    unmount()

    await act(async () => {
      resolveSecondSave({ ok: false })
    })

    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("does not navigate and shows an error toast when the Skip setup save fails", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })

    expect(toastErrorMock).toHaveBeenCalledWith("Couldn't save your setup — please try again.")
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("hides the team card's persona avatar from the accessibility tree so its name isn't announced twice", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText("Maya")).toBeInTheDocument())
    const card = screen.getByText("Maya").closest("button")!
    expect(card.querySelector('[aria-hidden="true"]')).toHaveTextContent("M")
  })

  it("hires the selected agent and navigates to /task/{taskId} on launch", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(hireAgentFromTemplateMock.mock.calls[0][0]).toEqual(
      expect.objectContaining({ templateId: "marketing-social-media-content-manager" })
    )
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  it("passes an abortIfIdentityChanged callback to hireAgentFromTemplate that reflects the identity swap guard live", async () => {
    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    const { abortIfIdentityChanged } = hireAgentFromTemplateMock.mock.calls[0][0]
    expect(typeof abortIfIdentityChanged).toBe("function")
    // Identity hasn't changed yet - the callback must say so.
    expect(abortIfIdentityChanged()).toBe(false)

    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    // Same callback reference from the same hireAgentFromTemplate call -
    // it must now report the swap live, proving it reads live session
    // storage directly (readAuthSessionSnapshot()), not a value captured
    // once when hireAgentFromTemplate was originally invoked.
    expect(abortIfIdentityChanged()).toBe(true)
  })

  it("keeps reporting a swap live from the abortIfIdentityChanged callback even after this page has unmounted", async () => {
    const { unmount } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    const { abortIfIdentityChanged } = hireAgentFromTemplateMock.mock.calls[0][0]
    expect(abortIfIdentityChanged()).toBe(false)

    unmount()
    authUser.id = "user-b"

    // No rerender happens (and none is possible) after unmount - a ref
    // kept fresh only by a mounted component's effect could never see
    // this, but a live storage read still can.
    expect(abortIfIdentityChanged()).toBe(true)
  })

  it("still attempts the onboarded save (but not navigation) if the component unmounts while hireAgentFromTemplate is in flight", async () => {
    let resolveHire!: (v: { taskId: number }) => void
    hireAgentFromTemplateMock.mockReturnValueOnce(
      new Promise((resolve) => { resolveHire = resolve })
    )

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())

    cleanup()

    await act(async () => {
      resolveHire({ taskId: 42 })
    })

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalledTimes(2))
    expect(updateUserPreferencesMock.mock.calls[1][0]).toEqual(expect.objectContaining({ onboarded: true }))
    expect(routerReplace).not.toHaveBeenCalledWith("/task/42")
  })

  it("does not include onboarded:true in the main save - only in a separate save once the agent is confirmed", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalledTimes(2))
    expect(updateUserPreferencesMock.mock.calls[0][0]).not.toHaveProperty("onboarded")
    expect(updateUserPreferencesMock.mock.calls[1][0]).toEqual(expect.objectContaining({ onboarded: true }))
    // The agent-confirming call must happen before the onboarded save, not
    // just eventually - hireAgentFromTemplateMock resolves synchronously in
    // this mock, so by the time updateUserPreferences is called a 2nd time,
    // the agent already exists.
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)
  })

  it("never sends onboarded:true if hireAgentFromTemplate itself fails", async () => {
    hireAgentFromTemplateMock.mockRejectedValueOnce(new Error("network down"))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalled())
    for (const call of updateUserPreferencesMock.mock.calls) {
      expect(call[0]).not.toHaveProperty("onboarded")
    }
  })

  it("still navigates to the hired agent, and marks the escape flag, if only the final onboarded save fails", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, retryable: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task/42"))
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  it("ignores a second click on 'Start with X' while the first save is still in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    const startButton = screen.getByText("Start with Maya")
    fireEvent.click(startButton)
    fireEvent.click(startButton)
    fireEvent.click(startButton)

    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveSave({ ok: true })
    })

    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)
  })

  it("shows an error toast and resets launching state so the user can retry when hireAgentFromTemplate fails", async () => {
    hireAgentFromTemplateMock.mockRejectedValueOnce(new Error("network down"))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(toastErrorMock).toHaveBeenCalledWith("Couldn't hire Maya. Please try again.")
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
    expect(screen.getByText("Start with Maya").closest("button")).not.toBeDisabled()

    // Retrying must actually work - launchingRef/setLaunching were reset.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  it("re-checks hired status right before hiring and redirects to the existing agent if another session hired it meanwhile", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...TEMPLATES[0], hired: true, hired_agent_id: 77 }),
        })
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task?agent=77"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
  })

  it("marks onboarded:true and navigates when the freshness recheck finds it already hired and the onboarded save succeeds", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...TEMPLATES[0], hired: true, hired_agent_id: 77 }),
        })
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task?agent=77"))
    expect(updateUserPreferencesMock).toHaveBeenCalledWith(expect.objectContaining({ onboarded: true }))
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("still navigates to the existing agent, and marks the escape flag, if the onboarded save fails after a freshness-recheck-confirmed hire", async () => {
    // 1st call is the main (includeOnboarded: false) save - must succeed so
    // handleLaunch proceeds past it; the 2nd call is markOnboardedAndNavigate's
    // dedicated {onboarded: true} save, which is the one this test fails.
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, retryable: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...TEMPLATES[0], hired: true, hired_agent_id: 77 }),
        })
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task?agent=77"))
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  it("skips hireAgentFromTemplate and goes straight to the agent when it's already hired", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task?agent=99"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(updateUserPreferencesMock).toHaveBeenCalledWith(expect.objectContaining({ onboarded: true }))
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("still navigates to the already-hired agent, and marks the escape flag, if the onboarded save fails", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, retryable: true })
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task?agent=99"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  it("does not send the completion PATCH under a swapped-in identity if the swap happens after this page has already unmounted", async () => {
    let resolveMainSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValueOnce(
      new Promise((resolve) => { resolveMainSave = resolve })
    )
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

    const { unmount } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    fireEvent.click(screen.getByText("Start with Maya"))
    // The main save (trySavePreferences) is now the one pending call.
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)

    // Unmount BEFORE the swap - no rerender happens afterward, so a ref
    // that only updates via a mounted component's effect could never see
    // this. Then the main save resolves, and handleLaunch (still running
    // post-unmount) reaches the already-hired shortcut and calls
    // markOnboardedAndNavigate for the very first time.
    unmount()
    authUser.id = "user-b"

    await act(async () => {
      resolveMainSave({ ok: true })
    })

    // markOnboardedAndNavigate's entry guard must catch the swap and bail
    // BEFORE sending its own completion PATCH at all - not just before
    // navigating with it.
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)
    expect(routerReplace).not.toHaveBeenCalledWith("/task?agent=99")
  })

  it("does not navigate to the already-hired agent if the swap happens while THIS call's own onboarded save is in flight", async () => {
    let resolveOnboardedSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: true }) // the main save (trySavePreferences), resolves immediately
      .mockReturnValueOnce(new Promise((resolve) => { resolveOnboardedSave = resolve })) // markOnboardedAndNavigate's own save
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    // The main save has resolved; markOnboardedAndNavigate's own save (the
    // 2nd updateUserPreferences call) is now the one pending.
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(2)

    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      resolveOnboardedSave({ ok: true })
    })

    expect(routerReplace).not.toHaveBeenCalledWith("/task?agent=99")
  })

  it("exposes the team-step loading state to assistive technology via role=status", async () => {
    apiRequestMock.mockReturnValue(new Promise(() => {}))

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByRole("status")).toBeInTheDocument()
  })

  it("fetches templates for the current locale, not an unlocalized default", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())
    expect(apiRequestMock.mock.calls[0][0]).toContain("/api/templates/?lang=en")
  })

  it("does not hire the agent or navigate when saving preferences fails", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("Couldn't save your setup — please try again."))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
  })

  it("proceeds to hire anyway after 2 consecutive RETRYABLE save failures on 'Start with X', marking the save-escape flag", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  it("resends the full collected fields (not just the marker) in the final completion write after escaping 2 failed field saves", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: true })
      .mockResolvedValueOnce({ ok: false, retryable: true })
      .mockResolvedValueOnce({ ok: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(3)
    // toEqual against the FULL exact object, not objectContaining a subset -
    // a test-pinning review found objectContaining would stay green even if
    // a future bug in buildPreferencesPayload silently dropped `voice`
    // (reached via the "How should" step above, so hasReachedVoiceRef.current
    // is true and it must be present) or `industry`.
    expect(updateUserPreferencesMock.mock.calls[2][0]).toEqual({
      onboarded: true,
      department: "marketing",
      industry: null,
      goals: ["social"],
      voice: "professional",
    })
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("never proceeds to hire after repeated NON-retryable save failures on 'Start with X'", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: false })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    for (let i = 0; i < 4; i++) {
      await act(async () => {
        fireEvent.click(screen.getByText("Start with Maya"))
      })
    }

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
  })

  it("does not escalate when failure #1 is retryable but failure #2 is not", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: true })
      .mockResolvedValueOnce({ ok: false, retryable: false })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("does not escalate on failure #2 even though it's retryable, since failure #1 was not", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: false })
      .mockResolvedValueOnce({ ok: false, retryable: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("still blocks escalation on a 3rd attempt after 2 permanent failures, even though the 3rd is itself retryable", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: false })
      .mockResolvedValueOnce({ ok: false, retryable: false })
      .mockResolvedValueOnce({ ok: false, retryable: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("does not inherit a prior payload's failure streak after the user edits their answers", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: true })
      .mockResolvedValueOnce({ ok: false, retryable: true })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // 1st attempt fails - payload is {department: marketing, goals: [social]}.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    // Go back and edit: add another goal, changing the payload.
    fireEvent.click(screen.getByText("What you need"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Keep my inbox under control"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // 2nd attempt is for the EDITED (different) payload - this is its 1st
    // failure, not the 2nd failure of the old streak, so it must NOT
    // escalate yet.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("does not let a permanent failure on the OLD payload block escalation for 2 retryable failures on an EDITED payload", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: false }) // old payload: permanent failure
      .mockResolvedValueOnce({ ok: false, retryable: true }) // edited payload: 1st failure
      .mockResolvedValueOnce({ ok: false, retryable: true }) // edited payload: 2nd failure - must escalate

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // 1st attempt fails PERMANENTLY - payload is {department: marketing, goals: [social]}.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    // Go back and edit: add another goal, changing the payload.
    fireEvent.click(screen.getByText("What you need"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Keep my inbox under control"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // Edited payload's 1st failure (purely retryable) - not yet at the
    // 2-failure threshold, must not escalate.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    // Edited payload's 2nd failure, also purely retryable - the OLD
    // payload's permanent failure must not still be blocking this streak.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)
  })

  it("allows a later streak of purely retryable failures to escalate after an earlier permanent failure was cleared by a success", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: false }) // 1st launch attempt: permanent failure
      .mockResolvedValueOnce({ ok: true }) // 2nd attempt: main save succeeds, clearing hasPermanent
      .mockResolvedValueOnce({ ok: false, retryable: true }) // 3rd attempt: 1st failure of a NEW streak
      .mockResolvedValueOnce({ ok: false, retryable: true }) // 4th attempt: 2nd failure of that streak
      .mockResolvedValue({ ok: true }) // the eventual last-mile onboarded save
    hireAgentFromTemplateMock
      .mockRejectedValueOnce(new Error("hire failed after a successful save")) // 2nd attempt's hire fails
      .mockResolvedValueOnce({ taskId: 42 }) // 4th attempt's hire succeeds

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // Attempt 1: permanent failure - retry in place, count=1, hasPermanent=true.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    // Attempt 2: main save succeeds (resets count=0, hasPermanent=false), but
    // hireAgentFromTemplate itself throws - handleLaunch's outer catch resets
    // launching so a 3rd click is possible, without touching the just-reset refs.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)

    // Attempt 3: 1st failure of a brand new streak - purely retryable, count=1.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)

    // Attempt 4: 2nd failure of that same streak, still purely retryable -
    // must escalate. If the success on attempt 2 hadn't cleared hasPermanent,
    // attempt 1's stale "permanent" memory would wrongly block this.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(2))
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  it("still proceeds to hire and marks the escape flag when the escalated freshness re-check itself throws", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.reject(new Error("network down"))
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  it("skips a recommended template that has no persona and falls back to the next valid one", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [
        { ...TEMPLATES[0], persona: null },
        TEMPLATES[1],
      ],
    })

    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Keep my inbox under control"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.queryByText("Maya")).not.toBeInTheDocument()
    expect(screen.getByText("Ellie")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^Start with/ })).not.toBeDisabled()
  })

  it("falls back to the 3 default agents when every recommended template is unavailable", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    // "docs" -> general-doc-summarizer-action-extractor, which isn't in the
    // fetched TEMPLATES fixture at all.
    fireEvent.click(screen.getByText("Summarise long documents"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByText("Maya")).toBeInTheDocument()
    expect(screen.getByText("Ellie")).toBeInTheDocument()
    expect(screen.getByText("Kevin")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^Start with/ })).not.toBeDisabled()
  })

  it("promotes a 4th-ranked match to fill a slot when a top-3 match has no persona", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [
        TEMPLATES[0], // social -> Maya, has persona
        TEMPLATES[1], // inbox -> Ellie, has persona
        { ...TEMPLATES[2], persona: null }, // meetings -> Kevin, NO persona this time
        {
          id: "support-ai-chatbot-agent",
          name: "Chatbot",
          category: "Support",
          description: "Answers customer questions.",
          features: [],
          persona: { name: "Nora", role: "Support Chatbot", intro: "Hi", kickoff_questions: [] },
          connections: [],
          setup_time: "5 min",
          tags: [],
          author: "xagent",
          version: "1.0",
          views: 0,
          likes: 0,
          used_count: 0,
        },
      ],
    })

    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media")) // social
    fireEvent.click(screen.getByText("Keep my inbox under control")) // inbox
    fireEvent.click(screen.getByText("Write up my meetings")) // meetings - no persona
    fireEvent.click(screen.getByText("Answer customer questions")) // support - 4th ranked, has persona
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByText("Maya")).toBeInTheDocument()
    expect(screen.getByText("Ellie")).toBeInTheDocument()
    expect(screen.getByText("Nora")).toBeInTheDocument()
    expect(screen.queryByText("Kevin")).not.toBeInTheDocument()
  })

  it("still reports 1 match waiting when 4 goals are all genuinely matched but only 3 cards fit", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [
        TEMPLATES[0], // social -> Maya, has persona
        TEMPLATES[1], // inbox -> Ellie, has persona
        TEMPLATES[2], // meetings -> Kevin, has persona
        {
          id: "support-ai-chatbot-agent",
          name: "Chatbot",
          category: "Support",
          description: "Answers customer questions.",
          features: [],
          persona: { name: "Nora", role: "Support Chatbot", intro: "Hi", kickoff_questions: [] },
          connections: [],
          setup_time: "5 min",
          tags: [],
          author: "xagent",
          version: "1.0",
          views: 0,
          likes: 0,
          used_count: 0,
        },
      ],
    })

    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media")) // social - matched
    fireEvent.click(screen.getByText("Keep my inbox under control")) // inbox - matched
    fireEvent.click(screen.getByText("Write up my meetings")) // meetings - matched
    fireEvent.click(screen.getByText("Answer customer questions")) // support - matched, 4th
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    // All 4 selected goals matched a real persona, but only 3 cards fit -
    // the 4th real match is still genuinely hidden and must be reported.
    expect(screen.getByText(/the other match is waiting in Templates/)).toBeInTheDocument()
  })

  it("the goals step's 'not sure yet' link persists onboarded:true but not voice, since the voice step was never reached", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Skip — show me everyone"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({
        onboarded: true,
      })
    )
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).not.toHaveProperty("goals")
    expect(call).not.toHaveProperty("voice")
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/templates"))
  })

  it("persists voice on the header Skip once the voice step has actually been visited", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: /Voice:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Playful" }))

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, voice: "playful" })
    )
  })

  it("still persists voice on the header Skip after going Back past the voice step", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: /Voice:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Playful" }))

    // Back to the first step; the visible voice choice must remain saved.
    fireEvent.click(screen.getByText("Back"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, voice: "playful" })
    )
  })

  it("does not discard already-selected goals when leaving via the goals step's 'not sure yet' link", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Skip — show me everyone"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, goals: ["social"] })
    )
  })

  it("sends an explicit empty goals array (not an omitted key) after selecting then deselecting every goal", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Post on social media")) // deselect - back to zero
    fireEvent.click(screen.getByText("Skip — show me everyone"))

    // toEqual against the FULL exact object, not objectContaining a subset
    // - a self-review finding caught that a subset check can't prove the
    // goals-clear latch doesn't leak into (or get masked by) sibling
    // fields: department must still be present, industry must still
    // correctly be explicit `null` (work isn't "other" - pre-existing
    // logic, unrelated to this fix), and voice must still be omitted
    // entirely (the voice step was never reached in this flow).
    expect(updateUserPreferencesMock).toHaveBeenCalledWith({
      onboarded: true,
      department: null,
      industry: null,
      goals: [],
    })
  })

  it("still sends an explicit empty goals array after selecting and deselecting multiple goals in a different order", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media")) // select A
    fireEvent.click(screen.getByText("Keep my inbox under control")) // select B
    fireEvent.click(screen.getByText("Post on social media")) // deselect A
    fireEvent.click(screen.getByText("Keep my inbox under control")) // deselect B - back to zero
    fireEvent.click(screen.getByText("Skip — show me everyone"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, goals: [] })
    )
  })

  it("omits goals entirely (not an explicit empty array) when the user never touched goal selection at all", async () => {
    await startOnboarding()


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Skip — show me everyone"))

    const payload = updateUserPreferencesMock.mock.calls[0][0]
    expect(payload).not.toHaveProperty("goals")
  })

  it("navigates away anyway after 2 consecutive save failures to the same destination, instead of trapping the user", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).toHaveBeenCalledWith("/templates")
    // Full-feature self-review finding: without telling AuthGuard about
    // this, its own onboarding check on "/task" would see onboarded still
    // false and immediately bounce the user right back, defeating the
    // point of escaping at all.
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
    // Pins a PR review finding: the escape flag must be bound to the
    // CURRENT user's id (not left identity-agnostic) so a cross-tab
    // identity swap can't let a different user consume it.
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledWith("user-a")
    // Pins a regression caught by self-reviewing the trySavePreferences
    // extraction: the shared-helper refactor initially dropped this reset
    // on the "escaped" outcome specifically (only restoring it on
    // retry_in_place), leaving the exit button visibly stuck disabled for
    // as long as router.replace takes to actually resolve and unmount.
    expect(screen.getByText("Skip setup").closest("button")).not.toBeDisabled()
  })

  it("does not claim 'other matches waiting' before templates have finished loading", async () => {
    let resolveTemplates!: () => void
    apiRequestMock.mockReturnValue(
      new Promise((resolve) => {
        resolveTemplates = () => resolve({ ok: true, json: async () => TEMPLATES })
      })
    )

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Keep my inbox under control"))
    fireEvent.click(screen.getByText("Write up my meetings"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    // 3 goals picked, templates still loading - must not yet claim any
    // goals are "waiting in Templates" (validRecommended is unresolved, not
    // confirmed short by 0).
    expect(screen.queryByText(/waiting in Templates/)).not.toBeInTheDocument()

    await act(async () => {
      resolveTemplates()
    })
    // All 3 goals map to the 3 fetched templates, so once loaded there's
    // nothing extra waiting either - this only pins the *during-load* state
    // above, not a specific post-load count.
  })

  it("counts a goal as unmatched (not fulfilled by fallback filler cards) when its recommended template never resolves", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    // "docs" maps to general-doc-summarizer-action-extractor, which isn't in
    // the TEMPLATES fixture - its recommendation can never resolve, and the
    // 3 fallback IDs (which ARE in the fixture) fill validRecommended instead.
    fireEvent.click(screen.getByText("Summarise long documents"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByText(/the other match is waiting in Templates/)).toBeInTheDocument()
  })

  it("counts exactly the unmatched goal when one of two selected goals matches and the other doesn't", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media")) // matches - in the TEMPLATES fixture
    fireEvent.click(screen.getByText("Summarise long documents")) // does not match - see the test above
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    // 1 of 2 goals matched, well below the 3-card cap - exactly 1 must be
    // reported as still waiting, not 0 (undercounting) and not 2
    // (double-counting the one that already has a card).
    expect(screen.getByText(/the other match is waiting in Templates/)).toBeInTheDocument()
  })

  it("reports the plural 'other N matches' when 2 selected goals both fail to resolve to a real template", async () => {
    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    // Neither "docs" nor "leads" maps to a template in the TEMPLATES
    // fixture - matchedRecommended stays empty, so validRecommended falls
    // back to filler cards and omittedGoalCount must equal goals.length (2).
    fireEvent.click(screen.getByText("Summarise long documents"))
    fireEvent.click(screen.getByText("Reply to new leads fast"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByText(/the other 2 matches are waiting in Templates/)).toBeInTheDocument()
  })

  it("captures wizardUserIdRef from the live session snapshot at mount, not from the user object", async () => {
    authUser.id = "user-a"
    sessionUserIdOverride.current = "user-b"

    await startOnboarding()

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    // Once mounted, resync the override to match authUser.id ("user-b") -
    // if wizardUserIdRef had wrongly captured "user-a" (from user?.id) at
    // mount, every guard would now see a mismatch against this genuinely
    // unchanged, correct identity and refuse to save at all.
    sessionUserIdOverride.current = undefined
    authUser.id = "user-b"

    fireEvent.click(screen.getByText("Post on social media"))
    await act(async () => {
      fireEvent.click(screen.getByText("Skip — show me everyone"))
    })

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, goals: ["social"] })
    )
  })

  it("does not save, hire, or navigate if the authenticated identity changes mid-wizard", async () => {
    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    // Simulates a same-origin cross-tab login as a different user while this
    // tab still has the wizard open, followed by the re-render that a real
    // identity swap (a React state update in AuthProvider) actually causes -
    // authUser is a stable mock reference, so mutating it and re-rendering
    // stands in for that state update without an actual remount.
    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(updateUserPreferencesMock).not.toHaveBeenCalled()
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("does not save, hire, or navigate if the session is cleared (logout) mid-wizard, not just swapped to another user", async () => {
    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    sessionUserIdOverride.current = null
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(updateUserPreferencesMock).not.toHaveBeenCalled()
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("does not save or navigate on Skip setup if the authenticated identity changes mid-wizard", async () => {
    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())

    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })

    expect(updateUserPreferencesMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("does not complete onboarding under a swapped-in identity if the swap happens while an earlier save in the same click is still in flight", async () => {
    let resolveFirstSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValueOnce(
      new Promise((resolve) => { resolveFirstSave = resolve })
    )
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ ...TEMPLATES[0], hired: true, hired_agent_id: 77 }),
        })
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    // The first (main fields) save is still pending - handleLaunch is
    // parked awaiting it, exactly the in-flight window the guard must cover.
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)

    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      resolveFirstSave({ ok: true })
    })

    // handleLaunch resumes past the first save (it was legitimately sent
    // under the original identity before the swap) and reaches the
    // already-hired shortcut, which calls markOnboardedAndNavigate - that
    // call's own guard must now see the swap and refuse to send the
    // completion write or navigate.
    expect(updateUserPreferencesMock).toHaveBeenCalledTimes(1)
    expect(routerReplace).not.toHaveBeenCalled()
  })

  it("does not hire an agent under a swapped-in identity if the swap happens after the freshness recheck but before the hire call", async () => {
    let resolveFreshCheck!: (v: { ok: boolean }) => void
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return new Promise((resolve) => { resolveFreshCheck = resolve })
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

    const { rerender } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByRole("heading", { name: /take off your plate/ })).toBeInTheDocument())


    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    await waitFor(() => expect(screen.getByText("Start with Maya")).toBeEnabled())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    // Main save (default beforeEach mock, resolves immediately) already went
    // through under the original identity; the freshness recheck is now the
    // one pending call.
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    authUser.id = "user-b"
    await act(async () => {
      rerender(<OnboardingPage />)
    })

    await act(async () => {
      resolveFreshCheck({ ok: false })
    })

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalled()
  })

})
