import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"
import OnboardingPage from "./page"

const routerPush = vi.hoisted(() => vi.fn())
const authUser = vi.hoisted(() => ({ username: "Shulei" as string | undefined }))
const apiRequestMock = vi.hoisted(() => vi.fn())
const updateUserPreferencesMock = vi.hoisted(() => vi.fn())
const hireAgentFromTemplateMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock("sonner", () => ({
  toast: { error: toastErrorMock },
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUser }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation("en", key as TranslationKey, vars),
    locale: "en",
  }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/user-preferences", () => ({
  updateUserPreferences: updateUserPreferencesMock,
  fetchUserPreferences: vi.fn(),
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

async function goToWelcomeThenBusiness() {
  render(<OnboardingPage />)
  await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
  fireEvent.click(screen.getByText("Let's go"))
}

describe("OnboardingPage", () => {
  beforeEach(() => {
    routerPush.mockClear()
    authUser.username = "Shulei"
    apiRequestMock.mockReset()
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => TEMPLATES })
    updateUserPreferencesMock.mockReset()
    updateUserPreferencesMock.mockResolvedValue({ ok: true })
    hireAgentFromTemplateMock.mockReset()
    hireAgentFromTemplateMock.mockResolvedValue({ taskId: 42 })
    toastErrorMock.mockReset()
  })

  afterEach(cleanup)

  // The reference UI (onboarding.html) hardcodes "Gerard Santos" as a
  // placeholder - this must come from the real logged-in user instead.
  it("greets the real authenticated user by username, not a hardcoded placeholder", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getAllByText("Shulei").length).toBeGreaterThan(0))
    expect(screen.queryByText(/Gerard Santos/)).not.toBeInTheDocument()
  })

  it("falls back to a generic name when the user has no username", async () => {
    authUser.username = undefined
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
    expect(screen.getAllByText("there").length).toBeGreaterThan(0)
  })

  it("requires an industry value before Continue is enabled when 'Other' is picked", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))

    const continueButton = screen.getByText("Continue").closest("button")!
    expect(continueButton).toBeDisabled()

    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "Legal services" },
    })
    expect(continueButton).not.toBeDisabled()
  })

  // Matches PREFERENCES_TEXT_FIELD_MAX_LENGTH in src/xagent/web/api/auth.py -
  // a defensive client-side cap flagged by PR review.
  it("caps the free-text industry field at 200 characters", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))

    expect(screen.getByPlaceholderText("e.g. Property management")).toHaveAttribute("maxLength", "200")
  })

  it("reorders goals to bring the selected work type's goals first", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Sales"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    const chipLabels = screen.getAllByRole("button").map((b) => b.textContent).filter(Boolean) as string[]
    const meetingsIndex = chipLabels.findIndex((t) => t.includes("Write up my meetings"))
    const inboxIndex = chipLabels.findIndex((t) => t.includes("Keep my inbox under control"))
    expect(meetingsIndex).toBeGreaterThanOrEqual(0)
    expect(inboxIndex).toBeGreaterThan(meetingsIndex)
  })

  // Pins the dangerouslySetInnerHTML removal: free-text industry input must
  // render as inert text on the Done step, never be interpreted as markup.
  it("renders a script-tag industry value as literal escaped text on the Done step, never executing it", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))
    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "Legal services <script>alert(1)</script>" },
    })
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText("You're all set.")).toBeInTheDocument())

    expect(document.querySelector("script")).toBeNull()
    expect(
      screen.getByText(
        (_, node) => node?.tagName === "SPAN" && node.textContent === "Working in Legal services <script>alert(1)</script>"
      )
    ).toBeInTheDocument()
  })

  // Pins C1 from the onboarding-flow self-review: persistAndLeave must await
  // the PATCH before navigating away - AuthGuard's onboarding recheck fires a
  // GET as soon as the destination route mounts, and if that GET wins the
  // race against an unawaited PATCH, it reads the old onboarded:false and
  // bounces the user straight back into onboarding right after they left it.
  it("waits for the preferences save to resolve before navigating away on Skip setup", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(expect.objectContaining({ onboarded: true }))
    // The save is still pending - navigation must not have happened yet.
    expect(routerPush).not.toHaveBeenCalled()

    await act(async () => {
      resolveSave({ ok: true })
    })

    expect(routerPush).toHaveBeenCalledWith("/task")
  })

  // Flagged by PR review (xorbitsai/xagent#1617): persistAndLeave had no
  // double-click guard, unlike handleLaunch's launchingRef - a fast second
  // click before the first PATCH resolved would fire two concurrent saves.
  it("ignores a second click on Skip setup while the first save is still in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

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

  // Flagged by PR review (xorbitsai/xagent#1617): persistAndLeave awaited the
  // save but never checked its result - a failed PATCH still navigated away,
  // and AuthGuard would immediately bounce the user back since onboarded
  // never actually got persisted server-side.
  it("does not navigate and shows an error toast when the Skip setup save fails", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })

    expect(toastErrorMock).toHaveBeenCalledWith("Couldn't save your setup — please try again.")
    expect(routerPush).not.toHaveBeenCalled()
  })

  it("hires the selected agent and navigates to /task/{taskId} on launch", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText("You're all set.")).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(hireAgentFromTemplateMock.mock.calls[0][0]).toEqual(
      expect.objectContaining({ templateId: "marketing-social-media-content-manager" })
    )
    expect(routerPush).toHaveBeenCalledWith("/task/42")
  })

  it("fetches templates for the current locale, not an unlocalized default", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())
    expect(apiRequestMock.mock.calls[0][0]).toContain("/api/templates/?lang=en")
  })

  // Pins C3 from the onboarding-flow self-review: a failed preferences save
  // must not be silently ignored - proceeding to hire anyway would leave the
  // user believing setup is done while the backend still has onboarded:false.
  it("does not hire the agent or navigate when saving preferences fails", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText("You're all set.")).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("Couldn't save your setup — please try again."))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(routerPush).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
  })

  // Pins C2: a recommended templateId that never loaded (or has no persona)
  // must not leave the team step stuck on an invalid, unrenderable pick.
  it("skips a recommended template that has no persona and falls back to the next valid one", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [
        { ...TEMPLATES[0], persona: null },
        TEMPLATES[1],
      ],
    })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Keep my inbox under control"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.queryByText("Maya")).not.toBeInTheDocument()
    expect(screen.getByText("Ellie")).toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).not.toBeDisabled()
  })

  // Pins a self-review finding: if EVERY recommended template for the
  // selected goals fails to load or has no persona (not just one of them),
  // the team step must fall back to the same 3 defaults it uses when no
  // goals were picked at all, rather than showing zero cards with Continue
  // stuck disabled and no explanation.
  it("falls back to the 3 default agents when every recommended template is unavailable", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    // "docs" -> general-doc-summarizer-action-extractor, which isn't in the
    // fetched TEMPLATES fixture at all.
    fireEvent.click(screen.getByText("Summarise long documents"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByText("Maya")).toBeInTheDocument()
    expect(screen.getByText("Ellie")).toBeInTheDocument()
    expect(screen.getByText("Kevin")).toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).not.toBeDisabled()
  })

  // Flagged by PR review (xorbitsai/xagent#1617): the templates fetch used to
  // set templatesLoading:false in a finally block regardless of outcome,
  // contradicting its own comment that failure should stay in a loading
  // state rather than show a broken, empty card grid. Also pins that
  // Continue can't go live off of a still-loading, unconfirmed pick.
  it("stays on the loading spinner (Continue disabled) when the templates fetch fails, instead of showing an empty grid", async () => {
    apiRequestMock.mockRejectedValue(new Error("network down"))

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(document.querySelector(".animate-spin, [class*='spin']")).toBeTruthy()
    expect(screen.queryByText("Maya")).not.toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).toBeDisabled()
  })

  it("the goals step's 'not sure yet' link persists onboarded:true and the current voice, with no goals selected yet", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))
    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "Something" },
    })
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Not sure yet — show me everyone"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({
        onboarded: true,
        department: "other",
        industry: "Something",
        voice: "professional",
      })
    )
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).not.toHaveProperty("goals")
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith("/templates"))
  })

  // Pins a bug found in self-review: the reference UI's finish() persists
  // whatever's been picked so far on EVERY exit path, unconditionally - this
  // page used to special-case the goals-step skip to drop goals/voice,
  // silently losing any selections already made before the user bailed out.
  it("does not discard already-selected goals when leaving via the goals step's 'not sure yet' link", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Not sure yet — show me everyone"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, goals: ["social"] })
    )
  })
})
