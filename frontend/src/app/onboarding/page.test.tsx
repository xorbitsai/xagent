import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"
import OnboardingPage from "./page"

const routerPush = vi.hoisted(() => vi.fn())
const routerReplace = vi.hoisted(() => vi.fn())
const authUser = vi.hoisted(() => ({ id: "user-a", username: "Shulei" as string | undefined }))
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

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUser }),
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

async function goToWelcomeThenBusiness() {
  render(<OnboardingPage />)
  await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
  fireEvent.click(screen.getByText("Let's go"))
}

describe("OnboardingPage", () => {
  beforeEach(() => {
    routerPush.mockClear()
    routerReplace.mockClear()
    i18nState.locale = "en"
    authUser.username = "Shulei"
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

  // The reference UI (onboarding.html) hardcodes "Gerard Santos" as a
  // placeholder - this must come from the real logged-in user instead.
  it("greets the real authenticated user by username, not a hardcoded placeholder", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getAllByText("Shulei").length).toBeGreaterThan(0))
    expect(screen.queryByText(/Gerard Santos/)).not.toBeInTheDocument()
  })

  // Pins a PR review finding: the welcome heading's closing punctuation used
  // to be a hardcoded ASCII period in JSX, appended directly after
  // <em>{firstName}</em> - now composed from a locale-owned titleSuffix key
  // instead. Asserts the actual rendered heading, not just the raw locale
  // string values (translations.test.ts covers those separately).
  it("renders the welcome heading with the name and closing punctuation combined with no stray whitespace", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getAllByText("Shulei").length).toBeGreaterThan(0))

    const heading = screen.getByRole("heading", { level: 1 })
    expect(heading.textContent).toBe("Welcome to Xagent,Shulei.")
  })

  it("falls back to a generic name when the user has no username", async () => {
    authUser.username = undefined
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
    expect(screen.getAllByText("there").length).toBeGreaterThan(0)
  })

  // Pins a gap found in full-feature self-review: each step swaps the
  // entire JSX subtree (only one `step === "..."` branch is ever mounted),
  // so the browser drops focus to <body> on every transition with nothing
  // to tell a keyboard/screen-reader user a new step even loaded. The new
  // step's heading must receive focus so it's announced and Tab resumes
  // from there instead of the very top of the page.
  it("moves focus to the new step's heading after advancing to the next step", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
    expect(document.activeElement).toBe(screen.getByRole("heading", { level: 1 }))

    fireEvent.click(screen.getByText("Let's go"))

    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("What doesyour team do?")
    )
    expect(document.activeElement).toBe(screen.getByRole("heading", { level: 1 }))
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

  // Pins a finding verified during PR review: switching the department away
  // from "Other" only cleared the LOCAL `industry` state - since the save
  // is a merge PATCH, omitting the key left a stale industry value stored
  // server-side, silently paired with the new department. `null` is this
  // endpoint's actual "clear this field" signal.
  it("explicitly clears a previously-typed industry when switching the department away from 'Other'", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))
    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "Legal services" },
    })

    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Skip setup"))

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalled())
    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ department: "marketing", industry: null })
    )
  })

  // Companion to the test above: an `industry: null` clear signal must only
  // ever fire once a real department has actually been chosen - a user who
  // never reached (or never touched) the business step must not have this
  // field explicitly cleared on their behalf, which would be a different,
  // wrong claim ("the user had an industry and removed it") from the true
  // one ("this field was never touched").
  it("omits industry entirely (neither a value nor a clear) when the department was never chosen", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalled())
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).not.toHaveProperty("industry")
    expect(call).not.toHaveProperty("department")
  })

  // Pins a test-coverage gap found in full-feature self-review: the header
  // "Skip setup" exit is reachable from any step regardless of Continue's
  // validation state (it's only disabled while a save is in flight), so a
  // user can pick "Other," type only whitespace, and exit before Continue
  // would have ever required real text. Whitespace-only industry must be
  // omitted, not sent as-is - `industry.trim() || undefined` regressing to
  // `industry || undefined` would send the raw whitespace string instead.
  it("omits industry (does not send whitespace) when 'Other' is picked but only whitespace was typed, exited via header Skip", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))
    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "   " },
    })

    fireEvent.click(screen.getByText("Skip setup"))

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalled())
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).toHaveProperty("department", "other")
    expect(call).not.toHaveProperty("industry")
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
    expect(routerReplace).not.toHaveBeenCalled()

    await act(async () => {
      resolveSave({ ok: true })
    })

    expect(routerReplace).toHaveBeenCalledWith("/task")
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

  // Pins a PR review finding: buildPreferencesPayload() snapshots its
  // answers before the PATCH is sent, so an answer changed while a save is
  // in flight would silently never be saved, then get discarded entirely
  // once the page navigates away - the data-entry controls (not just the
  // exit buttons) must be disabled for the duration of a save too.
  it("disables the work chips and industry input while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Other"))
    fireEvent.change(screen.getByPlaceholderText("e.g. Property management"), {
      target: { value: "Legal services" },
    })

    fireEvent.click(screen.getByText("Skip setup"))

    expect(screen.getByText("Other").closest("button")).toBeDisabled()
    expect(screen.getByPlaceholderText("e.g. Property management")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  // Pins a test-coverage gap found in incremental self-review: the test
  // above only reaches the business step's work chips - the goals step's
  // goal chips and the voice step's voice chips also got disabled={launching}
  // in the same round but were never actually exercised.
  it("disables the goal chips while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))

    fireEvent.click(screen.getByText("Not sure yet — show me everyone"))

    expect(screen.getByText("Post on social media").closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  it("disables the voice chips while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Playful"))

    fireEvent.click(screen.getByText("Skip setup"))

    expect(screen.getByText("Playful").closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  // Pins a bug found in full-feature self-review: the save-escape flag used
  // to be marked unconditionally on the 2nd consecutive failure, even though
  // the matching navigation right after it is gated on isMountedRef - an
  // unmount between the two (e.g. a forced re-auth redirect elsewhere) left
  // the flag live in sessionStorage with no navigation to consume it,
  // wrongly suppressing AuthGuard's next check on wherever the user actually
  // landed. The flag must only be marked alongside a navigation that's
  // actually about to happen.
  it("does not mark the save-escape flag if the component unmounts before the 2nd failure's navigation would fire", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: false })
    let resolveSecondSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValueOnce(
      new Promise((resolve) => { resolveSecondSave = resolve })
    )

    const { unmount } = render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

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
    expect(routerReplace).not.toHaveBeenCalled()
  })

  // Pins a finding from full-feature self-review: PersonaAvatar's own doc
  // comment explains `decorative` exists exactly for a picker button that
  // renders the name right next to the avatar - without it, a screen reader
  // announces the name twice ("Maya Maya...") since the avatar's fallback
  // initial and the visible name text are both inside the same interactive
  // <button aria-pressed>. The done-step's avatar reuse is inside a plain,
  // non-interactive div, so it doesn't need this - only the team-step picker did.
  it("hides the team card's persona avatar from the accessibility tree so its name isn't announced twice", async () => {
    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Let's go"))
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText("Maya")).toBeInTheDocument())
    const card = screen.getByText("Maya").closest("button")!
    expect(card.querySelector('[aria-hidden="true"]')).toHaveTextContent("M")
  })

  // Pins a test-coverage gap found in incremental self-review: every other
  // test in this file renders in English only, and the raw-string spacing
  // tests in translations.test.ts never render actual JSX - neither would
  // catch a future JSX-level literal space reintroduced specifically on
  // the Chinese render path. Drives the full flow in zh and asserts the
  // exact combined textContent has no space between the full-width colon
  // and the value, matching the originally-reported "所在领域： 市场营销" bug.
  it("renders the Done step's 'working in' line with no stray space in Chinese", async () => {
    i18nState.locale = "zh"

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/欢迎来到/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("开始吧"))
    await waitFor(() => expect(screen.getByText(/你的团队/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("市场营销"))
    fireEvent.click(screen.getByText("继续"))
    await waitFor(() => expect(screen.getByText("发布社交媒体内容")).toBeInTheDocument())
    fireEvent.click(screen.getByText("发布社交媒体内容"))
    fireEvent.click(screen.getByText("继续"))
    await waitFor(() => expect(screen.getByText("认识一下你的 AI 团队。")).toBeInTheDocument())
    fireEvent.click(screen.getByText("继续"))
    fireEvent.click(screen.getByText("继续"))
    await waitFor(() => expect(screen.getByText("一切就绪。")).toBeInTheDocument())

    const line = screen.getByText("市场营销").closest(".ob-sum-li")!
    expect(line.textContent).toBe("所在领域：市场营销")
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
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  // Pins a PR review finding: onboarded:true used to be saved durably as
  // part of the FIRST preferences save, before the freshness re-check and
  // hireAgentFromTemplate (both of which can still fail or the component
  // can still unmount) ever ran - a later failure could leave the backend
  // saying "onboarded" with no agent ever actually delivered, and no way
  // back into onboarding to finish it. It must now be saved separately,
  // only once an agent is confirmed to exist.
  it("does not include onboarded:true in the main save - only in a separate save once the agent is confirmed", async () => {
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

    await waitFor(() => expect(updateUserPreferencesMock).toHaveBeenCalledTimes(2))
    expect(updateUserPreferencesMock.mock.calls[0][0]).not.toHaveProperty("onboarded")
    expect(updateUserPreferencesMock.mock.calls[1][0]).toEqual({ onboarded: true })
    // The agent-confirming call must happen before the onboarded save, not
    // just eventually - hireAgentFromTemplateMock resolves synchronously in
    // this mock, so by the time updateUserPreferences is called a 2nd time,
    // the agent already exists.
    expect(hireAgentFromTemplateMock).toHaveBeenCalledTimes(1)
  })

  // Pins the same finding from the failure side: if hireAgentFromTemplate
  // itself throws, onboarded must never be sent at all - no agent was ever
  // delivered, so there is nothing to mark complete.
  it("never sends onboarded:true if hireAgentFromTemplate itself fails", async () => {
    hireAgentFromTemplateMock.mockRejectedValueOnce(new Error("network down"))

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

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalled())
    for (const call of updateUserPreferencesMock.mock.calls) {
      expect(call[0]).not.toHaveProperty("onboarded")
    }
  })

  // Pins the ordering fix's own fallback: an agent that's genuinely hired
  // must still navigate the user there even if the LAST-mile onboarded:true
  // save itself fails - and that failure must still tell AuthGuard to stand
  // down once, same as every other escape path, since the deliverable (the
  // agent) is real regardless of whether the flag landed.
  it("still navigates to the hired agent, and marks the escape flag, if only the final onboarded save fails", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, retryable: true })

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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/task/42"))
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  // Pins a test-coverage gap found in full-feature self-review: handleLaunch
  // has the same launchingRef double-click guard as persistAndLeave, but
  // every existing test awaits each click to full completion before firing
  // the next one - a fast double-click on the primary CTA before its PATCH
  // resolves was never actually exercised.
  it("ignores a second click on 'Start with X' while the first save is still in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

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

  // Pins a test-coverage gap found in full-feature self-review: handleLaunch's
  // outer catch (hireAgentFromTemplate itself rejecting, after a successful
  // save and freshness re-check) had zero coverage - a broken retry here
  // would strand a first-time user on the page with no way to proceed.
  it("shows an error toast and resets launching state so the user can retry when hireAgentFromTemplate fails", async () => {
    hireAgentFromTemplateMock.mockRejectedValueOnce(new Error("network down"))

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

    expect(toastErrorMock).toHaveBeenCalledWith("Couldn't hire Maya. Please try again.")
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
    expect(screen.getByText("Start with Maya").closest("button")).not.toBeDisabled()

    // Retrying must actually work - launchingRef/setLaunching were reset.
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
  })

  // Pins a PR review finding: mount-time `selected.hired` can go stale if
  // the same template gets hired from another tab/session mid-wizard.
  // handleLaunch must re-check right before hiring, mirroring the identical
  // guard in templates/[id]/page-client.tsx, instead of seeding a second
  // opening message onto an agent the user already has a real conversation with.
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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/agent/77"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
  })

  // Pins a test-coverage gap found in incremental self-review: the
  // freshness-recheck-hired branch's own onboarded-save call (via
  // markOnboardedAndNavigate) was only ever exercised with the default
  // succeeding mock - neither its success payload nor its failure path was
  // directly asserted, unlike the freshly-hired branch which got both.
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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/agent/77"))
    expect(updateUserPreferencesMock).toHaveBeenCalledWith({ onboarded: true })
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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/agent/77"))
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  // Pins a PR review test-coverage gap: handleLaunch's already-hired
  // shortcut (skip hireAgentFromTemplate, go straight to the existing agent)
  // was never exercised.
  it("skips hireAgentFromTemplate and goes straight to the agent when it's already hired", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/agent/99"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(updateUserPreferencesMock).toHaveBeenCalledWith({ onboarded: true })
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  // Pins a test-coverage gap found in incremental self-review: the
  // already-hired shortcut's own onboarded-save failure path (via
  // markOnboardedAndNavigate) was untested - only the freshly-hired branch
  // had this covered.
  it("still navigates to the already-hired agent, and marks the escape flag, if the onboarded save fails", async () => {
    updateUserPreferencesMock.mockResolvedValueOnce({ ok: true }).mockResolvedValueOnce({ ok: false, retryable: true })
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => [{ ...TEMPLATES[0], hired: true, hired_agent_id: 99 }, TEMPLATES[1], TEMPLATES[2]],
    })

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

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/agent/99"))
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  // Pins a PR review test-coverage gap: only the thrown/rejected variant of
  // a templates-fetch failure was tested - a non-throwing !response.ok, and
  // a 200 response whose body isn't an array, are separate code paths.
  it("stays on the loading spinner when the templates fetch resolves with a non-ok response (not a thrown error)", async () => {
    apiRequestMock.mockResolvedValue({ ok: false })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByTestId("onboarding-team-loading")).toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).toBeDisabled()
  })

  // Pins a test-coverage gap found in full-feature self-review: prior tests
  // only asserted the loading spinner's presence via data-testid, never its
  // accessible status semantics directly - a regression that dropped
  // role="status" (or moved it off the wrapping element) would pass every
  // other test in this file.
  it("exposes the team-step loading state to assistive technology via role=status", async () => {
    apiRequestMock.mockResolvedValue({ ok: false })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByRole("status")).toBeInTheDocument()
  })

  it("stays on the loading spinner when the templates fetch resolves ok with a non-array body", async () => {
    apiRequestMock.mockResolvedValue({ ok: true, json: async () => ({ not: "an array" }) })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    expect(screen.getByTestId("onboarding-team-loading")).toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).toBeDisabled()
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
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
  })

  // Flagged by PR review (xorbitsai/xagent#1617, major finding #4): unlike
  // every persistAndLeave exit, "Start with X" had no escape hatch at all -
  // a save that kept failing left the primary CTA permanently unusable.
  it("proceeds to hire anyway after 2 consecutive RETRYABLE save failures on 'Start with X', marking the save-escape flag", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: true })

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
    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
  })

  // Pins a finding verified during PR review: unlike persistAndLeave's
  // escape hatch (which just leaves without saving), handleLaunch's
  // proceeds to an IRREVERSIBLE hireAgentFromTemplate call. A NON-retryable
  // failure (e.g. a 422 the backend will reject identically every time)
  // must never escalate to "proceed anyway," no matter how many times it's
  // retried - it hires an agent while preferences were never actually
  // saved, for a save that could never have succeeded.
  it("never proceeds to hire after repeated NON-retryable save failures on 'Start with X'", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: false })

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

    for (let i = 0; i < 4; i++) {
      await act(async () => {
        fireEvent.click(screen.getByText("Start with Maya"))
      })
    }

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalledWith(expect.stringContaining("/task/"))
  })

  // Pins a test-coverage gap found in incremental self-review: the escape
  // decision looks only at the CURRENT attempt's retryable flag alongside
  // the cumulative failure count, not some "was ever non-retryable" memory
  // - these two tests exercise both attempts having DIFFERENT retryable
  // values, which a constant-mock test can't distinguish from a sticky/
  // AND'd implementation.
  it("does not escalate when failure #1 is retryable but failure #2 is not", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: true })
      .mockResolvedValueOnce({ ok: false, retryable: false })

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
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    expect(hireAgentFromTemplateMock).not.toHaveBeenCalled()
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  it("escalates on failure #2 when it's retryable, even though failure #1 was not", async () => {
    updateUserPreferencesMock
      .mockResolvedValueOnce({ ok: false, retryable: false })
      .mockResolvedValueOnce({ ok: false, retryable: true })

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
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    // The escape flag is no longer tied to the main fields save escalating -
    // only to whether the SEPARATE final onboarded:true save (below,
    // deliberately not mocked to fail here) succeeds. It does, by the
    // default beforeEach mock, so nothing needs to tell AuthGuard to stand
    // down.
    expect(markOnboardingSaveEscapedMock).not.toHaveBeenCalled()
  })

  // Pins a test-coverage gap found in full-feature self-review: the escape
  // path still has the freshness re-check and hireAgentFromTemplate ahead of
  // it after escalating, both of which can throw - the freshness re-check's
  // own try/catch is documented as "fall through" on failure, but nothing
  // proved the escalation (shouldMarkSaveEscape, the eventual hire and
  // navigation) actually survives that specific combination rather than,
  // say, an unhandled rejection short-circuiting the whole function.
  it("still proceeds to hire and marks the escape flag when the escalated freshness re-check itself throws", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false, retryable: true })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes("/api/templates/marketing-social-media-content-manager")) {
        return Promise.reject(new Error("network down"))
      }
      return Promise.resolve({ ok: true, json: async () => TEMPLATES })
    })

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
    await act(async () => {
      fireEvent.click(screen.getByText("Start with Maya"))
    })

    await waitFor(() => expect(hireAgentFromTemplateMock).toHaveBeenCalled())
    expect(markOnboardingSaveEscapedMock).toHaveBeenCalledTimes(1)
    expect(routerReplace).toHaveBeenCalledWith("/task/42")
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

  // Flagged by PR review (xorbitsai/xagent#1617): the 3-card cap used to be
  // applied before filtering for persona-availability, so a 4th-ranked
  // match with a real persona could never fill a slot vacated by a top-3
  // match that turned out to have none - the user saw fewer cards than
  // their own goal selections actually supported.
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

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

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
    expect(screen.getByTestId("onboarding-team-loading")).toBeInTheDocument()
    expect(screen.queryByText("Maya")).not.toBeInTheDocument()
    expect(screen.getByText("Continue").closest("button")).toBeDisabled()
  })

  // Pins a PR review finding: "professional" is a real, legitimate voice
  // choice, not an empty placeholder like goals' [] - so unlike goals,
  // voice must not be sent at all until the voice step has actually been
  // reached. The goals step's skip is reachable well before that.
  it("the goals step's 'not sure yet' link persists onboarded:true but not voice, since the voice step was never reached", async () => {
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
      })
    )
    const call = updateUserPreferencesMock.mock.calls.at(-1)![0]
    expect(call).not.toHaveProperty("goals")
    expect(call).not.toHaveProperty("voice")
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/templates"))
  })

  // Pins the same PR review finding from the other direction: once the
  // voice step HAS been reached (even via the header Skip, not just a
  // completed flow), the choice made there must be persisted.
  it("persists voice on the header Skip once the voice step has actually been visited", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Playful"))

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, voice: "playful" })
    )
  })

  // Pins a bug found in self-review: whether the voice step has been reached
  // was a LIVE comparison against the current step index, not "has this ever
  // happened" - so going Back past the voice step after already picking one
  // silently dropped that choice, the same class of bug as the two tests
  // above but in the opposite direction (losing a real choice instead of
  // persisting one never made).
  it("still persists voice on the header Skip after going Back past the voice step", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Post on social media"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/How should/)).toBeInTheDocument())
    fireEvent.click(screen.getByText("Playful"))

    // Back past voice, to "Meet your AI team".
    fireEvent.click(screen.getByText("Back"))
    await waitFor(() => expect(screen.getByText(/Meet your AI team/)).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))

    expect(updateUserPreferencesMock).toHaveBeenCalledWith(
      expect.objectContaining({ onboarded: true, voice: "playful" })
    )
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

  // Pins a PR review finding: the rail's "About you"/"My team" links always
  // jumped to the FIRST step of their (multi-step) group via findIndex,
  // never the step the user actually last visited in it.
  it("returns to the last-visited step in a group when clicking its rail link, not always the first step", async () => {
    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))

    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())
    // "About you" (group 0) spans welcome+business - business was the last
    // step actually visited there, so clicking it from a later step must
    // return to business ("What does your team do?"), not the welcome splash.
    fireEvent.click(screen.getByText("About you"))
    expect(screen.getByText(/What does/)).toBeInTheDocument()
    expect(screen.queryByText(/Welcome to Xagent/)).not.toBeInTheDocument()

    // Drive through to voice, so "My team" (group 2) becomes reachable from
    // "done" with voice as its last-visited step.
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

    fireEvent.click(screen.getByText("My team"))
    expect(screen.getByText(/How should/)).toBeInTheDocument()
    expect(screen.queryByText(/Meet your AI team/)).not.toBeInTheDocument()
  })

  // Pins a test-coverage gap found in full-feature self-review: prior rail
  // tests only asserted via visible step content changing, never via the
  // aria-current attribute itself - a regression that stopped setting it
  // (while still updating the visual is-now CSS class) would pass every
  // other test in this file.
  it("marks only the current rail group with aria-current=step", async () => {
    await goToWelcomeThenBusiness()

    expect(screen.getByText("About you").closest("button")).toHaveAttribute("aria-current", "step")
    expect(screen.getByText("Goals").closest("button")).not.toHaveAttribute("aria-current")

    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    expect(screen.getByText("Goals").closest("button")).toHaveAttribute("aria-current", "step")
    expect(screen.getByText("About you").closest("button")).not.toHaveAttribute("aria-current")
  })

  // Pins a self-review finding: a rail link (e.g. "About you", reachable
  // from every later step) could still jump the user elsewhere while a
  // header "Skip setup" save was in flight, racing that save's own
  // eventual router.push - every other button was already guarded.
  it("disables rail navigation links while a save is in flight", async () => {
    let resolveSave!: (v: { ok: boolean }) => void
    updateUserPreferencesMock.mockReturnValue(new Promise((resolve) => { resolveSave = resolve }))

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    fireEvent.click(screen.getByText("Skip setup"))
    expect(screen.getByText("About you").closest("button")).toBeDisabled()

    await act(async () => {
      resolveSave({ ok: true })
    })
  })

  // Pins a PR review finding: persistAndLeave used to block navigation
  // forever on a failed save, trapping the user on this full-screen page
  // (with no other nav) if the backend kept rejecting it.
  it("navigates away anyway after 2 consecutive save failures to the same destination, instead of trapping the user", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    render(<OnboardingPage />)
    await waitFor(() => expect(screen.getByText(/Welcome to Xagent/)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).not.toHaveBeenCalled()

    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).toHaveBeenCalledWith("/task")
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

  // Pins a self-review finding: the failure count is per-destination, not
  // one global tally - a failure on one exit action must not spend down a
  // DIFFERENT destination's own first-attempt retry-in-place chance.
  it("does not let a failure on one destination's exit count toward a different destination's first attempt", async () => {
    updateUserPreferencesMock.mockResolvedValue({ ok: false })

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
    await waitFor(() => expect(screen.getByText(/take off your plate/)).toBeInTheDocument())

    // First failure via header "Skip setup" -> /task.
    await act(async () => {
      fireEvent.click(screen.getByText("Skip setup"))
    })
    expect(routerReplace).not.toHaveBeenCalled()

    // First-ever attempt via the goals step's own skip -> /templates: must
    // still get its own retry-in-place chance, not be force-navigated
    // immediately just because /task's counter is already at 1.
    await act(async () => {
      fireEvent.click(screen.getByText("Not sure yet — show me everyone"))
    })
    expect(routerReplace).not.toHaveBeenCalled()
  })

  // Pins a test-coverage gap flagged in self-review: the "N other matches"
  // subtitle must not claim a count before templates have actually loaded.
  it("does not claim 'other matches waiting' before templates have finished loading", async () => {
    let resolveTemplates!: () => void
    apiRequestMock.mockReturnValue(
      new Promise((resolve) => {
        resolveTemplates = () => resolve({ ok: true, json: async () => TEMPLATES })
      })
    )

    await goToWelcomeThenBusiness()
    fireEvent.click(screen.getByText("Marketing"))
    fireEvent.click(screen.getByText("Continue"))
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
})
