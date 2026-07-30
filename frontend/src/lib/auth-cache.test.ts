import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  AUTH_CACHE_KEY, AUTH_LOGIN_INTENT_KEY, AUTH_OIDC_INTENT_KEY, AUTH_REVOKED_LOGIN_INTENT_KEY, LEGACY_AUTH_TOKEN_KEY, LEGACY_AUTH_USER_KEY,
  AUTH_TOKEN_UPDATED_EVENT, claimAuthLoginIntent, claimOidcAuthLoginIntent, clearAuthSessionIfCurrent, clearStoredAuth, commitAuthSessionRefresh, compareAuthSession, createAuthSession,
  inspectAuthSession, migrateLegacyAuthSession, readAuthSessionSnapshot, updateAuthSessionUser, type AuthTokenPayload,
  takeOidcAuthLoginIntent,
} from "@/lib/auth-cache"

const user = { id: "1", username: "alice", email: null, is_admin: false }
function installLock(before: () => void | Promise<void> = () => {}) {
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: vi.fn(async (_: string, action: () => Promise<unknown>) => { await before(); return action() }) } })
}
async function created(payload: AuthTokenPayload = { user, access_token: "access", refresh_token: "refresh" }) {
  const claim = await claimAuthLoginIntent()
  expect(claim.status).toBe("claimed")
  if (claim.status !== "claimed") throw new Error("expected login intent")
  const result = await createAuthSession(payload, claim.intent)
  expect(result.status).toBe("created")
  if (result.status !== "created") throw new Error("expected creation")
  return result.projection.snapshot
}
function current() {
  const inspection = inspectAuthSession()
  expect(inspection.status).toBe("valid")
  if (inspection.status !== "valid") throw new Error("expected session")
  return inspection.projection
}
const AUTH_OWNED_LOCAL_KEYS = [AUTH_CACHE_KEY, AUTH_LOGIN_INTENT_KEY, AUTH_REVOKED_LOGIN_INTENT_KEY, LEGACY_AUTH_TOKEN_KEY, LEGACY_AUTH_USER_KEY]
function authStorageSnapshot() {
  return Object.fromEntries(AUTH_OWNED_LOCAL_KEYS.map(key => [key, localStorage.getItem(key)]))
}
describe("auth cache lineage", () => {
  beforeEach(() => { localStorage.clear(); vi.restoreAllMocks(); installLock() })
  afterEach(() => vi.unstubAllGlobals())
  it.each([
    undefined,
    {},
    {
      getItem: () => { throw new Error("storage blocked") },
      setItem: () => { throw new Error("storage blocked") },
      removeItem: () => { throw new Error("storage blocked") },
    },
  ])("treats unavailable or malformed browser storage as absent and fails mutations safely", async storage => {
    vi.stubGlobal("localStorage", storage)
    expect(inspectAuthSession()).toEqual({ status: "absent", projection: null })
    await expect(migrateLegacyAuthSession()).resolves.toEqual({ status: "unavailable", reason: "storage_unavailable" })
    expect(["superseded", "unavailable"]).toContain((await createAuthSession({ user, access_token: "access" })).status)
    await expect(clearStoredAuth()).resolves.toMatchObject({ status: "unavailable", reason: "storage_unavailable", credentialsCleared: false })
  })
  it("does not create a bearer session without the login intent that authorized it", async () => {
    expect(await createAuthSession({ user, access_token: "late-access" })).toEqual({ status: "superseded" })
    expect(inspectAuthSession().status).toBe("absent")
  })
  it("consumes a claimed intent so its response cannot be committed twice", async () => {
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    await expect(createAuthSession({ user, access_token: "first-access" }, claim.intent)).resolves.toMatchObject({ status: "created" })
    await expect(createAuthSession({ user, access_token: "replayed-access" }, claim.intent)).resolves.toEqual({ status: "superseded" })
    expect(current().snapshot.accessToken).toBe("first-access")
  })
  it("returns the exact OIDC intent claimed before a same-tab redirect", async () => {
    const claim = await claimOidcAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected OIDC intent")
    expect(takeOidcAuthLoginIntent()).toEqual({ status: "present", intent: claim.intent })
    expect(takeOidcAuthLoginIntent()).toEqual({ status: "absent" })
  })
  it("rejects a pending password response after explicit logout supersedes its intent", async () => {
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    await expect(clearStoredAuth()).resolves.toMatchObject({ status: "cleared", credentialsCleared: true })

    expect(await createAuthSession({ user, access_token: "late-access" }, claim.intent)).toEqual({ status: "superseded" })
    expect(inspectAuthSession().status).toBe("absent")
  })
  it("keeps the newer password session when an older login response arrives later", async () => {
    const older = await claimAuthLoginIntent()
    expect(older.status).toBe("claimed")
    if (older.status !== "claimed") throw new Error("expected older login intent")
    const newer = await claimAuthLoginIntent()
    expect(newer.status).toBe("claimed")
    if (newer.status !== "claimed") throw new Error("expected newer login intent")

    await expect(createAuthSession({ user, access_token: "new-access" }, newer.intent)).resolves.toMatchObject({ status: "created" })
    await expect(createAuthSession({ user, access_token: "old-access" }, older.intent)).resolves.toEqual({ status: "superseded" })
    expect(current().snapshot.accessToken).toBe("new-access")
  })
  it("validates a cross-tab replacement intent from storage inside the commit lock", async () => {
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    localStorage.setItem(AUTH_LOGIN_INTENT_KEY, JSON.stringify({ schemaVersion: 1, id: "claimed-in-another-tab" }))

    expect(await createAuthSession({ user, access_token: "stale-access" }, claim.intent)).toEqual({ status: "superseded" })
    expect(inspectAuthSession().status).toBe("absent")
  })
  it("rejects an OIDC response after a later password login supersedes its originating intent", async () => {
    const oidc = await claimAuthLoginIntent()
    expect(oidc.status).toBe("claimed")
    if (oidc.status !== "claimed") throw new Error("expected OIDC intent")
    const password = await claimAuthLoginIntent()
    expect(password.status).toBe("claimed")
    if (password.status !== "claimed") throw new Error("expected password intent")

    await expect(createAuthSession({ user, access_token: "password-access" }, password.intent)).resolves.toMatchObject({ status: "created" })
    await expect(createAuthSession({ user, access_token: "oidc-access" }, oidc.intent)).resolves.toEqual({ status: "superseded" })
    expect(current().snapshot.accessToken).toBe("password-access")
  })
  it("creates a distinct lineage for the same user", async () => {
    const first = await created({ user, access_token: "old", refresh_token: "old-refresh" })
    const replacement = await created({ user, access_token: "new", refresh_token: "new-refresh" })
    expect(replacement.sessionId).not.toBe(first.sessionId)
    expect(compareAuthSession(first).status).toBe("replaced")
  })
  it("keeps expired or corrupt inspection pure", () => {
    const now = Date.now()
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ schemaVersion: 2, sessionId: "x", credentialRevision: 0, profileRevision: 0, user, token: "a", refreshToken: "r", timestamp: now - 130 * 60_000, refreshExpiresAt: now - 1 }))
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    expect(inspectAuthSession(now).status).toBe("expired")
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
  })
  it("rejects unsafe persisted absolute deadlines", () => {
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
      schemaVersion: 2, sessionId: "x", credentialRevision: 0, profileRevision: 0,
      user, token: "a", refreshToken: "r", timestamp: Date.now(), expiresAt: Number.MAX_VALUE,
    }))
    expect(inspectAuthSession().status).toBe("invalid")
  })
  it("migrates legacy data exactly once under the coordinator", async () => {
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "legacy"); localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    const first = await migrateLegacyAuthSession(); const second = await migrateLegacyAuthSession()
    expect(first).toMatchObject({ status: "migrated", projection: { snapshot: { accessToken: "legacy" } } })
    expect(second.status).toBe("advanced")
    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
  })
  it("upgrades a usable v1 canonical cache without losing its opaque credentials or expiry", async () => {
    const now = Date.now()
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
      user, token: "opaque access ", refreshToken: "opaque refresh ", timestamp: now - 50,
      expiresAt: now + 60_000, refreshExpiresAt: now + 120_000,
    }))
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "shadow-token")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify({ ...user, username: "shadow" }))

    const migrated = await migrateLegacyAuthSession()

    expect(migrated).toMatchObject({ status: "migrated", projection: { cache: {
      schemaVersion: 2, credentialRevision: 0, profileRevision: 0,
      token: "opaque access ", refreshToken: "opaque refresh ", timestamp: now - 50,
      expiresAt: now + 60_000, refreshExpiresAt: now + 120_000,
    } } })
    expect(migrated.status === "migrated" && migrated.projection.snapshot.sessionId).toBeTruthy()
    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_USER_KEY)).toBeNull()
  })
  it("does not let a queued v1 migration overwrite a concurrent login", async () => {
    let tail: Promise<void> = Promise.resolve()
    Object.defineProperty(navigator, "locks", { configurable: true, value: { request: vi.fn((_name: string, action: () => Promise<unknown>) => {
      const next = tail.then(action)
      tail = next.then(() => undefined, () => undefined)
      return next
    }) } })
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ user, token: "v1", refreshToken: "v1-refresh", timestamp: Date.now() }))
    const migration = migrateLegacyAuthSession()
    const login = created({ user, access_token: "new-login", refresh_token: "new-refresh" })
    await expect(migration).resolves.toMatchObject({ status: "migrated" })
    await expect(login).resolves.toMatchObject({ accessToken: "new-login" })
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { cache: { token: "new-login", refreshToken: "new-refresh" } } })
  })
  it("does not resurrect legacy data when canonical storage is invalid", async () => {
    localStorage.setItem(AUTH_CACHE_KEY, "broken"); localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "legacy"); localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    expect((await migrateLegacyAuthSession()).status).toBe("invalid")
    expect(inspectAuthSession().status).toBe("invalid")
  })
  it("preserves a profile update when credentials advance", async () => {
    const captured = await created({ user, access_token: "old", refresh_token: "refresh" })
    await updateAuthSessionUser(captured, { ...user, email: "profile@example.com" })
    await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })
    expect(current().cache).toMatchObject({ token: "new", user: { email: "profile@example.com" }, credentialRevision: 1, profileRevision: 1 })
  })
  it("preserves credential advance when profile updates afterwards", async () => {
    const captured = await created({ user, access_token: "old", refresh_token: "refresh" })
    await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })
    await updateAuthSessionUser(captured, { ...user, email: "profile@example.com" })
    expect(current().cache).toMatchObject({ token: "new", user: { email: "profile@example.com" }, credentialRevision: 1, profileRevision: 1 })
  })
  it("keeps existing expiry fields only when a lower-level refresh payload omits them", async () => {
    const captured = await created({ user, access_token: "old", refresh_token: "old-refresh" })
    const before = current().cache

    const committed = await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })

    expect(committed).toMatchObject({ status: "updated", projection: { snapshot: { accessToken: "new", refreshToken: "new-refresh" } } })
    expect(current().cache).toMatchObject({ expiresAt: before.expiresAt, refreshExpiresAt: before.refreshExpiresAt })
  })
  it("rejects a refresh response that does not satisfy the refresh response contract", async () => {
    const captured = await created({ user, access_token: "old", refresh_token: "old-refresh" })

    expect(await commitAuthSessionRefresh(captured, { access_token: "new", refresh_token: "new-refresh" })).toEqual({ status: "invalid" })
    expect(current().snapshot).toMatchObject({ accessToken: "old", refreshToken: "old-refresh" })
  })
  it("publishes an auth update only after the mutation lock is released", async () => {
    let lockHeld = false
    Object.defineProperty(navigator, "locks", { configurable: true, value: {
      request: vi.fn(async (_name: string, action: () => Promise<unknown>) => {
        lockHeld = true
        try { return await action() } finally { lockHeld = false }
      }),
    } })
    const captured = await created({ user, access_token: "old", refresh_token: "old-refresh" })
    const observedWhileLocked: boolean[] = []
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, () => observedWhileLocked.push(lockHeld), { once: true })

    await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })

    expect(observedWhileLocked).toEqual([false])
  })
  it("rejects profile identity changes while accepting numeric zero and nullable metadata", async () => {
    const zero = await created({ user: { id: 0, username: "zero", email: null, is_admin: false }, access_token: "token" })
    expect(zero.userId).toBe("0")
    expect((await updateAuthSessionUser(zero, { id: "other", username: "other" })).status).toBe("invalid")
    expect((await updateAuthSessionUser(zero, { id: 0, username: "zero-2", email: null, is_admin: true })).status).toBe("updated")
  })
  it("reports coordination_unavailable and makes zero storage mutations for conditional operations without locks", async () => {
    const captured = await created()
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    expect(await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })).toEqual({ status: "unavailable", reason: "coordination_unavailable" })
    expect(await createAuthSession({ user, access_token: "late" }, claim.intent)).toEqual({ status: "unavailable", reason: "coordination_unavailable" })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
    expect(current().snapshot.accessToken).toBe("access")
  })
  it("does not let old same-user lineage clear a replacement", async () => {
    const old = await created({ user, access_token: "old", refresh_token: "r" })
    const replacement = await created({ user, access_token: "new", refresh_token: "r2" })
    expect(await clearAuthSessionIfCurrent(old)).toEqual({ status: "not_current" })
    expect(readAuthSessionSnapshot().sessionId).toBe(replacement.sessionId)
  })
  it("reports operation_failed for lock rejection without storage mutation", async () => {
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    Object.defineProperty(navigator, "locks", { configurable: true, value: { request: vi.fn(async () => { throw new Error("lock failed") }) } })
    expect(await claimAuthLoginIntent()).toEqual({ status: "unavailable", reason: "operation_failed" })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
  })
  it("reports operation_failed and makes zero storage mutations when a conditional callback throws", async () => {
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    const before = authStorageSnapshot()
    const originalSetItem = localStorage.setItem.bind(localStorage)
    vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === AUTH_CACHE_KEY) throw new Error("write blocked")
      originalSetItem(key, value)
    })

    expect(await createAuthSession({ user, access_token: "access" }, claim.intent)).toEqual({ status: "unavailable", reason: "operation_failed" })

    expect(authStorageSnapshot()).toEqual(before)
  })
  it("continues rollback after one key fails, restores later keys, and publishes a value-free invalidation", async () => {
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "legacy-access")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    let rollingBack = false
    vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === LEGACY_AUTH_USER_KEY) {
        rollingBack = true
        throw new Error("legacy user cleanup blocked")
      }
      if (rollingBack && key === AUTH_CACHE_KEY) throw new Error("cache rollback blocked")
      originalRemoveItem(key)
    })
    const events: Event[] = []
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, event => events.push(event), { once: true })

    await expect(migrateLegacyAuthSession()).resolves.toEqual({ status: "unavailable", reason: "operation_failed" })

    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe("legacy-access")
    expect(events).toHaveLength(1)
    expect(events[0]).not.toBeInstanceOf(StorageEvent)
  })
  it("does not supersede the local login intent when OIDC session binding is unavailable", async () => {
    localStorage.setItem(AUTH_LOGIN_INTENT_KEY, JSON.stringify({ schemaVersion: 1, id: "existing-intent" }))
    const before = authStorageSnapshot()
    vi.stubGlobal("sessionStorage", undefined)

    await expect(claimOidcAuthLoginIntent()).resolves.toEqual({ status: "unavailable", reason: "storage_unavailable" })

    expect(authStorageSnapshot()).toEqual(before)
  })
  it("compensates the local and session OIDC intent state when session binding fails after the lock begins", async () => {
    localStorage.setItem(AUTH_LOGIN_INTENT_KEY, JSON.stringify({ schemaVersion: 1, id: "existing-intent" }))
    sessionStorage.setItem(AUTH_OIDC_INTENT_KEY, JSON.stringify({ schemaVersion: 1, id: "existing-oidc-intent" }))
    const beforeLocal = authStorageSnapshot()
    const beforeSession = sessionStorage.getItem(AUTH_OIDC_INTENT_KEY)
    const browserSessionStorage = window.sessionStorage
    const descriptor = Object.getOwnPropertyDescriptor(window, "sessionStorage")
    Object.defineProperty(window, "sessionStorage", { configurable: true, value: {
      getItem: browserSessionStorage.getItem.bind(browserSessionStorage),
      removeItem: browserSessionStorage.removeItem.bind(browserSessionStorage),
      setItem: (key: string, value: string) => {
        if (key === AUTH_OIDC_INTENT_KEY) throw new Error("OIDC session write blocked")
        browserSessionStorage.setItem(key, value)
      },
    } })
    try {
      await expect(claimOidcAuthLoginIntent()).resolves.toEqual({ status: "unavailable", reason: "operation_failed" })

      expect(authStorageSnapshot()).toEqual(beforeLocal)
      expect(browserSessionStorage.getItem(AUTH_OIDC_INTENT_KEY)).toBe(beforeSession)
    } finally {
      if (descriptor) Object.defineProperty(window, "sessionStorage", descriptor)
    }
  })
  it("reports OIDC intent storage failures through the typed take result", () => {
    vi.stubGlobal("sessionStorage", undefined)

    expect(takeOidcAuthLoginIntent()).toEqual({ status: "unavailable", reason: "storage_unavailable" })
  })
  it.each(["claim", "oidc_claim", "create", "migrate", "refresh", "profile", "clear"] as const)("reports coordination_unavailable without changing any auth-owned key for %s", async operation => {
    const captured = await created()
    const pending = await claimAuthLoginIntent()
    expect(pending.status).toBe("claimed")
    if (pending.status !== "claimed") throw new Error("expected pending intent")
    const before = authStorageSnapshot()
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })

    const result = await ({
      claim: () => claimAuthLoginIntent(),
      oidc_claim: () => claimOidcAuthLoginIntent(),
      create: () => createAuthSession({ user, access_token: "new-access" }, pending.intent),
      migrate: () => migrateLegacyAuthSession(),
      refresh: () => commitAuthSessionRefresh(captured, { success: true, access_token: "new-access", refresh_token: "new-refresh" }),
      profile: () => updateAuthSessionUser(captured, { ...user, email: "new@example.com" }),
      clear: () => clearAuthSessionIfCurrent(captured),
    })[operation]()

    expect(result).toEqual({ status: "unavailable", reason: "coordination_unavailable" })
    expect(authStorageSnapshot()).toEqual(before)
  })
  it("returns discriminated conditional-clear outcomes", async () => {
    const currentSession = await created()
    await expect(clearAuthSessionIfCurrent(currentSession)).resolves.toEqual({ status: "cleared" })

    const replacement = await created({ user, access_token: "replacement" })
    await expect(clearAuthSessionIfCurrent(currentSession)).resolves.toEqual({ status: "not_current" })
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    await expect(clearAuthSessionIfCurrent(replacement)).resolves.toEqual({ status: "unavailable", reason: "coordination_unavailable" })
  })
  it("rejects malformed login and refresh payloads without writing", async () => {
    const invalid = await createAuthSession({ user: { id: "", username: "alice" }, access_token: "access" })
    expect(invalid.status).toBe("invalid")
    expect(inspectAuthSession().status).toBe("absent")
    const malformedExpiry = await createAuthSession({ user, access_token: "access", expires_in: 0 })
    expect(malformedExpiry.status).toBe("invalid")
  })
  it("rejects blank or unsafe credential expiry values without changing opaque tokens", async () => {
    expect((await createAuthSession({ user, access_token: "   " })).status).toBe("invalid")
    expect((await createAuthSession({ user, access_token: "opaque token ", expires_in: Number.MAX_VALUE })).status).toBe("invalid")
    const opaqueCreated = await created({ user, access_token: "opaque token ", expires_in: 60 })
    expect(opaqueCreated).toMatchObject({ accessToken: "opaque token " })
  })
  it("marks same credential revision with changed token as invalid", async () => {
    const captured = await created()
    const raw = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    raw.token = "tampered"
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(raw))
    expect(compareAuthSession(captured).status).toBe("invalid")
  })
  it("rejects revision regressions and same-revision profile metadata tampering", async () => {
    const captured = await created()
    const advanced = await commitAuthSessionRefresh(captured, { success: true, access_token: "new", refresh_token: "new-refresh" })
    if (advanced.status !== "updated") throw new Error("expected credential advance")
    const regressed = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    regressed.credentialRevision = 0
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(regressed))
    expect(compareAuthSession(advanced.projection.snapshot).status).toBe("invalid")

    const fresh = await created()
    const tampered = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    tampered.user.email = "tampered@example.com"
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(tampered))
    expect(compareAuthSession(fresh).status).toBe("invalid")
  })
  it("rejects credential and profile advancement at the maximum revision without writing", async () => {
    const captured = await created()
    const raw = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    raw.credentialRevision = Number.MAX_SAFE_INTEGER
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(raw))
    const credentialSnapshot = { ...captured, credentialRevision: Number.MAX_SAFE_INTEGER }
    const beforeCredential = localStorage.getItem(AUTH_CACHE_KEY)
    expect((await commitAuthSessionRefresh(credentialSnapshot, { success: true, access_token: "new", refresh_token: "new-refresh" })).status).toBe("invalid")
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(beforeCredential)

    raw.credentialRevision = 0
    raw.profileRevision = Number.MAX_SAFE_INTEGER
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(raw))
    const profileSnapshot = { ...captured, profileRevision: Number.MAX_SAFE_INTEGER }
    const beforeProfile = localStorage.getItem(AUTH_CACHE_KEY)
    expect((await updateAuthSessionUser(profileSnapshot, { ...user, email: "new@example.com" })).status).toBe("invalid")
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(beforeProfile)
  })
  it("does not clear storage when the logout lock is rejected", async () => {
    await created()
    Object.defineProperty(navigator, "locks", { configurable: true, value: { request: vi.fn(async () => { throw new Error("lock failed") }) } })
    await expect(clearStoredAuth()).resolves.toMatchObject({ status: "unavailable", reason: "operation_failed", credentialsCleared: false })
    expect(inspectAuthSession()).toMatchObject({ status: "valid", projection: { snapshot: { accessToken: "access" } } })
  })
  it("allows explicit logout without Web Locks", async () => {
    await created()
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    await expect(clearStoredAuth()).resolves.toMatchObject({ status: "cleared", credentialsCleared: true })
    expect(inspectAuthSession().status).toBe("absent")
  })
  it("removes credentials and invalidates the old login intent when barrier persistence fails", async () => {
    const claim = await claimAuthLoginIntent()
    expect(claim.status).toBe("claimed")
    if (claim.status !== "claimed") throw new Error("expected login intent")
    await createAuthSession({ user, access_token: "access", refresh_token: "refresh" }, claim.intent)
    const originalSetItem = localStorage.setItem.bind(localStorage)
    vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === AUTH_LOGIN_INTENT_KEY) throw new Error("intent write failed")
      originalSetItem(key, value)
    })
    const events: string[] = []
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, () => events.push("updated"), { once: true })

    const result = await clearStoredAuth()

    expect(result).toEqual({ status: "cleared", credentialsCleared: true, barrier: "removed" })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(localStorage.getItem(AUTH_LOGIN_INTENT_KEY)).toBeNull()
    expect(events).toEqual(["updated"])
    await expect(createAuthSession({ user, access_token: "late" }, claim.intent)).resolves.toEqual({ status: "superseded" })
  })
  it("revokes the exact pending login intent when a logout cannot replace or remove it", async () => {
    await created({ user, access_token: "existing-access", refresh_token: "existing-refresh" })
    const pending = await claimAuthLoginIntent()
    expect(pending.status).toBe("claimed")
    if (pending.status !== "claimed") throw new Error("expected pending intent")
    const originalSetItem = localStorage.setItem.bind(localStorage)
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    const setItem = vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === AUTH_LOGIN_INTENT_KEY) throw new Error("intent write failed")
      originalSetItem(key, value)
    })
    const removeItem = vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === AUTH_LOGIN_INTENT_KEY) throw new Error("intent removal failed")
      originalRemoveItem(key)
    })

    const result = await clearStoredAuth()

    expect(result).toMatchObject({ status: "cleared", credentialsCleared: true, barrier: "revoked" })
    await expect(createAuthSession({ user, access_token: "late-access" }, pending.intent)).resolves.toEqual({ status: "superseded" })
    expect(inspectAuthSession().status).toBe("absent")
    setItem.mockRestore()
    removeItem.mockRestore()
    const later = await claimAuthLoginIntent()
    expect(later.status).toBe("claimed")
    if (later.status !== "claimed") throw new Error("expected later intent")
    await expect(createAuthSession({ user, access_token: "later-access" }, later.intent)).resolves.toMatchObject({ status: "created" })
  })
  it("reports unavailable when logout cannot persist a revoked pending intent", async () => {
    await created({ user, access_token: "existing-access", refresh_token: "existing-refresh" })
    const pending = await claimAuthLoginIntent()
    expect(pending.status).toBe("claimed")
    const originalSetItem = localStorage.setItem.bind(localStorage)
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === AUTH_LOGIN_INTENT_KEY || key === "auth_revoked_login_intent") throw new Error("barrier persistence failed")
      originalSetItem(key, value)
    })
    vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === AUTH_LOGIN_INTENT_KEY) throw new Error("intent removal failed")
      originalRemoveItem(key)
    })

    const result = await clearStoredAuth()

    expect(result).toMatchObject({ status: "unavailable", reason: "operation_failed", credentialsCleared: true, barrier: "unavailable" })
    expect(inspectAuthSession().status).toBe("absent")
  })
  it("publishes credential deletion after lock release even when a later removal fails", async () => {
    let lockHeld = false
    Object.defineProperty(navigator, "locks", { configurable: true, value: {
      request: vi.fn(async (_name: string, action: () => Promise<unknown>) => {
        lockHeld = true
        try { return await action() } finally { lockHeld = false }
      }),
    } })
    await created()
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === LEGACY_AUTH_TOKEN_KEY) throw new Error("legacy cleanup failed")
      originalRemoveItem(key)
    })
    const observedWhileLocked: boolean[] = []
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, () => observedWhileLocked.push(lockHeld), { once: true })

    const result = await clearStoredAuth()

    expect(result).toMatchObject({ status: "unavailable", reason: "operation_failed", credentialsCleared: false })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(observedWhileLocked).toEqual([false])
  })
})
