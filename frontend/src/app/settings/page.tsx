"use client"

import React, { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Settings,
  Lock,
  User,
} from "lucide-react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import { useI18n, type Locale } from "@/contexts/i18n-context"
import { Select } from "@/components/ui/select"
import { clearAuthSessionIfCurrent, inspectAuthSession, updateAuthSessionUser } from "@/lib/auth-cache"
import { TaskRuntimeSettingsExtension } from "@/lib/task-runtime-ui-extension"
import { ExecutionBudgetSettings } from "@/components/settings/execution-budget"
import { useSetupStatus } from "@/hooks/use-setup-status"

function emailErrorText(data: { message?: unknown; detail?: unknown }, fallback: string): string {
  if (typeof data.message === "string" && data.message) return data.message
  if (typeof data.detail === "string" && data.detail) return data.detail
  return fallback
}

export default function SettingsPage() {
  const { user, session } = useAuth()
  const { t, locale, setLocale } = useI18n()
  const [email, setEmail] = useState(user?.email || "")
  const [emailCurrentPassword, setEmailCurrentPassword] = useState("")
  const [emailConfirmationToken, setEmailConfirmationToken] = useState("")
  const [pendingEmail, setPendingEmail] = useState<string | null>(null)
  const [emailChangeAvailable, setEmailChangeAvailable] = useState(true)
  const [confirmationMinutes, setConfirmationMinutes] = useState(15)
  const [isProfileLoading, setIsProfileLoading] = useState(true)
  const [isSavingEmail, setIsSavingEmail] = useState(false)
  const [emailMessage, setEmailMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isChangingPassword, setIsChangingPassword] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const { passwordMinLength } = useSetupStatus()
  const hasPassword = user?.has_password !== false

  useEffect(() => {
    let isMounted = true

    const loadProfile = async () => {
      setIsProfileLoading(true)
      try {
        const capturedSession = session
        const response = await apiRequest(`${getApiUrl()}/api/auth/me`)
        const data = await response.json()

        if (!isMounted) return

        if (response.ok && data.success) {
          const updated = await syncCachedUser(capturedSession, data.user)
          if (!isMounted) return
          if (updated.status === "updated") {
            setEmail(data.user.email || "")
            setPendingEmail(data.user.pending_email || null)
            setEmailChangeAvailable(data.user.email_change_available !== false)
            setConfirmationMinutes(data.user.email_confirmation_expire_minutes || 15)
            setEmailMessage(null)
          } else if (updated.status === "advanced") {
            applyCanonicalEmail()
          }
        } else {
          setEmailMessage({ type: 'error', text: emailErrorText(data, t("settings.email.failed")) })
        }
      } catch {
        if (isMounted) {
          setEmailMessage({ type: 'error', text: t("settings.email.errors.network") })
        }
      } finally {
        if (isMounted) {
          setIsProfileLoading(false)
        }
      }
    }

    void loadProfile()

    return () => {
      isMounted = false
    }
  }, [session.sessionId, t])

  return (
    <div className="h-full w-full overflow-y-auto p-8 space-y-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-[22px] font-bold leading-tight">{t("settings.title")}</h1>
          <p className="text-[13px] text-muted-foreground mt-0.5">{t("settings.description")}</p>
        </div>
      </div>

      <div className="space-y-6">
        <TaskRuntimeSettingsExtension />

        {/* Language Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              {t("settings.language.title")}
            </CardTitle>
            <CardDescription>
              {t("settings.language.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="language-select">{t("settings.language.title")}</Label>
              <Select
                value={locale}
                onValueChange={(val) => setLocale(val as Locale)}
                options={[
                  { value: "zh", label: "简体中文" },
                  { value: "en", label: "English" },
                ]}
                placeholder={t("settings.language.title")}
              />
            </div>
          </CardContent>
        </Card>

        {/* Password Change Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              {t("settings.email.title")}
            </CardTitle>
            <CardDescription>
              {t("settings.email.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {emailMessage && (
              <Alert className={emailMessage.type === 'error' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                <AlertDescription className={emailMessage.type === 'error' ? 'text-red-800' : 'text-green-800'}>
                  {emailMessage.text}
                </AlertDescription>
              </Alert>
            )}

            {!user?.email?.trim() && user?.username?.trim() && (
              <div className="space-y-2">
                <Label htmlFor="account-username">{t("settings.email.username")}</Label>
                <Input id="account-username" value={user.username} disabled />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="account-email">{t("settings.email.current")}</Label>
              <Input
                id="account-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder={t("settings.email.placeholder")}
                disabled={isProfileLoading || isSavingEmail || !hasPassword || !emailChangeAvailable}
              />
            </div>

            {!hasPassword ? (
              <p className="text-sm text-muted-foreground">{t("settings.email.providerManaged")}</p>
            ) : !emailChangeAvailable && (
              <p className="text-sm text-muted-foreground">{t("settings.email.smtpRequired")}</p>
            )}

            {hasPassword && (
              <div className="space-y-2">
                <Label htmlFor="email-current-password">{t("settings.password.current")}</Label>
                <Input
                  id="email-current-password"
                  type="password"
                  value={emailCurrentPassword}
                  onChange={(e) => setEmailCurrentPassword(e.target.value)}
                  placeholder={t("settings.password.current_placeholder")}
                  disabled={isProfileLoading || isSavingEmail}
                />
              </div>
            )}

            <Button
              onClick={() => void handleEmailUpdate()}
              disabled={isProfileLoading || isSavingEmail || !hasPassword || !emailChangeAvailable || !email.trim() || !emailCurrentPassword}
              className="w-full"
            >
              {isSavingEmail ? t("settings.email.submitting") : t("settings.email.submit")}
            </Button>
            {hasPassword && pendingEmail && <div className="space-y-2">
              <p className="text-sm">{t("settings.email.pendingAddress", { email: pendingEmail })}</p>
              <Label htmlFor="email-confirmation-code">{t("settings.email.confirmationCode")}</Label>
              <Input
                id="email-confirmation-code"
                value={emailConfirmationToken}
                onChange={(event) => setEmailConfirmationToken(event.target.value)}
                autoComplete="off"
                maxLength={128}
                disabled={isProfileLoading || isSavingEmail}
              />
              <p className="text-sm text-muted-foreground">{t("settings.email.confirmationHint", { minutes: confirmationMinutes })}</p>
              <Button
                onClick={() => void handleEmailUpdate(true)}
                disabled={isProfileLoading || isSavingEmail || !emailConfirmationToken.trim()}
                className="w-full"
              >
                {t("settings.email.confirm")}
              </Button>
            </div>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              {t("settings.password.title")}
            </CardTitle>
            <CardDescription>
              {t("settings.password.description")}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {passwordMessage && (
              <Alert className={passwordMessage.type === 'error' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                <AlertDescription className={passwordMessage.type === 'error' ? 'text-red-800' : 'text-green-800'}>
                  {passwordMessage.text}
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="current-password">{t("settings.password.current")}</Label>
              <Input
                id="current-password"
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                placeholder={t("settings.password.current_placeholder")}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="new-password">{t("settings.password.new")}</Label>
              <Input
                id="new-password"
                type="password"
                minLength={passwordMinLength}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder={t("settings.password.new_placeholder")}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirm-password">{t("settings.password.confirm")}</Label>
              <Input
                id="confirm-password"
                type="password"
                minLength={passwordMinLength}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder={t("settings.password.confirm_placeholder")}
              />
            </div>

            <Button
              onClick={handlePasswordChange}
              disabled={isChangingPassword || !currentPassword || !newPassword || !confirmPassword}
              className="w-full"
            >
              {isChangingPassword ? t("settings.password.submitting") : t("settings.password.submit")}
            </Button>
          </CardContent>
        </Card>
        <ExecutionBudgetSettings key={session.sessionId} isAdmin={user?.is_admin} />
      </div>
    </div>
  )

  async function handlePasswordChange() {
    if (!currentPassword || !newPassword || !confirmPassword) {
      setPasswordMessage({ type: 'error', text: t("settings.password.errors.fill_all") })
      return
    }

    if (newPassword !== confirmPassword) {
      setPasswordMessage({ type: 'error', text: t("settings.password.errors.mismatch") })
      return
    }

    if (newPassword.length < passwordMinLength) {
      setPasswordMessage({
        type: 'error',
        text: t("settings.password.errors.too_short", {
          min: passwordMinLength,
        }),
      })
      return
    }

    setIsChangingPassword(true)
    setPasswordMessage(null)

    try {
      const capturedSession = session
      const response = await apiRequest(`${getApiUrl()}/api/auth/change-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword
        })
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setPasswordMessage({ type: 'success', text: t("settings.password.sign_in_again") })
        setCurrentPassword('')
        setNewPassword('')
        setConfirmPassword('')
        // A refresh may have advanced credentials during this request. Clear
        // only this login lineage, never a replacement login in another tab.
        const current = inspectAuthSession()
        if (current.status === "valid" && current.projection.snapshot.sessionId === capturedSession.sessionId) {
          const cleared = await clearAuthSessionIfCurrent(current.projection.snapshot)
          if (cleared.status === "cleared") window.location.href = "/login?password_changed=1"
        }
      } else {
        setPasswordMessage({ type: 'error', text: data.message || t("settings.password.failed") })
      }
    } catch {
      setPasswordMessage({ type: 'error', text: t("settings.password.errors.network") })
    } finally {
      setIsChangingPassword(false)
    }
  }

  async function handleEmailUpdate(confirm = false) {
    if (!confirm && !email.trim()) {
      setEmailMessage({ type: 'error', text: t("settings.email.errors.required") })
      return
    }

    setIsSavingEmail(true)
    setEmailMessage(null)

    try {
      const capturedSession = session
      const response = await apiRequest(`${getApiUrl()}/api/auth/email${confirm ? "/confirm" : ""}`, {
        method: confirm ? 'POST' : 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(confirm ? { token: emailConfirmationToken.trim() } : {
          email,
          current_password: emailCurrentPassword,
        })
      })

      const data = await response.json()

      if (response.ok && data.success) {
        const nextEmail = data.user?.email || ""
        const nextPendingEmail = data.user?.pending_email || (data.pending_confirmation ? email.trim() : null)
        const messageKey = data.unchanged
          ? nextPendingEmail ? "settings.email.unchangedPending" : "settings.email.unchanged"
          : data.pending_confirmation ? "settings.email.confirmationSent" : "settings.email.success"
        const updated = await syncCachedUser(capturedSession, data.user)
        if (updated.status === "updated") {
          setEmail(nextEmail)
        } else if (updated.status === "advanced") {
          // Keep the newer canonical profile, but retain this request's pending
          // confirmation UI. Never carry it across a replacement login.
          applyCanonicalEmail()
        } else {
          applyCanonicalEmail()
          setEmailMessage({ type: 'error', text: t("settings.email.failed") })
          return
        }
        setPendingEmail(nextPendingEmail)
        setEmailCurrentPassword('')
        setEmailConfirmationToken('')
        setEmailMessage({ type: 'success', text: t(messageKey) })
      } else {
        const localizedError = data.error_code === "smtp_not_configured"
          ? t("settings.email.smtpRequired")
          : data.error_code === "provider_managed" ? t("settings.email.providerManaged") : null
        if (data.error_code === "smtp_not_configured") setEmailChangeAvailable(false)
        setEmailMessage({ type: 'error', text: localizedError || emailErrorText(data, t("settings.email.failed")) })
      }
    } catch {
      setEmailMessage({ type: 'error', text: t("settings.email.errors.network") })
    } finally {
      setIsSavingEmail(false)
    }
  }

  async function syncCachedUser(
    capturedSession: typeof session,
    nextUser: { id: string; username: string; email?: string | null; is_admin?: boolean; has_password?: boolean },
  ) {
    return updateAuthSessionUser(capturedSession, nextUser)
  }

  function applyCanonicalEmail() {
    setPendingEmail(null)
    const inspection = inspectAuthSession()
    if (inspection.status === "valid") setEmail(inspection.projection.cache.user.email || "")
  }
}
