"use client"

import { useEffect, useState } from "react"
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
import { useI18n } from "@/contexts/i18n-context"
import { Select } from "@/components/ui/select"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT } from "@/lib/auth-cache"

export default function SettingsPage() {
  const { user } = useAuth()
  const { t, locale, setLocale } = useI18n()
  const [email, setEmail] = useState(user?.email || "")
  const [isProfileLoading, setIsProfileLoading] = useState(true)
  const [isSavingEmail, setIsSavingEmail] = useState(false)
  const [emailMessage, setEmailMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isChangingPassword, setIsChangingPassword] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadProfile = async () => {
      setIsProfileLoading(true)
      try {
        const response = await apiRequest(`${getApiUrl()}/api/auth/me`)
        const data = await response.json()

        if (!isMounted) return

        if (response.ok && data.success) {
          setEmail(data.user.email || "")
          syncCachedUser(data.user)
          setEmailMessage(null)
        } else {
          setEmailMessage({ type: 'error', text: data.message || data.detail || t("settings.email.failed") })
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
  }, [t])

  return (
    <div className="h-full w-full overflow-y-auto p-8 space-y-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-[22px] font-bold leading-tight">{t("settings.title")}</h1>
          <p className="text-[13px] text-muted-foreground mt-0.5">{t("settings.description")}</p>
        </div>
      </div>

      <div className="space-y-6">
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
                onValueChange={(val) => setLocale(val as any)}
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

            <div className="space-y-2">
              <Label htmlFor="account-username">{t("settings.email.username")}</Label>
              <Input
                id="account-username"
                value={user?.username || ""}
                disabled
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="account-email">{t("settings.email.current")}</Label>
              <Input
                id="account-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder={t("settings.email.placeholder")}
                disabled={isProfileLoading || isSavingEmail}
              />
            </div>

            <Button
              onClick={handleEmailUpdate}
              disabled={isProfileLoading || isSavingEmail || !email.trim()}
              className="w-full"
            >
              {isSavingEmail ? t("settings.email.submitting") : t("settings.email.submit")}
            </Button>
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

    if (newPassword.length < 6) {
      setPasswordMessage({ type: 'error', text: t("settings.password.errors.too_short") })
      return
    }

    setIsChangingPassword(true)
    setPasswordMessage(null)

    try {
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

      if (response.ok) {
        setPasswordMessage({ type: 'success', text: t("settings.password.success") })
        setCurrentPassword('')
        setNewPassword('')
        setConfirmPassword('')
      } else {
        setPasswordMessage({ type: 'error', text: data.message || t("settings.password.failed") })
      }
    } catch {
      setPasswordMessage({ type: 'error', text: t("settings.password.errors.network") })
    } finally {
      setIsChangingPassword(false)
    }
  }

  async function handleEmailUpdate() {
    if (!email.trim()) {
      setEmailMessage({ type: 'error', text: t("settings.email.errors.required") })
      return
    }

    setIsSavingEmail(true)
    setEmailMessage(null)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/email`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
        })
      })

      const data = await response.json()

      if (response.ok && data.success) {
        const nextEmail = data.user?.email || ""
        setEmail(nextEmail)
        syncCachedUser(data.user)
        setEmailMessage({ type: 'success', text: t("settings.email.success") })
      } else {
        setEmailMessage({ type: 'error', text: data.message || data.detail || t("settings.email.failed") })
      }
    } catch {
      setEmailMessage({ type: 'error', text: t("settings.email.errors.network") })
    } finally {
      setIsSavingEmail(false)
    }
  }

  function syncCachedUser(nextUser: { id: string; username: string; email?: string | null; is_admin?: boolean }) {
    localStorage.setItem("auth_user", JSON.stringify(nextUser))

    const cached = localStorage.getItem(AUTH_CACHE_KEY)
    if (cached) {
      try {
        const parsed = JSON.parse(cached)
        parsed.user = {
          ...(parsed.user || {}),
          ...nextUser,
        }
        localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(parsed))
        window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, {
          key: AUTH_CACHE_KEY,
          newValue: localStorage.getItem(AUTH_CACHE_KEY),
        }))
      } catch {
        // Ignore cache sync failures and keep the latest profile in auth_user.
      }
    }
  }
}
