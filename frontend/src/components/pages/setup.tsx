"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { useI18n } from "@/contexts/i18n-context"
import { Database, Lock, ShieldCheck, User, Workflow } from "lucide-react"
import { useSetupStatus } from "@/hooks/use-setup-status"
import { AuthPageShell } from "@/components/auth/auth-page-shell"

export function SetupPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const { isLoading: isChecking } = useSetupStatus({
    redirectToLoginIfInitialized: true,
  })

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!username || !password) {
      setError(t("setup.errors.required"))
      return
    }
    if (password.length < 6) {
      setError(t("setup.errors.passwordTooShort"))
      return
    }
    if (password !== confirmPassword) {
      setError(t("setup.errors.passwordMismatch"))
      return
    }

    setIsLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/setup-admin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })
      const data = await response.json()
      if (response.ok && data.success) {
        window.location.href = "/login"
        return
      }
      setError(data.message || t("setup.errors.failed"))
    } catch {
      setError(t("setup.errors.failed"))
    } finally {
      setIsLoading(false)
    }
  }

  if (isChecking) {
    return null
  }

  const features = [
    {
      icon: Workflow,
      title: t("setup.features.bootstrap.title"),
      description: t("setup.features.bootstrap.description"),
    },
    {
      icon: Database,
      title: t("setup.features.config.title"),
      description: t("setup.features.config.description"),
    },
    {
      icon: ShieldCheck,
      title: t("setup.features.security.title"),
      description: t("setup.features.security.description"),
    },
  ]

  return (
    <AuthPageShell
      appName={branding.appName}
      logoPath={branding.logoPath}
      logoAlt={branding.logoAlt}
      leftDescription={t("setup.description")}
      features={features}
    >
      <Card className="p-8 bg-background/10 backdrop-blur-lg border-border shadow-2xl">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-foreground mb-2">
                  {t("setup.title", { appName: branding.appName })}
                </h2>
                <p className="text-muted-foreground">{t("setup.description")}</p>
              </div>

              <form onSubmit={onSubmit} className="space-y-6">
                {error ? (
                  <div className="p-3 rounded-lg bg-destructive/20 border border-destructive/50">
                    <p className="text-sm text-destructive-foreground">{error}</p>
                  </div>
                ) : null}

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("setup.form.username")}
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder={t("setup.form.username")}
                      className="pl-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("setup.form.password")}
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder={t("setup.form.password")}
                      className="pl-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("setup.form.confirmPassword")}
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder={t("setup.form.confirmPassword")}
                      className="pl-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3"
                  disabled={isLoading}
                >
                  {isLoading ? t("setup.form.submitting") : t("setup.form.submit")}
                </Button>
              </form>
      </Card>
    </AuthPageShell>
  )
}
