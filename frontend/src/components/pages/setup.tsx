"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { useI18n } from "@/contexts/i18n-context"
import { Database, Lock, ShieldCheck, User, Workflow } from "lucide-react"

export function SetupPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isChecking, setIsChecking] = useState(true)

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/auth/setup-status`)
        if (!response.ok) {
          return
        }
        const data = await response.json()
        if (!data.needs_setup) {
          window.location.href = "/login"
        }
      } finally {
        setIsChecking(false)
      }
    }
    checkStatus()
  }, [])

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
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/10 to-background relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>

      <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-accent/30 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse delay-1000"></div>

      <div className="relative z-10 flex min-h-screen">
        <div className="hidden lg:flex lg:w-1/2 items-center justify-center p-12">
          <div className="max-w-lg">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <img
                  src={branding.logoPath}
                  alt={branding.logoAlt}
                  className="h-16 w-16"
                />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                  {branding.appName}
                </h1>
              </div>
              <p className="text-xl text-muted-foreground leading-relaxed">
                {t("setup.description")}
              </p>
            </div>

            <div className="space-y-6">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start gap-4 group">
                  <div className="h-12 w-12 rounded-lg bg-background/10 backdrop-blur-sm flex items-center justify-center group-hover:bg-accent transition-colors">
                    <feature.icon className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-1">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <div className="w-full max-w-md">
            <div className="lg:hidden text-center mb-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <img
                  src={branding.logoPath}
                  alt={branding.logoAlt}
                  className="h-12 w-12"
                />
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary via-primary/80 to-primary/60 bg-clip-text text-transparent">
                  {branding.appName}
                </h1>
              </div>
            </div>

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
          </div>
        </div>
      </div>
    </div>
  )
}
