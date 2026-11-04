"use client"

import React, { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { updateUserPreferences } from "@/lib/user-preferences"

type Defaults = { default_tokens: number | null; max_tokens: number | null; soft_limit_percent: number }
type Budget = {
  effective: { max_tokens: number | null; soft_limit_percent: number; source: string; soft_limit_source: string }
  preferences: { execution_budget_tokens: number | null; execution_budget_soft_percent: number | null }
}

const numberOrNull = (value: string) => value.trim() === "" ? null : Number(value)

export function ExecutionBudgetSettings({ isAdmin = false }: { isAdmin?: boolean }) {
  const { t } = useI18n()
  const [budget, setBudget] = useState<Budget | null>(null)
  const [tokens, setTokens] = useState("")
  const [percent, setPercent] = useState("")
  const [defaults, setDefaults] = useState<Defaults | null>(null)
  const [defaultTokens, setDefaultTokens] = useState("")
  const [maximumTokens, setMaximumTokens] = useState("")
  const [defaultPercent, setDefaultPercent] = useState("80")
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState("")

  useEffect(() => {
    let active = true
    async function load() {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/execution-budget/me`)
        if (!response.ok) throw new Error()
        const data: Budget = await response.json()
        if (!data.effective || !data.preferences) throw new Error()
        if (!active) return
        setBudget(data)
        setTokens(String(data.preferences.execution_budget_tokens ?? ""))
        setPercent(String(data.preferences.execution_budget_soft_percent ?? ""))
        if (isAdmin) {
          const adminResponse = await apiRequest(`${getApiUrl()}/api/execution-budget/defaults`)
          if (!adminResponse.ok) throw new Error()
          const adminData: Defaults = await adminResponse.json()
          if (!active) return
          setDefaults(adminData)
          setDefaultTokens(String(adminData.default_tokens ?? ""))
          setMaximumTokens(String(adminData.max_tokens ?? ""))
          setDefaultPercent(String(adminData.soft_limit_percent))
        }
      } catch {
        if (active) setStatus(t("settings.budget.failed"))
      }
    }
    void load()
    return () => { active = false }
  }, [isAdmin, t])

  async function save(admin: boolean) {
    setSaving(true)
    setStatus("")
    try {
      if (admin) {
        const response = await apiRequest(`${getApiUrl()}/api/execution-budget/defaults`, {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ default_tokens: numberOrNull(defaultTokens), max_tokens: numberOrNull(maximumTokens), soft_limit_percent: Number(defaultPercent) }),
        })
        if (!response.ok) throw new Error()
      } else {
        const result = await updateUserPreferences({ execution_budget_tokens: numberOrNull(tokens), execution_budget_soft_percent: numberOrNull(percent) })
        if (!result.ok) throw new Error()
      }
      const response = await apiRequest(`${getApiUrl()}/api/execution-budget/me`)
      if (!response.ok) throw new Error()
      const data: Budget = await response.json()
      if (!data.effective) throw new Error()
      setBudget(data)
      setStatus(t("settings.budget.saved"))
    } catch {
      setStatus(t("settings.budget.failed"))
    } finally {
      setSaving(false)
    }
  }

  const source = (value: string) => {
    if (value === "personal") return t("settings.budget.sources.personal")
    if (value === "system") return t("settings.budget.sources.system")
    if (value === "system_maximum") return t("settings.budget.sources.system_maximum")
    return t("settings.budget.sources.external")
  }

  return <Card>
    <CardHeader>
      <CardTitle>{t("settings.budget.title")}</CardTitle>
      <CardDescription>{t("settings.budget.description")}</CardDescription>
    </CardHeader>
    <CardContent className="space-y-4">
      {budget && <p className="text-sm" data-testid="effective-budget">
        {t("settings.budget.effective")}: {budget.effective.max_tokens?.toLocaleString() ?? t("settings.budget.unlimited")} ({source(budget.effective.source)}); {budget.effective.soft_limit_percent}% ({source(budget.effective.soft_limit_source)})
      </p>}
      <form className="space-y-3" onSubmit={(event) => { event.preventDefault(); void save(false) }}>
        <Label htmlFor="personal-budget-tokens">{t("settings.budget.tokens")}</Label>
        <Input id="personal-budget-tokens" type="number" min={1} step={1} value={tokens} onChange={(event) => setTokens(event.target.value)} placeholder={t("settings.budget.inherit")} disabled={!budget || saving} />
        <Label htmlFor="personal-budget-percent">{t("settings.budget.percent")}</Label>
        <Input id="personal-budget-percent" type="number" min={1} max={99} step={1} value={percent} onChange={(event) => setPercent(event.target.value)} placeholder={t("settings.budget.inherit")} disabled={!budget || saving} />
        <p className="text-sm text-muted-foreground">{t("settings.budget.personalHelp")}</p>
        <Button type="submit" disabled={!budget || saving}>{t("settings.budget.savePersonal")}</Button>
      </form>
      {isAdmin && <form className="space-y-3 border-t pt-4" onSubmit={(event) => { event.preventDefault(); void save(true) }}>
        <h3>{t("settings.budget.admin")}</h3>
        <Label htmlFor="default-budget-tokens">{t("settings.budget.defaultTokens")}</Label>
        <Input id="default-budget-tokens" type="number" min={1} step={1} value={defaultTokens} onChange={(event) => setDefaultTokens(event.target.value)} disabled={!defaults || saving} placeholder={t("settings.budget.unlimited")} />
        <Label htmlFor="maximum-budget-tokens">{t("settings.budget.maximumTokens")}</Label>
        <Input id="maximum-budget-tokens" type="number" min={1} step={1} value={maximumTokens} onChange={(event) => setMaximumTokens(event.target.value)} disabled={!defaults || saving} placeholder={t("settings.budget.unlimited")} />
        <Label htmlFor="default-budget-percent">{t("settings.budget.percent")}</Label>
        <Input id="default-budget-percent" type="number" min={1} max={99} step={1} required value={defaultPercent} onChange={(event) => setDefaultPercent(event.target.value)} disabled={!defaults || saving} />
        <Button type="submit" disabled={!defaults || saving}>{t("settings.budget.saveDefaults")}</Button>
      </form>}
      {status && <p role="status">{status}</p>}
    </CardContent>
  </Card>
}
