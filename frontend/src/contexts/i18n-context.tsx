"use client"

import React, { createContext, useContext, useEffect, useMemo, useState } from "react"
import {
  resolveDynamicTranslation,
  resolveTranslation,
  type Locale,
  type TranslationKey,
  type TranslationVariables,
} from "@/i18n/translations"

export type { Locale } from "@/i18n/translations"

export type Translate = (
  key: TranslationKey,
  vars?: TranslationVariables,
) => string
export type TranslateDynamic = (
  key: string,
  fallback: string,
  vars?: TranslationVariables,
) => string

interface I18nContextValue {
  locale: Locale
  setLocale: (l: Locale) => void
  t: Translate
  tDynamic: TranslateDynamic
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined)
const reportedMissingKeys = new Set<string>()

function reportMissingTranslation(key: string): void {
  if (process.env.NODE_ENV === "production" || reportedMissingKeys.has(key)) return
  reportedMissingKeys.add(key)
  console.warn(`Missing translation key: ${key}`)
}

export function I18nProvider({
  children,
  initialLocale = "en",
}: {
  children: React.ReactNode
  initialLocale?: Locale
}) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale)

  useEffect(() => {
    try {
      const stored = typeof window !== "undefined" ? localStorage.getItem("app_locale") : null
      if (stored === "en" || stored === "zh") {
        if (stored !== locale) {
          setLocaleState(stored as Locale)
        }
      }
    } catch {
      // ignore
    }
  }, [locale])

  const setLocale = (l: Locale) => {
    setLocaleState(l)
    try {
      localStorage.setItem("app_locale", l)
      document.cookie = `app_locale=${l}; path=/; max-age=31536000; samesite=lax`
    } catch {
      // ignore
    }
  }

  // Sync <html lang> attribute
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = locale
    }
  }, [locale])

  const t = useMemo<Translate>(
    () => (key, vars) =>
      resolveTranslation(locale, key, vars, {
        onMissing: reportMissingTranslation,
      }),
    [locale],
  )

  const tDynamic = useMemo<TranslateDynamic>(
    () => (key, fallback, vars) =>
      resolveDynamicTranslation(locale, key, fallback, vars, {
        onMissing: reportMissingTranslation,
      }),
    [locale],
  )

  const value = useMemo(
    () => ({ locale, setLocale, t, tDynamic }),
    [locale, t, tDynamic],
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error("useI18n must be used within I18nProvider")
  return ctx
}
