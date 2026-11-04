import zh from "./locales/zh"
import en from "./locales/en"

export const translations = {
  zh,
  en,
} as const

export type Locale = keyof typeof translations
export type TranslationVariables = Record<string, string | number>

type TranslationLeafPaths<T> = {
  [K in Extract<keyof T, string>]: T[K] extends string
    ? K
    : T[K] extends Record<string, unknown>
      ? `${K}.${TranslationLeafPaths<T[K]>}`
      : never
}[Extract<keyof T, string>]

export type TranslationKey = TranslationLeafPaths<typeof en>

interface TranslationResolutionOptions {
  onMissing?: (key: string) => void
}

function isTranslationBranch(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function interpolate(value: string, vars?: TranslationVariables): string {
  if (!vars) return value
  return Object.entries(vars).reduce(
    (result, [key, replacement]) =>
      result.replace(new RegExp(`\\{${key}\\}`, "g"), String(replacement)),
    value,
  )
}

function resolveTranslationValue(
  locale: Locale,
  key: string,
  fallback: string,
  vars?: TranslationVariables,
  options: TranslationResolutionOptions = {},
): string {
  let value: unknown = translations[locale]
  for (const part of key.split(".")) {
    if (!isTranslationBranch(value) || !(part in value)) {
      options.onMissing?.(key)
      return interpolate(fallback, vars)
    }
    value = value[part]
  }

  if (typeof value !== "string") {
    options.onMissing?.(key)
    return interpolate(fallback, vars)
  }
  return interpolate(value, vars)
}

export function resolveTranslation(
  locale: Locale,
  key: TranslationKey,
  vars?: TranslationVariables,
  options: TranslationResolutionOptions = {},
): string {
  return resolveTranslationValue(locale, key, key, vars, options)
}

export function resolveDynamicTranslation(
  locale: Locale,
  key: string,
  fallback: string,
  vars?: TranslationVariables,
  options: TranslationResolutionOptions = {},
): string {
  return resolveTranslationValue(locale, key, fallback, vars, options)
}
