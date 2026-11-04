export interface BrandingConfig {
  appName: string
  logoPath: string
  whiteLogoPath: string
  logoAlt: string
  subtitle: string
  description: string
  tagline: string
  gradientFrom: string
  gradientVia: string
  gradientTo: string
  siteUrl: string
}

export const defaultBranding: BrandingConfig = {
  appName: 'Xagent',
  logoPath: '/xagent_logo.png',
  whiteLogoPath: '/xagent_white_logo.png',
  logoAlt: 'Xagent Logo',
  subtitle: 'Next generation agent operating system',
  description: 'AI-powered agent and workflow management system',
  tagline: 'AI agent and workflow automation platform',
  gradientFrom: 'blue-400',
  gradientVia: 'blue-500',
  gradientTo: 'indigo-500',
  siteUrl: 'https://cloud.xagent.co',
}

export function getBrandingFromEnv(): BrandingConfig {
  return {
    appName: process.env.NEXT_PUBLIC_APP_NAME || defaultBranding.appName,
    logoPath: process.env.NEXT_PUBLIC_LOGO_PATH || defaultBranding.logoPath,
    whiteLogoPath: process.env.NEXT_PUBLIC_WHITE_LOGO_PATH || defaultBranding.whiteLogoPath,
    logoAlt: process.env.NEXT_PUBLIC_LOGO_ALT || defaultBranding.logoAlt,
    subtitle: process.env.NEXT_PUBLIC_APP_SUBTITLE || defaultBranding.subtitle,
    description: process.env.NEXT_PUBLIC_APP_DESCRIPTION || defaultBranding.description,
    tagline: process.env.NEXT_PUBLIC_APP_TAGLINE || defaultBranding.tagline,
    gradientFrom: process.env.NEXT_PUBLIC_GRADIENT_FROM || defaultBranding.gradientFrom,
    gradientVia: process.env.NEXT_PUBLIC_GRADIENT_VIA || defaultBranding.gradientVia,
    gradientTo: process.env.NEXT_PUBLIC_GRADIENT_TO || defaultBranding.gradientTo,
    siteUrl: process.env.NEXT_PUBLIC_SITE_URL || defaultBranding.siteUrl,
  }
}

// A malformed NEXT_PUBLIC_SITE_URL (e.g. missing scheme) must not take down
// the production build; fall back to the default site URL instead.
//
// Lives here rather than in app/layout.tsx: Next.js's typed-routes checker
// restricts route modules (layout.tsx/page.tsx) to a fixed set of named
// exports, so an extra export there fails `tsc` against .next/types.
export function resolveMetadataBase(siteUrl: string): URL {
  try {
    return new URL(siteUrl)
  } catch {
    console.error(`Invalid NEXT_PUBLIC_SITE_URL "${siteUrl}", falling back to ${defaultBranding.siteUrl}`)
    return new URL(defaultBranding.siteUrl)
  }
}
