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
  }
}
