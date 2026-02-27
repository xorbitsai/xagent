import type { Metadata } from "next";
import type { CSSProperties } from "react";
import { cookies, headers } from "next/headers";
import "./globals.css";
import { AuthProvider } from "@/contexts/auth-context";
import { ThemeProvider } from "@/contexts/theme-context";
import { AuthGuard } from "@/components/auth/auth-guard";
import { LayoutContent } from "@/components/layout/layout-content";
import { getBrandingFromEnv } from "@/lib/branding";
import { I18nProvider } from "@/contexts/i18n-context";
import { getThemeFromEnv, themes } from "@/lib/theme";
import { Toaster } from "@/components/ui/sonner";

const branding = getBrandingFromEnv();

export const metadata: Metadata = {
  title: branding.appName,
  description: branding.description,
  icons: {
    icon: branding.logoPath,
    apple: branding.logoPath,
  },
};

type Locale = "en" | "zh";

const resolveInitialLocale = async (): Promise<Locale> => {
  try {
    const cookieStore = await cookies();
    const cookieLocale = cookieStore.get("app_locale")?.value;
    if (cookieLocale === "en" || cookieLocale === "zh") {
      return cookieLocale;
    }

    const headerStore = await headers();
    const acceptLanguage = headerStore.get("accept-language")?.toLowerCase() || "";
    return acceptLanguage.includes("zh") ? "zh" : "en";
  } catch {
    return "en";
  }
};

const buildThemeStyle = (themeName: string): CSSProperties => {
  const theme = themes[themeName] || themes.dark;
  const styleVars: Record<string, string> = {
    "--background": theme.colors.background,
    "--foreground": theme.colors.foreground,
    "--card": theme.colors.card,
    "--card-foreground": theme.colors.cardForeground,
    "--popover": theme.colors.popover,
    "--popover-foreground": theme.colors.popoverForeground,
    "--primary": theme.colors.primary,
    "--primary-foreground": theme.colors.primaryForeground,
    "--secondary": theme.colors.secondary,
    "--secondary-foreground": theme.colors.secondaryForeground,
    "--muted": theme.colors.muted,
    "--muted-foreground": theme.colors.mutedForeground,
    "--accent": theme.colors.accent,
    "--accent-foreground": theme.colors.accentForeground,
    "--destructive": theme.colors.destructive,
    "--destructive-foreground": theme.colors.destructiveForeground,
    "--border": theme.colors.border,
    "--input": theme.colors.input,
    "--ring": theme.colors.ring,
  };

  if (theme.colors.cardHover) {
    styleVars["--card-hover"] = theme.colors.cardHover;
  }
  if (theme.colors.borderHighlight) {
    styleVars["--border-highlight"] = theme.colors.borderHighlight;
  }
  if (theme.colors.accentBg) {
    styleVars["--accent-bg"] = theme.colors.accentBg;
  }
  if (theme.colors.accentBorder) {
    styleVars["--accent-border"] = theme.colors.accentBorder;
  }
  if (theme.colors.shadowColor) {
    styleVars["--shadow-color"] = theme.colors.shadowColor;
  }
  if (theme.colors.gradientFrom) {
    styleVars["--gradient-from"] = theme.colors.gradientFrom;
  }
  if (theme.colors.gradientTo) {
    styleVars["--gradient-to"] = theme.colors.gradientTo;
  }
  if (theme.colors.sidebarActiveBgFrom) {
    styleVars["--sidebar-active-bg-from"] = theme.colors.sidebarActiveBgFrom;
  }
  if (theme.colors.sidebarActiveBgTo) {
    styleVars["--sidebar-active-bg-to"] = theme.colors.sidebarActiveBgTo;
  }
  if (theme.colors.sidebarActiveText) {
    styleVars["--sidebar-active-text"] = theme.colors.sidebarActiveText;
  }
  if (theme.colors.sidebarActiveBorder) {
    styleVars["--sidebar-active-border"] = theme.colors.sidebarActiveBorder;
  }

  return styleVars as CSSProperties;
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const initialLocale = await resolveInitialLocale();
  const themeName = getThemeFromEnv();
  const theme = themes[themeName] || themes.dark;
  const themeMode = theme.mode === "light" ? "light" : "dark";
  const themeStyle = buildThemeStyle(themeName);

  return (
    <html lang={initialLocale} className={themeMode} style={themeStyle} suppressHydrationWarning>
      <body
        className={`antialiased bg-background text-foreground theme-${themeName}`}
        suppressHydrationWarning
      >
        <I18nProvider initialLocale={initialLocale}>
          <ThemeProvider>
            <AuthProvider>
              <AuthGuard>
                <LayoutContent>{children}</LayoutContent>
                <Toaster />
              </AuthGuard>
            </AuthProvider>
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
