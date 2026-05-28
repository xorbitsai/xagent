"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Menu } from "lucide-react";
import { Sidebar } from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { useI18n } from "@/contexts/i18n-context";
import { getBrandingFromEnv } from "@/lib/branding";

interface HeaderMeta {
  eyebrow: string;
  title: string;
}

function resolveHeaderMeta(pathname: string, t: (key: string) => string, appName: string): HeaderMeta {
  if (pathname === "/") {
    return { eyebrow: appName, title: t("nav.home") };
  }
  if (pathname.startsWith("/task")) {
    return { eyebrow: t("nav.sections.agentDevelopment"), title: t("nav.task") };
  }
  if (pathname.startsWith("/build")) {
    return { eyebrow: t("nav.sections.agentDevelopment"), title: t("nav.build") };
  }
  if (pathname.startsWith("/templates")) {
    return { eyebrow: t("nav.sections.agentDevelopment"), title: t("nav.templates") };
  }
  if (pathname.startsWith("/kb")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.knowledgeBase") };
  }
  if (pathname.startsWith("/models")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.models") };
  }
  if (pathname.startsWith("/memory")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.memory") };
  }
  if (pathname.startsWith("/tools")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.tools") };
  }
  if (pathname.startsWith("/files")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.files") };
  }
  if (pathname.startsWith("/channels")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.channels") };
  }
  if (pathname.startsWith("/monitoring")) {
    return { eyebrow: t("nav.sections.resources"), title: t("nav.monitoring") };
  }
  if (pathname.startsWith("/settings")) {
    return { eyebrow: appName, title: t("nav.settings") };
  }
  if (pathname.startsWith("/users")) {
    return { eyebrow: appName, title: t("nav.userManagement") };
  }
  if (pathname.startsWith("/admin-mcp")) {
    return { eyebrow: appName, title: t("nav.adminMcp") };
  }
  if (pathname.startsWith("/dashboard")) {
    return { eyebrow: appName, title: t("nav.dashboard") };
  }
  return { eyebrow: appName, title: appName };
}

export function AppHeader() {
  const pathname = usePathname();
  const branding = getBrandingFromEnv();
  const { t } = useI18n();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    setIsMenuOpen(false);
  }, [pathname]);

  const headerMeta = useMemo(() => resolveHeaderMeta(pathname, t, branding.appName), [pathname, t, branding.appName]);

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/85 sm:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="xl:hidden">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Open navigation</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-[85vw] max-w-sm p-0">
            <Sidebar className="w-full border-r-0" allowCollapse={false} />
          </SheetContent>
        </Sheet>

        <Link href="/" className="flex items-center gap-3 xl:hidden">
          <img src={branding.logoPath} alt={branding.logoAlt} className="h-9 w-9 rounded-lg" />
          <span className="max-w-[9rem] truncate text-base font-semibold text-foreground">
            {branding.appName}
          </span>
        </Link>

        <div className="hidden min-w-0 xl:block">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground/80">
            {headerMeta.eyebrow}
          </p>
          <h1 className="truncate text-lg font-semibold text-foreground">{headerMeta.title}</h1>
        </div>
      </div>
    </header>
  );
}
