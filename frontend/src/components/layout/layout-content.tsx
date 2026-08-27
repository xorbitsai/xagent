"use client"

import React from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { MobileHeader } from "@/components/layout/mobile-header";
import { AppProvider } from "@/contexts/app-context-chat";
import { isAuthPublicPath, isChromelessAuthenticatedPath } from "@/lib/auth-pages";

interface LayoutContentProps {
  children: React.ReactNode;
}

export function LayoutContent({ children }: LayoutContentProps) {
  const pathname = usePathname();
  const isAuthPage = isAuthPublicPath(pathname);

  if (isAuthPage || isChromelessAuthenticatedPath(pathname)) {
    // Auth pages and full-screen authenticated pages (e.g. onboarding) both
    // render without the sidebar shell - AuthGuard (a parent of this
    // component) is what actually enforces the login requirement for the
    // latter, this check is presentation-only.
    return <>{children}</>;
  }

  // For other pages, show sidebar and main layout
  return (
    <AppProvider>
      <div className="flex h-screen bg-background relative">
        <div className="hidden xl:flex xl:shrink-0">
          <Sidebar />
        </div>
        <div className="flex flex-1 flex-col overflow-hidden bg-background">
          <MobileHeader />
          <main className="flex-1 overflow-hidden bg-background">
            {children}
          </main>
        </div>
      </div>
    </AppProvider>
  );
}
