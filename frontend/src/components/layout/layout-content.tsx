"use client"

import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { AppProvider } from "@/contexts/app-context-chat";
import { useAuth } from "@/contexts/auth-context";

interface LayoutContentProps {
  children: React.ReactNode;
}

export function LayoutContent({ children }: LayoutContentProps) {
  const pathname = usePathname();
  const { token } = useAuth();
  const isAuthPage =
    pathname === "/login" || pathname === "/register" || pathname === "/setup";

  if (isAuthPage) {
    // For auth pages, just render children without sidebar
    return <>{children}</>;
  }

  // For other pages, show sidebar and main layout
  return (
    <AppProvider token={token || undefined}>
      <div className="flex h-screen bg-background relative">
        <Sidebar />
        <main className="flex-1 flex flex-col overflow-hidden bg-background">
          {children}
        </main>
      </div>
    </AppProvider>
  );
}
