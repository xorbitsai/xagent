"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useEffect, useState } from "react"
import { Menu } from "lucide-react"
import { getBrandingFromEnv } from "@/lib/branding"
import { Sidebar } from "@/components/layout/sidebar"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet"

export function MobileHeader() {
  const pathname = usePathname()
  const branding = getBrandingFromEnv()
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  useEffect(() => {
    setIsMenuOpen(false)
  }, [pathname])

  return (
    <header className="xl:hidden sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-border bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <Link href="/" className="flex items-center gap-3">
        <img
          src={branding.logoPath}
          alt={branding.logoAlt}
          className="h-9 w-9 rounded-lg"
        />
        <span className="max-w-[9rem] truncate text-base font-semibold text-foreground">
          {branding.appName}
        </span>
      </Link>
      <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="h-10 w-10">
            <Menu className="h-5 w-5" />
            <span className="sr-only">Open navigation</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[85vw] max-w-sm p-0">
          <Sidebar className="w-full border-r-0" allowCollapse={false} />
        </SheetContent>
      </Sheet>
    </header>
  )
}
