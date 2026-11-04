"use client";

import { ShieldAlert, ShieldCheck } from "lucide-react";

import { useI18n } from "@/contexts/i18n-context";
import type { ScanStatus, SkillSource } from "@/types/skill-hub";

interface SourceBadge {
  label: "builtin" | "user" | "team" | "external"
  classes: string
}

export function badgeForSource(source: SkillSource): SourceBadge {
  switch (source) {
    case "builtin":
      return { label: "builtin", classes: "bg-violet-500/10 text-violet-600 border-violet-500/30" };
    case "user":
      return { label: "user", classes: "bg-emerald-500/10 text-emerald-600 border-emerald-500/30" };
    case "team":
      return { label: "team", classes: "bg-blue-500/10 text-blue-600 border-blue-500/30" };
    default:
      return { label: "external", classes: "bg-amber-500/10 text-amber-600 border-amber-500/30" };
  }
}

export function ScanBadge({ status }: { status: ScanStatus }) {
  const { t } = useI18n();
  if (status === "clean") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600">
        <ShieldCheck className="h-3 w-3" /> {t("skillHub.discover.scanBadge.clean")}
      </span>
    );
  }
  if (status === "suspicious") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-600">
        <ShieldAlert className="h-3 w-3" /> {t("skillHub.discover.scanBadge.flagged")}
      </span>
    );
  }
  if (status === "malicious") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-medium text-rose-600">
        <ShieldAlert className="h-3 w-3" /> {t("skillHub.discover.scanBadge.malicious")}
      </span>
    );
  }
  return null;
}
