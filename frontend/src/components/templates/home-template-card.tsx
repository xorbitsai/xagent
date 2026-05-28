"use client";

import type { KeyboardEvent } from "react";
import {
  Bot,
  Briefcase,
  Clock3,
  Megaphone,
  MessageCircle,
  Play,
  Search,
  Sparkles,
} from "lucide-react";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";

interface HomeTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  runsLabel: string;
  onUse: (templateId: string) => void;
  className?: string;
}

const CATEGORY_ICON_MAP = {
  sales: Briefcase,
  marketing: Megaphone,
  support: MessageCircle,
  research: Search,
  productivity: Bot,
} as const;

function getTemplateIcon(category?: string) {
  const key = category?.toLowerCase().replace(/\s+/g, "_") as keyof typeof CATEGORY_ICON_MAP | undefined;
  return (key && CATEGORY_ICON_MAP[key]) || Sparkles;
}

function HomeConnections({ template }: { template: Template }) {
  const visibleConnections = template.connections?.slice(0, 3) || [];

  return (
    <div className="flex items-center gap-1">
      {visibleConnections.map((connection, index) => (
        <div
          key={`${connection.name}-${index}`}
          className="flex h-5 w-5 items-center justify-center overflow-hidden rounded-md"
        >
          {connection.logo ? (
            <img src={connection.logo} alt={connection.name} className="h-3.5 w-3.5 object-contain" />
          ) : (
            <span className="text-[8px] font-bold text-white">
              {(connection.name || "").substring(0, 1).toUpperCase()}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export function HomeTemplateCard({
  template,
  categoryLabel,
  runsLabel,
  onUse,
  className,
}: HomeTemplateCardProps) {
  const handleActivate = () => onUse(template.id);
  const Icon = getTemplateIcon(template.category);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleActivate();
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleActivate}
      onKeyDown={handleKeyDown}
      className={cn(
        "group flex min-h-[154px] w-full min-w-0 cursor-pointer flex-col overflow-hidden rounded-2xl border border-[#E7EAF3] bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md",
        className
      )}
    >
      <div className="h-[3px] w-full bg-[linear-gradient(90deg,#5B67FF_0%,#8B5CF6_50%,#EC4899_100%)]" />
      <div className="flex flex-1 flex-col p-[14px]">
        <div className="flex items-center justify-between gap-3">
          <div className="grid h-[30px] w-[30px] place-items-center rounded-[8px] bg-[linear-gradient(135deg,#EFF4FF,#F5F3FF)] text-[#3B5AF6]">
            <Icon className="h-[14px] w-[14px] text-primary" />
          </div>
          <span className="inline-flex items-center gap-1 text-[10.5px] text-[#94A3B8]">
            <Clock3 className="h-[10px] w-[10px]" />
            {template.setup_time}
          </span>
        </div>

        <h3 className="mt-3 text-[13.5px] font-bold leading-5 text-[#111827]">
          {template.name}
        </h3>

        <div className="mt-[3px] text-[10.5px] font-semibold uppercase tracking-[0.04em] text-gray-500">
          {categoryLabel || template.category}
        </div>

        <p className="mt-2 line-clamp-2 min-h-[36px] flex-1 text-[12px] leading-[1.45] text-[#6B7280]">
          {template.description}
        </p>

        <div className="mt-3 flex items-center gap-2 border-t border-[#E5E7EB] pt-[10px]">
          <HomeConnections template={template} />
          <div className="flex-1" />
          <div className="flex items-center gap-1 text-[11px] text-[#94A3B8]">
            <Play className="h-[11px] w-[11px] fill-current text-[#64748B]" />
            <span>{template.used_count ?? 0} {runsLabel}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
