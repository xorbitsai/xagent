"use client";

import type { KeyboardEvent, MouseEvent } from "react";
import { Heart, Play, Star } from "lucide-react";
import type { Template } from "@/types/template";
import { cn } from "@/lib/utils";
import { isNestedInteractiveElement } from "./template-card-utils";

interface FeaturedTemplateCardProps {
  template: Template;
  categoryLabel?: string;
  popularLabel: string;
  runsLabel: string;
  onUse: (templateId: string) => void;
  onLike?: (templateId: string, event: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

function FeaturedConnections({ template }: { template: Template }) {
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

export function FeaturedTemplateCard({
  template,
  categoryLabel,
  popularLabel,
  runsLabel,
  onUse,
  onLike,
  className,
}: FeaturedTemplateCardProps) {
  const handleActivate = () => onUse(template.id);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (isNestedInteractiveElement(event.target, event.currentTarget)) {
      return;
    }

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
        "group flex min-h-[156px] cursor-pointer flex-col overflow-hidden rounded-2xl border border-[#E7EAF3] bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md",
        className
      )}
    >
      <div className="h-[3px] w-full bg-[linear-gradient(90deg,#5B67FF_0%,#8B5CF6_50%,#EC4899_100%)]" />
      <div className="flex flex-1 flex-col p-[18px]">
        <div className="flex items-center justify-between gap-3">
          <span className="text-[10.5px] font-bold uppercase tracking-[0.06em] text-[#16A34A]">
            {categoryLabel || template.category}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-[#FEF3C7] px-[7px] py-[1px] text-[10px] font-bold uppercase tracking-[0.04em] text-[#92400E]">
            <Star className="h-[9px] w-[9px] fill-current" />
            {popularLabel}
          </span>
        </div>

        <h3 className="mt-[10px] line-clamp-2 text-[16px] font-bold leading-[1.3] text-[#111827]">
          {template.name}
        </h3>

        <p className="mt-[6px] line-clamp-2 flex-1 text-[12.5px] leading-[1.5] text-[#6B7280]">
          {template.description}
        </p>

        <div className="mt-[14px] flex items-center justify-between gap-[10px]">
          <FeaturedConnections template={template} />
          <div className="flex items-center gap-[10px] text-[11.5px] text-[#6B7280]">
            <div className="flex items-center gap-1">
              <Play className="h-[10px] w-[10px] fill-current text-[#64748B]" />
              <span>{template.used_count ?? 0} {runsLabel}</span>
            </div>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onLike?.(template.id, event);
              }}
              className={cn(
                "flex items-center gap-1 transition-colors",
                template.is_liked ? "text-pink-500" : onLike ? "hover:text-pink-500" : "cursor-default"
              )}
            >
              <Heart
                className={cn(
                  "h-[11px] w-[11px] fill-current",
                  template.is_liked ? "text-rose-500" : "text-rose-400/70"
                )}
              />
              <span>{template.likes ?? 0}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
