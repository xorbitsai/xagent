"use client";

import React from "react";
import { cn } from "@/lib/utils";

export interface SegmentedTabItem {
  id: string;
  label: React.ReactNode;
}

interface SegmentedTabsProps {
  items: SegmentedTabItem[];
  value: string;
  onValueChange: (value: string) => void;
  className?: string;
  listClassName?: string;
  triggerClassName?: string;
  activeTriggerClassName?: string;
  inactiveTriggerClassName?: string;
}

export function SegmentedTabs({
  items,
  value,
  onValueChange,
  className,
  listClassName,
  triggerClassName,
  activeTriggerClassName = "bg-white text-[#111827] shadow-sm ring-1 ring-[#E7EAF3]",
  inactiveTriggerClassName = "text-[#64748B] hover:bg-white/70 hover:text-[#111827]",
}: SegmentedTabsProps) {
  return (
    <div className={cn("flex flex-wrap items-center", className)}>
      <div
        className={cn(
          "flex flex-wrap items-center gap-1 rounded-[18px] bg-[#EEF2F7] p-[3px]",
          listClassName
        )}
      >
        {items.map((item) => {
          const isActive = item.id === value;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onValueChange(item.id)}
              aria-pressed={isActive}
              className={cn(
                "rounded-full px-3 py-1.5 text-[11px] font-semibold transition-all",
                triggerClassName,
                isActive ? activeTriggerClassName : inactiveTriggerClassName
              )}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
