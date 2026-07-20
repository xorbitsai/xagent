import React, { type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: ReactNode;
  description?: ReactNode;
  /** Rendered inline next to the title (e.g. a count badge). */
  titleExtra?: ReactNode;
  /** Right-aligned actions (search, create button, …). */
  actions?: ReactNode;
  className?: string;
}

/**
 * Shared top bar for full-page views: a full-width bar with a bottom border,
 * a 22px title over a 13px muted description on the left, and optional actions
 * on the right. Keeps the header style consistent across pages so it can't
 * drift per-page.
 */
export function PageHeader({ title, description, titleExtra, actions, className }: PageHeaderProps) {
  return (
    <div
      className={cn(
        "flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-border/60 px-6 md:px-8 py-5 md:py-6",
        className,
      )}
    >
      <div className="w-full sm:w-auto">
        {titleExtra ? (
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-[22px] font-bold leading-tight">{title}</h1>
            {titleExtra}
          </div>
        ) : (
          <h1 className="text-[22px] font-bold leading-tight">{title}</h1>
        )}
        {description ? (
          <p className="text-[13px] text-muted-foreground mt-0.5">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-3 w-full sm:w-auto">{actions}</div> : null}
    </div>
  );
}
