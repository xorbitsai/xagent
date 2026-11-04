"use client";

import { useCallback } from "react";

import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { useI18n } from "@/contexts/i18n-context";
import { cn } from "@/lib/utils";

/**
 * Side-by-side SKILL.md editor: raw textarea on the left, live
 * markdown preview on the right. Used by both the "create new
 * skill" page and the in-place edit mode on the detail page.
 *
 * Intentionally minimal — no syntax highlighting, no Monaco. v0
 * trades editor power for bundle size + zero setup. Bumping to
 * Monaco later is a swap of this one component.
 */
export function MarkdownEditor({
  value,
  onChange,
  placeholder,
  className,
  rows = 22,
}: {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  className?: string;
  rows?: number;
}) {
  const { t } = useI18n();
  // Tab-key indent — sounds tiny but writing YAML frontmatter without
  // tab support is genuinely annoying.
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key !== "Tab") return;
      e.preventDefault();
      const ta = e.currentTarget;
      const { selectionStart: s, selectionEnd: end, value: v } = ta;
      const next = `${v.slice(0, s)}  ${v.slice(end)}`;
      onChange(next);
      // Restore caret after React rerenders.
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = s + 2;
      });
    },
    [onChange],
  );

  return (
    <div className={cn("grid gap-3 lg:grid-cols-2", className)}>
      <div className="flex flex-col">
        <div className="mb-2 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
          {t("skillHub.editor.skillMd")}
        </div>
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={rows}
          spellCheck={false}
          className="w-full flex-1 resize-none rounded-md border bg-background p-3 font-mono text-xs leading-relaxed focus:outline-none focus:ring-2 focus:ring-primary/40"
        />
      </div>
      <div className="flex flex-col">
        <div className="mb-2 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
          {t("skillHub.editor.preview")}
        </div>
        <div className="flex-1 overflow-y-auto rounded-md border bg-card p-4">
          {value.trim() ? (
            <MarkdownRenderer
              content={value}
              className="prose-sm text-foreground prose-headings:text-foreground prose-strong:text-foreground prose-code:text-foreground"
            />
          ) : (
            <div className="text-xs italic text-muted-foreground">
              {t("skillHub.editor.livePreviewHint")}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
