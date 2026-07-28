import React, { useMemo } from "react";
import Link from "next/link";
import type { ComponentType, SVGProps } from "react";
import {
  Bot,
  FileText,
  Share2,
  Inbox,
  Megaphone,
  Target,
  Headphones,
  Star,
  Zap,
  Layers,
} from "lucide-react";
import { useI18n } from "@/contexts/i18n-context";
import type { TranslationKey } from "@/i18n/translations";
import { cn } from "@/lib/utils";
import type { Template, SamplePrompt } from "@/types/template";
import {
  FEATURED_CATEGORY_ID,
  getOrderedCategoriesWithCounts,
  getTemplatesForCategory,
} from "@/lib/template-categories";

type IconComponent = ComponentType<SVGProps<SVGSVGElement>>;

const CATEGORY_ICONS: Record<string, IconComponent> = {
  [FEATURED_CATEGORY_ID]: Star,
  Marketing: Megaphone,
  Sales: Target,
  Support: Headphones,
};

const TEMPLATE_ICON_BY_ID: Record<string, IconComponent> = {
  "general-doc-summarizer-action-extractor": FileText,
  "marketing-social-media-content-manager": Share2,
  "support-ai-chatbot-agent": Bot,
  "support-inbox-manager": Inbox,
};

// A dedicated short label for the Featured pill/heading here - the shared
// `templates.categoryTitles.featured` string ("Featured Templates") is sized
// for the /templates library page's section heading, not this compact pill.
const CATEGORY_LABEL_KEYS: Record<string, TranslationKey> = {
  featured: "chatPage.templateQuickAccess.featuredLabel",
  marketing: "templates.categoryTitles.marketing",
  sales: "templates.categoryTitles.sales",
  support: "templates.categoryTitles.support",
};

// Shared indigo accent used throughout this component for the active/hover/
// selected states (category tab, count badge, "All templates" link, card
// hover border, card title hover, selected sample-prompt row).
const ACCENT_TEXT_CLASS = "text-[#3040cf]";
const ACCENT_ACTIVE_PILL_CLASSES = "border-[#3040cf] bg-[#eef1ff] text-[#3040cf]";
const ACCENT_ACTIVE_BADGE_CLASSES = "bg-[#3040cf]/10 text-[#3040cf]";
const ACCENT_SELECTED_PROMPT_CLASSES = "bg-[#eef1ff] font-medium text-[#3040cf]";
const ACCENT_HOVER_BORDER_CLASS = "hover:border-[#3040cf]";
const ACCENT_HOVER_TITLE_CLASS = "group-hover/text:text-[#3040cf]";

function getTemplateIcon(template: Template): IconComponent {
  return (
    TEMPLATE_ICON_BY_ID[template.id] ||
    CATEGORY_ICONS[template.category] ||
    Bot
  );
}

export interface TemplateQuickAccessProps {
  templates: Template[];
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
  selectedPromptKey: string | null;
  onPromptSelect: (template: Template, prompt: SamplePrompt, index: number) => void;
}

export function TemplateQuickAccess({
  templates,
  selectedCategory,
  onCategoryChange,
  selectedPromptKey,
  onPromptSelect,
}: TemplateQuickAccessProps) {
  const { t } = useI18n();

  const categoryTabs = useMemo(
    () => getOrderedCategoriesWithCounts(templates),
    [templates]
  );

  // The parent may hand us a selectedCategory (e.g. the "Featured" default)
  // that no longer has a matching tab - a deployment with no featured
  // templates, for instance. Fall back to the first available tab instead of
  // rendering an empty, unlabeled panel.
  const activeCategory = categoryTabs.some((tab) => tab.id === selectedCategory)
    ? selectedCategory
    : categoryTabs[0]?.id ?? selectedCategory;

  const cards = useMemo(
    () => getTemplatesForCategory(templates, activeCategory, 4),
    [templates, activeCategory]
  );

  const categoryLabel = (categoryId: string) => {
    const key = CATEGORY_LABEL_KEYS[categoryId.toLowerCase()];
    return key ? t(key) : categoryId;
  };

  if (categoryTabs.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-center gap-2">
        {categoryTabs.map((tab) => {
          const Icon = CATEGORY_ICONS[tab.id] || Layers;
          const isActive = tab.id === activeCategory;
          return (
            <button
              key={tab.id}
              type="button"
              aria-pressed={isActive}
              onClick={() => onCategoryChange(tab.id)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] font-medium transition-colors",
                isActive
                  ? ACCENT_ACTIVE_PILL_CLASSES
                  : "border-border text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{categoryLabel(tab.id)}</span>
              <span
                className={cn(
                  "rounded-full px-1.5 py-0.5 text-[11px] font-semibold",
                  isActive ? ACCENT_ACTIVE_BADGE_CLASSES : "bg-muted text-muted-foreground"
                )}
              >
                {tab.count}
              </span>
            </button>
          );
        })}
      </div>

      {cards.length > 0 && (
        <>
          <div className="flex items-baseline justify-between px-1">
            <h3 className="text-[14px] font-semibold text-foreground">
              {t("chatPage.templateQuickAccess.categoryHeading", {
                category: categoryLabel(activeCategory),
              })}
            </h3>
            <Link
              href="/templates"
              className={cn("text-[13px] font-medium hover:underline", ACCENT_TEXT_CLASS)}
            >
              {t("chatPage.templateQuickAccess.allTemplates")}
            </Link>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {cards.map((template) => {
              const Icon = getTemplateIcon(template);
              const prompts = (template.sample_prompts || []).slice(0, 2);

              return (
                <div
                  key={template.id}
                  className={cn(
                    "space-y-3 rounded-xl border border-border bg-card p-4 text-left transition-colors",
                    ACCENT_HOVER_BORDER_CLASS
                  )}
                >
                  <Link
                    href={`/build/new?template=${encodeURIComponent(template.id)}`}
                    className="group/text flex items-start gap-3"
                  >
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-300">
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <h4
                        className={cn(
                          "truncate text-[14px] font-semibold text-foreground transition-colors",
                          ACCENT_HOVER_TITLE_CLASS
                        )}
                      >
                        {template.name}
                      </h4>
                      <p className="line-clamp-2 text-[12.5px] text-muted-foreground">
                        {template.description}
                      </p>
                    </div>
                  </Link>

                  {prompts.length > 0 && (
                    <div className="space-y-1">
                      {prompts.map((prompt, index) => {
                        const key = `${template.id}:${index}`;
                        const isSelected = key === selectedPromptKey;
                        return (
                          <button
                            key={key}
                            type="button"
                            title={prompt.title}
                            onClick={() => onPromptSelect(template, prompt, index)}
                            className={cn(
                              "flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors",
                              isSelected
                                ? ACCENT_SELECTED_PROMPT_CLASSES
                                : "text-foreground/80 hover:bg-muted/60"
                            )}
                          >
                            <Zap className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">{prompt.title}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
