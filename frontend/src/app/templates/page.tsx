"use client";

import { useI18n } from "@/contexts/i18n-context";
import { Loader2 } from "lucide-react";
import { Suspense, useState, useEffect, useMemo } from "react";
import { getApiUrl } from "@/lib/utils";
import { useRouter, useSearchParams } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import { SearchInput } from "@/components/ui/search-input";
import { PageHeader } from "@/components/ui/page-header";
import { SegmentedTabs } from "@/components/ui/segmented-tabs";
import type { Template } from "@/types/template";
import { LibraryTemplateCard } from "@/components/templates/library-template-card";
import type { TranslationKey } from "@/i18n/translations";
import { normalizeCategoryKey, orderCategoriesWithPreferred } from "@/lib/template-categories";

interface CategorySection {
  id: string;
  title: string;
  templates: Template[];
}

const CATEGORY_LABEL_KEYS: Record<string, TranslationKey> = {
  general: "templates.categoryTitles.general",
  sales: "templates.categoryTitles.sales",
  marketing: "templates.categoryTitles.marketing",
  support: "templates.categoryTitles.support",
  research: "templates.sections.knowledge",
  productivity: "templates.categoryTitles.general_productivity",
  healthcare_fitness: "templates.categoryTitles.healthcare_fitness",
  general_productivity: "templates.categoryTitles.general_productivity",
  customer_service: "templates.categoryTitles.customer_service",
  finance_lms_ops: "templates.categoryTitles.finance_lms_ops",
  security: "templates.categoryTitles.security",
  operations: "templates.categoryTitles.operations",
};

const formatFallbackLabel = (category: string) =>
  category
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export default function TemplatesPage() {
  return (
    <Suspense fallback={null}>
      <TemplatesPageContent />
    </Suspense>
  );
}

function TemplatesPageContent() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const searchParams = useSearchParams();
  // Honors the category the "All templates" escape hatch on the /task
  // quick-access panel links back to (?category=<id>), so following it
  // lands on the same category instead of always resetting to "All".
  const [selectedCategory, setSelectedCategory] = useState(
    () => searchParams.get("category") || "All"
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoading(true);
        const response = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (response.ok) {
          const data = await response.json();
          setTemplates(data);
        }
      } catch (error) {
        console.error("Failed to fetch templates:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchTemplates();
  }, [locale]);

  const categoryLabel = (category?: string) => {
    const c = category || "Others";
    const key = CATEGORY_LABEL_KEYS[normalizeCategoryKey(c)];
    return key ? t(key) : formatFallbackLabel(c);
  };

  const categories = useMemo(() => {
    const preferred = ["Sales", "Marketing", "Support", "Research", "Productivity"];
    const dynamic = Array.from(new Set(templates.map((template) => template.category).filter(Boolean)));
    const orderedDynamic = orderCategoriesWithPreferred(dynamic, preferred);

    return [
      { id: "All", label: t("templates.categoryTitles.all") },
      ...orderedDynamic.map((category) => ({
        id: category,
        label: categoryLabel(category),
      })),
    ];
  }, [t, templates]);

  const featuredTemplates = useMemo(
    () => templates.filter((template) => template.featured),
    [templates]
  );

  const filteredTemplates = useMemo(
    () =>
      templates.filter((template) => {
        const matchesCategory = selectedCategory === "All" || template.category === selectedCategory;
        const matchesSearch =
          !searchQuery ||
          template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          template.description.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesSearch;
      }),
    [searchQuery, selectedCategory, templates]
  );

  const sections = useMemo(() => {
    const grouped: Record<string, Template[]> = {};
    const featuredIds = new Set(featuredTemplates.map((template) => template.id));
    const shouldHideFeaturedFromSections = selectedCategory === "All" && !searchQuery;

    filteredTemplates.forEach((template) => {
      if (shouldHideFeaturedFromSections && featuredIds.has(template.id)) {
        return;
      }
      const category = template.category || "Others";
      if (!grouped[category]) grouped[category] = [];
      grouped[category].push(template);
    });

    const orderedCategories = categories
      .map((category) => category.id)
      .filter((category) => category !== "All");

    const orderedSections: CategorySection[] = orderedCategories
      .filter((category) => grouped[category]?.length)
      .map((category) => ({
        id: normalizeCategoryKey(category),
        title: categoryLabel(category),
        templates: grouped[category],
      }));

    Object.keys(grouped).forEach((category) => {
      if (!orderedCategories.includes(category)) {
        orderedSections.push({
          id: normalizeCategoryKey(category),
          title: categoryLabel(category),
          templates: grouped[category],
        });
      }
    });

    return orderedSections;
  }, [categories, featuredTemplates, filteredTemplates, searchQuery, selectedCategory]);

  const handleUseTemplate = async (templateId: string) => {
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, { method: "POST" });
    } catch { }
    router.push(`/build/new?template=${templateId}`);
  };

  const handleLikeTemplate = async (templateId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const response = await apiRequest(`${getApiUrl()}/api/templates/${templateId}/like`, { method: "POST" });
      if (response.ok) {
        const res = await apiRequest(`${getApiUrl()}/api/templates/?lang=${locale}`);
        if (res.ok) setTemplates(await res.json());
      }
    } catch { }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <PageHeader
        title={t("templates.title")}
        description={t("templates.subtitle")}
        actions={
          <SearchInput
            placeholder={t("templates.searchPlaceholder")}
            value={searchQuery}
            onChange={setSearchQuery}
            containerClassName="flex-1 sm:w-64"
          />
        }
      />

      <div className="px-6 py-6 md:px-8">
      {/* Segmented category filter */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <SegmentedTabs
          items={categories}
          value={selectedCategory}
          onValueChange={setSelectedCategory}
          listClassName="gap-0.5 rounded-[13px] bg-muted p-1"
          triggerClassName="rounded-[10px] px-4 py-2 text-sm duration-300"
          activeTriggerClassName="bg-background font-semibold text-foreground shadow-sm"
          inactiveTriggerClassName="font-medium text-muted-foreground hover:text-foreground"
        />
        <span className="rounded-full bg-muted px-3 py-1.5 text-[13px] font-medium text-muted-foreground">
          {filteredTemplates.length === 1
            ? t("templates.countOne", { count: filteredTemplates.length })
            : t("templates.countOther", { count: filteredTemplates.length })}
        </span>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="flex flex-col gap-12">
          {/* Featured section */}
          {selectedCategory === "All" && !searchQuery && featuredTemplates.length > 0 && (
            <TemplateSection
              title={t("templates.categoryTitles.featured")}
              count={featuredTemplates.length}
              templates={featuredTemplates}
              categoryLabel={categoryLabel}
              useLabel={t("templates.useTemplate")}
              defaultSetupTime={t("templates.defaultSetupTime")}
              onUse={handleUseTemplate}
              onLike={handleLikeTemplate}
            />
          )}

          {/* Library sections */}
          {sections.map((section) => (
            <TemplateSection
              key={section.id}
              title={section.title}
              count={section.templates.length}
              templates={section.templates}
              categoryLabel={categoryLabel}
              useLabel={t("templates.useTemplate")}
              defaultSetupTime={t("templates.defaultSetupTime")}
              onUse={handleUseTemplate}
              onLike={handleLikeTemplate}
            />
          ))}

          {sections.length === 0 && (
            <div className="rounded-[14px] border border-dashed border-border bg-background p-[72px_24px] text-center text-muted-foreground">
              <p className="text-[15px]">{t("templates.noResults")}</p>
            </div>
          )}
        </div>
      )}
      </div>
    </div>
  );
}

interface TemplateSectionProps {
  title: string;
  count: number;
  templates: Template[];
  categoryLabel: (category: string) => string;
  useLabel: string;
  defaultSetupTime: string;
  onUse: (templateId: string) => void;
  onLike: (templateId: string, event: React.MouseEvent<HTMLButtonElement>) => void;
}

function TemplateSection({
  title,
  count,
  templates,
  categoryLabel,
  useLabel,
  defaultSetupTime,
  onUse,
  onLike,
}: TemplateSectionProps) {
  return (
    <section>
      <div className="mb-4 flex items-baseline gap-2.5">
        <h2 className="text-[19px] font-semibold tracking-[-0.02em] text-foreground">{title}</h2>
        <span className="text-[13px] font-medium text-muted-foreground">{count}</span>
      </div>
      <div
        className="grid gap-5"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(340px, 100%), 1fr))" }}
      >
        {templates.map((template) => (
          <LibraryTemplateCard
            key={template.id}
            template={template}
            categoryLabel={categoryLabel(template.category)}
            useLabel={useLabel}
            defaultSetupTime={defaultSetupTime}
            onUse={onUse}
            onLike={onLike}
          />
        ))}
      </div>
    </section>
  );
}
