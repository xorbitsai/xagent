"use client";

import { useI18n } from "@/contexts/i18n-context";
import { Loader2, Search } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { getApiUrl, cn } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import type { Template } from "@/types/template";
import { FeaturedTemplateCard } from "@/components/templates/featured-template-card";
import { LibraryTemplateCard } from "@/components/templates/library-template-card";
import type { TranslationKey } from "@/i18n/translations";

interface CategorySection {
  id: string;
  title: string;
  templates: Template[];
}

const CATEGORY_LABEL_KEYS: Record<string, TranslationKey> = {
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
};

const normalizeCategoryKey = (category: string) =>
  category.toLowerCase().replace(/\s*&\s*/g, "_").replace(/\s+/g, "_");

const formatFallbackLabel = (category: string) =>
  category
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export default function TemplatesPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const [selectedCategory, setSelectedCategory] = useState("All");
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

  const categoryLabel = (category: string) => {
    const key = CATEGORY_LABEL_KEYS[normalizeCategoryKey(category)];
    return key ? t(key) : formatFallbackLabel(category);
  };

  const categories = useMemo(() => {
    const preferred = ["Sales", "Marketing", "Support", "Research", "Productivity"];
    const dynamic = Array.from(new Set(templates.map((template) => template.category).filter(Boolean)));
    const orderedDynamic = [
      ...preferred.filter((category) => dynamic.includes(category)),
      ...dynamic.filter((category) => !preferred.includes(category)),
    ];

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
    <div className="flex h-full flex-col overflow-y-auto p-[48px_52px_72px]">
      {/* Hero — bleeds to edges via negative margin */}
      <div className="m-[-48px_-52px_36px] flex flex-col items-center gap-[14px] border-b border-border bg-background p-[48px_64px_44px] text-center">
        <div className="flex w-full flex-col items-center gap-[14px]">
          <div>
            <div className="mb-1 text-[30px] font-extrabold tracking-[-0.04em] text-foreground">
              {t("templates.title")}
            </div>
            <div className="text-[13px] tracking-[0.01em] text-muted-foreground">
              {t("templates.subtitle")}
            </div>
          </div>

          {/* Pill search bar */}
          <div className="flex w-full max-w-[620px]">
            <div className="flex w-full items-center gap-3 rounded-full border-[1.5px] border-border bg-background px-5 py-[13px] shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
              <Search className="h-[18px] w-[18px] flex-shrink-0 text-muted-foreground" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={t("templates.searchPlaceholder")}
                className="w-full border-none bg-transparent text-sm text-foreground outline-none"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Category tabs */}
      <div className="relative mb-5 flex flex-wrap items-center justify-center gap-1.5">
        {categories.map((cat) => {
          const isActive = selectedCategory === cat.id;
          return (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={cn(
                "rounded-full px-4 py-1.5 text-xs transition-all duration-150",
                isActive
                  ? "border-none bg-[linear-gradient(135deg,rgb(48,64,207),rgb(60,131,246))] font-semibold text-white"
                  : "border border-[rgba(60,131,246,0.16)] bg-transparent font-medium text-muted-foreground"
              )}
            >
              {cat.label}
            </button>
          );
        })}
        <span className="absolute right-0 rounded-full border border-[rgba(60,131,246,0.18)] bg-[rgba(60,131,246,0.08)] px-[10px] py-[3px] text-[11px] font-semibold text-[rgb(60,131,246)]">
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
            <div>
              {/* Featured cards row */}
              <div className="mb-8 flex w-full flex-wrap gap-[10px]">
                <div className="mb-1.5 basis-full text-[10.5px] font-bold uppercase tracking-[0.08em] text-muted-foreground">
                  {t("templates.categoryTitles.featured")}
                </div>
                {featuredTemplates.map((template) => (
                  <div key={template.id} className="flex min-w-[200px] flex-1">
                    <FeaturedTemplateCard
                      template={template}
                      categoryLabel={categoryLabel(template.category)}
                      onUse={handleUseTemplate}
                      onLike={handleLikeTemplate}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Library sections */}
          {sections.map((section) => (
            <div key={section.id}>
              {/* Section title */}
              <div className="mb-[18px] flex items-center gap-3 text-[10.5px] font-bold uppercase tracking-[0.11em] text-[rgb(60,131,246)]">
                {section.title}
                <span className="h-px flex-1 bg-[linear-gradient(90deg,rgba(60,131,246,0.22)_0%,transparent_100%)]" />
              </div>

              {/* 4-column grid */}
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                {section.templates.map((template) => (
                  <LibraryTemplateCard
                    key={template.id}
                    template={template}
                    categoryLabel={categoryLabel(template.category)}
                    useLabel={t("templates.useTemplate")}
                    defaultSetupTime={t("templates.defaultSetupTime")}
                    onUse={handleUseTemplate}
                    onLike={handleLikeTemplate}
                  />
                ))}
              </div>
            </div>
          ))}

          {sections.length === 0 && (
            <div className="rounded-[14px] border border-dashed border-border bg-background p-[72px_24px] text-center text-muted-foreground">
              <p className="text-[15px]">{t("templates.noResults")}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
