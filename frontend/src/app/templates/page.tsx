"use client";

import { useI18n } from "@/contexts/i18n-context";
import { Loader2, Search, Star } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { getApiUrl } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import type { Template } from "@/types/template";
import { FeaturedTemplateCard } from "@/components/templates/featured-template-card";
import { LibraryTemplateCard } from "@/components/templates/library-template-card";
import { SegmentedTabs } from "@/components/ui/segmented-tabs";

interface CategorySection {
  id: string;
  title: string;
  templates: Template[];
}

const CATEGORY_LABEL_KEYS: Record<string, string> = {
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

const CATEGORY_ACCENTS: Record<string, { border: string; text: string; hex: string }> = {
  support: { border: "bg-[#3B5AF6]", text: "text-[#3B5AF6]", hex: "#3B5AF6" },
  sales: { border: "bg-[#15A34A]", text: "text-[#15A34A]", hex: "#15A34A" },
  marketing: { border: "bg-[#EC4899]", text: "text-[#EC4899]", hex: "#EC4899" },
  research: { border: "bg-[#7C3AED]", text: "text-[#7C3AED]", hex: "#7C3AED" },
  productivity: { border: "bg-[#F59E0B]", text: "text-[#F59E0B]", hex: "#F59E0B" },
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

  const categoryAccent = (category: string) =>
    CATEGORY_ACCENTS[normalizeCategoryKey(category)] || {
      border: "bg-[#94A3B8]",
      text: "text-[#94A3B8]",
      hex: "#94A3B8",
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
    <div className="flex h-full flex-col overflow-y-auto bg-[#F6F7FB] dark:bg-background">
      <div className="mx-auto w-full p-6">
        <div className="mb-[22px] max-w-[600px]">
          <h1 className="text-[34px] font-bold tracking-tight text-foreground">{t("templates.title")}</h1>
          <p className="mt-1 text-[13.5px] leading-[1.55] text-muted-foreground">{t("templates.subtitle")}</p>
        </div>

        <div className="mb-[22px] flex flex-wrap items-center gap-3">
          <div className="relative w-full max-w-[320px]">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-[14px] w-[14px] -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t("templates.searchPlaceholder")}
              className="h-[38px] w-full rounded-[8px] border border-[#E7EAF3] bg-white pl-9 pr-4 text-[13px] placeholder:text-muted-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-primary/10"
            />
          </div>

          <SegmentedTabs
            items={categories}
            value={selectedCategory}
            onValueChange={setSelectedCategory}
            className="shrink-0"
          />

          <div className="flex-1" />

          <div className="shrink-0 rounded-full bg-[#EEF2FF] px-[10px] py-[5px] text-[12px] font-semibold text-[#3B5AF6]">
            {filteredTemplates.length === 1
              ? t("templates.countOne", { count: filteredTemplates.length })
              : t("templates.countOther", { count: filteredTemplates.length })}
          </div>
        </div>

        {loading ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="space-y-7">
            {selectedCategory === "All" && !searchQuery && featuredTemplates.length > 0 ? (
              <section className="space-y-3">
                <div className="flex items-center gap-[6px] text-[11.5px] font-bold uppercase tracking-[0.06em] text-[#64748B]">
                  <Star className="h-[13px] w-[13px] text-yellow-500" />
                  <span>{t("templates.categoryTitles.featured")}</span>
                </div>
                <div className="grid grid-cols-1 gap-[14px] xl:grid-cols-3">
                  {featuredTemplates.map((template) => (
                    <FeaturedTemplateCard
                      key={template.id}
                      template={template}
                      categoryLabel={categoryLabel(template.category)}
                      popularLabel={t("templates.popular")}
                      runsLabel={t("templates.runs")}
                      onUse={handleUseTemplate}
                      onLike={handleLikeTemplate}
                    />
                  ))}
                </div>
              </section>
            ) : null}

            {sections.map((section) => (
              <section key={section.id} className="space-y-3">
                <div className="flex items-center gap-[6px] text-[11.5px] font-bold uppercase tracking-[0.06em] text-[#64748B]">
                  <span
                    className="h-[5px] w-[5px] rounded-full"
                    style={{ background: categoryAccent(section.templates[0]?.category || section.title).hex }}
                  />
                  <span>{section.title}</span>
                </div>
                <div className="grid grid-cols-1 gap-[14px] xl:grid-cols-3">
                  {section.templates.map((template) => (
                    <LibraryTemplateCard
                      key={template.id}
                      template={template}
                      categoryLabel={categoryLabel(template.category)}
                      useLabel={t("templates.useTemplate")}
                      defaultSetupTime={t("templates.defaultSetupTime")}
                      accentColorClassName={categoryAccent(template.category).border}
                      accentSoftClassName={categoryAccent(template.category).text}
                      accentHex={categoryAccent(template.category).hex}
                      onUse={handleUseTemplate}
                      onLike={handleLikeTemplate}
                    />
                  ))}
                </div>
              </section>
            ))}

            {sections.length === 0 && (
              <div className="rounded-2xl border border-dashed border-border bg-white px-6 py-20 text-center text-muted-foreground">
                <p className="text-[15px]">{t("templates.noResults")}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
