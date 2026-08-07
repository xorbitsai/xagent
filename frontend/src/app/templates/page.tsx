"use client";

import { useI18n } from "@/contexts/i18n-context";
import { Loader2 } from "lucide-react";
import React, { Suspense, useState, useEffect, useMemo, useRef } from "react";
import { getApiUrl } from "@/lib/utils";
import { useRouter, useSearchParams } from "next/navigation";
import { apiRequest, isJsonRecord, parseApiResponse } from "@/lib/api-wrapper";
import { toast } from "sonner";
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

// The workforce-creation error paths return a structured
// `detail: {code, message, params}` (see the WORKFORCE_*_CODE constants in
// src/xagent/web/services/workforce_creator.py), mapped here by machine
// code rather than by byte-matching human-readable English strings - a
// backend wording change used to silently degrade every mapped error to
// the generic toast. Any error without a mapped code (including
// plain-string details from other paths) falls back to the generic
// translated message rather than leaking raw English.
const WORKFORCE_USE_ERROR_CODE_KEYS: Record<string, TranslationKey> = {
  workforce_create_access_denied: "templates.errors.useWorkforceAccessDenied",
  workforce_create_conflict: "templates.errors.useWorkforceRetry",
};

// Unlike the codes above, this error can never be resolved by retrying, so
// it renders its own message interpolating the agent's name from
// detail.params rather than a static translation.
const WORKFORCE_WORKER_UNPUBLISHED_CODE = "workforce_worker_unpublished";

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
  // Same one-way seeding as selectedCategory above, for a ?type=<id> link.
  // Neither is written back to the URL on change today, so - like
  // selectedCategory - the current filter combination still isn't
  // shareable via the address bar once the user changes it in-page.
  const [selectedType, setSelectedType] = useState(
    () => searchParams.get("type") || "All"
  );
  const [creatingWorkforceId, setCreatingWorkforceId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  // If the user navigates away while a workforce creation request is still
  // in flight, the request still resolves - without this guard, its
  // .then()/finally would call setState and router.push on an unmounted
  // page, yanking the user back to the just-created workforce's canvas
  // with no way to know why.
  const isMountedRef = useRef(true);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

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

  const formatAgentsCount = (count: number) =>
    count === 1
      ? t("templates.agentsCountOne", { count })
      : t("templates.agentsCountOther", { count });

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

  const typeTabs = useMemo(
    () => [
      { id: "All", label: t("templates.typeFilter.all") },
      { id: "agent", label: t("templates.typeFilter.agent") },
      { id: "workforce", label: t("templates.typeFilter.workforce") },
    ],
    [t]
  );

  const featuredTemplates = useMemo(
    () => templates.filter((template) => template.featured),
    [templates]
  );

  const filteredTemplates = useMemo(
    () =>
      templates.filter((template) => {
        const matchesCategory = selectedCategory === "All" || template.category === selectedCategory;
        const matchesType =
          selectedType === "All" || (template.type || "agent") === selectedType;
        const matchesSearch =
          !searchQuery ||
          template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          template.description.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesType && matchesSearch;
      }),
    [searchQuery, selectedCategory, selectedType, templates]
  );

  const sections = useMemo(() => {
    const grouped: Record<string, Template[]> = {};
    const featuredIds = new Set(featuredTemplates.map((template) => template.id));
    const shouldHideFeaturedFromSections =
      selectedCategory === "All" && selectedType === "All" && !searchQuery;

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
  }, [categories, featuredTemplates, filteredTemplates, searchQuery, selectedCategory, selectedType]);

  const handleUseTemplate = async (templateId: string) => {
    // A workforce creation in flight locks EVERY card, not just other
    // workforce cards - clicking a plain Agent card mid-creation used to
    // still work, landing the user on /build/new only to be redirected back
    // to the just-created workforce's canvas once that request resolved.
    if (creatingWorkforceId) return;

    const template = templates.find((item) => item.id === templateId);
    if (template?.type === "workforce") {
      setCreatingWorkforceId(templateId);
      try {
        const response = await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use-as-workforce?lang=${locale}`, {
          method: "POST",
        });
        // The user may have navigated away while this request was in
        // flight; the workforce still got created (or not) server-side
        // either way, but this page must not act on that anymore.
        if (!isMountedRef.current) return;
        if (response.ok) {
          const data = await response.json();
          if (!isMountedRef.current) return;
          router.push(`/workforces/${data.workforce_id}?view=canvas`);
          return;
        }
        const parsed = await parseApiResponse(response);
        const detail = isJsonRecord(parsed.data) ? parsed.data.detail : null;
        const code =
          isJsonRecord(detail) && typeof detail.code === "string" ? detail.code : null;
        if (code === WORKFORCE_WORKER_UNPUBLISHED_CODE) {
          const params = isJsonRecord(detail) ? detail.params : null;
          const agentName =
            isJsonRecord(params) && typeof params.agent_name === "string"
              ? params.agent_name
              : "";
          toast.error(t("templates.errors.useWorkforceUnpublishedAgent", { agentName }));
        } else {
          const messageKey = code ? WORKFORCE_USE_ERROR_CODE_KEYS[code] : undefined;
          toast.error(messageKey ? t(messageKey) : t("templates.errors.useWorkforceFailed"));
        }
      } catch {
        if (isMountedRef.current) toast.error(t("templates.errors.useWorkforceFailed"));
      } finally {
        if (isMountedRef.current) setCreatingWorkforceId(null);
      }
      return;
    }
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, { method: "POST" });
    } catch { }
    if (!isMountedRef.current) return;
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
      {/* Segmented category + type filters */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <SegmentedTabs
            items={categories}
            value={selectedCategory}
            onValueChange={setSelectedCategory}
            listClassName="gap-0.5 rounded-[13px] bg-muted p-1"
            triggerClassName="rounded-[10px] px-4 py-2 text-sm duration-300"
            activeTriggerClassName="bg-background font-semibold text-foreground shadow-sm"
            inactiveTriggerClassName="font-medium text-muted-foreground hover:text-foreground"
          />
          <SegmentedTabs
            items={typeTabs}
            value={selectedType}
            onValueChange={setSelectedType}
            listClassName="gap-0.5 rounded-[13px] bg-muted p-1"
            triggerClassName="rounded-[10px] px-4 py-2 text-sm duration-300"
            activeTriggerClassName="bg-background font-semibold text-foreground shadow-sm"
            inactiveTriggerClassName="font-medium text-muted-foreground hover:text-foreground"
          />
        </div>
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
          {selectedCategory === "All" && selectedType === "All" && !searchQuery && featuredTemplates.length > 0 && (
            <TemplateSection
              title={t("templates.categoryTitles.featured")}
              count={featuredTemplates.length}
              templates={featuredTemplates}
              categoryLabel={categoryLabel}
              useLabel={t("templates.useTemplate")}
              defaultSetupTime={t("templates.defaultSetupTime")}
              workforceBadgeLabel={t("templates.workforceBadge")}
              formatAgentsCount={formatAgentsCount}
              creatingWorkforceId={creatingWorkforceId}
              busyLabel={t("templates.creatingWorkforce")}
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
              workforceBadgeLabel={t("templates.workforceBadge")}
              formatAgentsCount={formatAgentsCount}
              creatingWorkforceId={creatingWorkforceId}
              busyLabel={t("templates.creatingWorkforce")}
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
  workforceBadgeLabel: string;
  formatAgentsCount: (count: number) => string;
  creatingWorkforceId: string | null;
  busyLabel: string;
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
  workforceBadgeLabel,
  formatAgentsCount,
  creatingWorkforceId,
  busyLabel,
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
            workforceBadgeLabel={workforceBadgeLabel}
            formatAgentsCount={formatAgentsCount}
            isBusy={creatingWorkforceId === template.id}
            busyLabel={busyLabel}
            // Locks every OTHER card - agent or workforce - while a
            // workforce is being created, matching the same-scoped lock
            // handleUseTemplate enforces.
            disabled={
              creatingWorkforceId !== null && creatingWorkforceId !== template.id
            }
            onUse={onUse}
            onLike={onLike}
          />
        ))}
      </div>
    </section>
  );
}
