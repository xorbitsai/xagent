"use client";

import { useI18n } from "@/contexts/i18n-context";
import { Play, Heart, Loader2, Clock, ChevronRight, Search } from "lucide-react";
import { useState, useEffect } from "react";
import { cn, getApiUrl } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import type { Template } from "@/types/template";

interface CategorySection {
  id: string;
  title: string;
  templates: Template[];
  isFeatured?: boolean;
}

export default function TemplatesPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);

  const categories = [
    { id: "All", label: t("templates.categoryTitles.all") },
    { id: "Sales", label: t("templates.categoryTitles.sales") },
    { id: "Marketing", label: t("templates.categoryTitles.marketing") },
    { id: "Support", label: t("templates.categoryTitles.support") },
  ];

  const categoryConfig: Record<string, string> = {
    Featured: t("templates.categoryTitles.featured"),
    Sales: t("templates.categoryTitles.sales"),
    Marketing: t("templates.categoryTitles.marketing"),
    Support: t("templates.categoryTitles.support"),
  };

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

  const filteredTemplates = templates.filter((template) => {
    const matchesCategory =
      selectedCategory === "All" ||
      template.category === selectedCategory;
    const matchesSearch =
      !searchQuery ||
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const buildSections = (): CategorySection[] => {
    const sections: CategorySection[] = [];

    const featuredTemplates = filteredTemplates.filter((t) => t.featured);
    if (featuredTemplates.length > 0 && selectedCategory === "All" && !searchQuery) {
      sections.push({ id: "featured", title: t("templates.categoryTitles.featured") || "Featured", templates: featuredTemplates, isFeatured: true });
    }

    const grouped: Record<string, Template[]> = {};
    filteredTemplates.forEach((template) => {
      const cat = template.category || "Others";
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(template);
    });

    const orderedCats = ["Sales", "Marketing", "Support", "IT", "Others"];
    orderedCats.forEach((cat) => {
      if (grouped[cat]?.length) {
        sections.push({
          id: cat.toLowerCase(),
          title: categoryConfig[cat] || cat,
          templates: grouped[cat],
          isFeatured: false,
        });
      }
    });

    Object.keys(grouped).forEach((cat) => {
      if (!orderedCats.includes(cat) && grouped[cat]?.length) {
        sections.push({ id: cat.toLowerCase(), title: categoryConfig[cat] || cat, templates: grouped[cat], isFeatured: false });
      }
    });

    return sections;
  };

  const sections = buildSections();

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
    <div className="flex flex-col h-full bg-background overflow-y-auto">
      {/* Top Hero */}
      <div className="w-full bg-background border-b border-border/60 pt-10 pb-8">
        <div className="max-w-4xl mx-auto px-6 flex flex-col items-center text-center">
          <h1 className="text-3xl font-bold mb-2 text-foreground tracking-tight">
            {t("templates.title")}
          </h1>
          <p className="text-[15px] text-muted-foreground mb-6">
            {t("templates.subtitle")}
          </p>
          <div className="w-full max-w-xl relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t("templates.searchPlaceholder")}
              className="w-full pl-11 pr-4 py-2.5 rounded-full border border-border bg-card text-[14px] placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/10 transition-all"
            />
          </div>
        </div>
      </div>

      {/* Category Tabs + Count */}
      <div className="sticky top-0 mt-6 z-10 bg-background/95 backdrop-blur border-b border-border/40">
        <div className="relative mx-auto px-4 sm:px-16 py-3">
          <div className="-mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
            <div className="flex min-w-max items-center justify-center gap-1.5 sm:min-w-0 sm:flex-wrap">
            {categories.map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={cn(
                  "px-4 py-1.5 rounded-full text-[13px] font-medium transition-all whitespace-nowrap",
                  selectedCategory === cat.id
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {cat.label}
              </button>
            ))}
            </div>
          </div>
          <div className="mt-3 flex justify-center sm:hidden">
            <span className="text-[13px] font-medium text-primary bg-primary/10 border border-primary/20 px-3 py-1 rounded-full whitespace-nowrap shrink-0">
              {filteredTemplates.length === 1
                ? t("templates.countOne", { count: filteredTemplates.length })
                : t("templates.countOther", { count: filteredTemplates.length })}
            </span>
          </div>
          <span className="hidden sm:inline-flex absolute right-16 top-1/2 -translate-y-1/2 text-[13px] font-medium text-primary bg-primary/10 border border-primary/20 px-3 py-1 rounded-full whitespace-nowrap shrink-0">
            {filteredTemplates.length === 1
              ? t("templates.countOne", { count: filteredTemplates.length })
              : t("templates.countOther", { count: filteredTemplates.length })}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="mx-auto px-16 py-8 space-y-12">
            {sections.map((section) => (
              <div key={section.id}>
                <h2 className="text-[11px] font-bold tracking-widest text-muted-foreground uppercase mb-5">
                  {section.title}
                </h2>

                {section.isFeatured ? (
                  /* Featured: horizontal wide cards, 3-col */
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                    {section.templates.slice(0, 3).map((template) => (
                      <div
                        key={template.id}
                        className="flex flex-col bg-slate-50 dark:bg-slate-900/40 rounded-xl border border-border/50 shadow-sm hover:shadow-md transition-all cursor-pointer group overflow-hidden"
                        onClick={() => handleUseTemplate(template.id)}
                      >
                        <div
                          className="h-[3px] w-full shrink-0"
                          style={{ background: "linear-gradient(to right, #4338ca, #7c3aed, #9333ea, #c026d3)" }}
                        />
                        <div className="flex flex-col flex-1 p-5">
                          <div className="mb-3">
                            <span className="text-[10px] font-bold tracking-widest text-primary uppercase">
                              {categoryConfig[template.category] || template.category}
                            </span>
                          </div>
                          <h3 className="font-bold text-[17px] mb-2 text-foreground group-hover:text-primary transition-colors">
                            {template.name}
                          </h3>
                          <p className="text-[13px] text-muted-foreground leading-relaxed flex-1 mb-4 line-clamp-2">
                            {template.description}
                          </p>
                          <div className="flex items-center gap-4 text-[12px] text-muted-foreground mt-auto pt-3 border-t border-border/50">
                            <div className="flex items-center gap-1">
                              <Play className="w-3 h-3 fill-current text-primary/60" />
                              <span className="font-semibold text-foreground/70">
                                {template.used_count ?? 0} {t("templates.runs")}
                              </span>
                            </div>
                            <button
                              onClick={(e) => handleLikeTemplate(template.id, e)}
                              className="flex items-center gap-1 hover:text-pink-500 transition-colors"
                            >
                              <Heart className="w-3 h-3 fill-current text-rose-400/70" />
                              <span className="font-semibold text-foreground/70">{template.likes ?? 0}</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  /* Category: standard cards, 4-col */
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-5">
                    {section.templates.map((template) => (
                      <div
                        key={template.id}
                        className="flex flex-col bg-blue-50/60 dark:bg-blue-950/20 rounded-xl border border-blue-200/60 dark:border-blue-800/40 border-t-4 border-t-blue-500 shadow-sm hover:shadow-md transition-all p-5 group"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-[10px] font-bold tracking-widest text-blue-600 dark:text-blue-400 uppercase">
                            {categoryConfig[template.category] || template.category}
                          </span>
                          <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
                            <Clock className="w-3 h-3" />
                            <span>{template.setup_time || t("templates.defaultSetupTime")}</span>
                          </div>
                        </div>

                        <h3 className="font-bold text-[15px] mb-3 text-foreground group-hover:text-primary transition-colors line-clamp-1">
                          {template.name}
                        </h3>

                        <div className="flex-1 space-y-1.5 mb-5">
                          {(template.features && template.features.length > 0)
                            ? template.features.slice(0, 3).map((feature, idx) => (
                              <div key={idx} className="flex items-start gap-1.5 text-[12px] text-muted-foreground">
                                <ChevronRight className="w-3.5 h-3.5 text-primary shrink-0 mt-px" />
                                <span className="line-clamp-2 leading-snug">{feature}</span>
                              </div>
                            ))
                            : (
                              <div className="flex items-start gap-1.5 text-[12px] text-muted-foreground">
                                <ChevronRight className="w-3.5 h-3.5 text-primary shrink-0 mt-px" />
                                <span className="line-clamp-3 leading-snug">{template.description}</span>
                              </div>
                            )
                          }
                        </div>

                        <div className="h-px bg-blue-200/60 dark:bg-blue-800/30 mb-4" />

                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-1">
                            {template.connections?.slice(0, 4).map((conn, idx) => (
                              <div key={idx} className="w-7 h-7 rounded-lg bg-white dark:bg-slate-900 border border-blue-200/60 dark:border-blue-800/40 flex items-center justify-center overflow-hidden">
                                {conn.logo
                                  ? <img src={conn.logo} alt={conn.name} className="w-4 h-4 object-contain" />
                                  : <span className="text-[9px] font-bold text-primary/70">{conn.name.substring(0, 2).toUpperCase()}</span>
                                }
                              </div>
                            ))}
                            {(template.connections?.length ?? 0) > 4 && (
                              <div className="w-7 h-7 rounded-lg bg-muted border border-border flex items-center justify-center">
                                <span className="text-[9px] font-medium text-muted-foreground">+{(template.connections?.length ?? 0) - 4}</span>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center gap-3 text-[12px] text-muted-foreground">
                            <div className="flex items-center gap-1">
                              <Play className="w-3 h-3 fill-current text-primary/60" />
                              <span className="font-semibold">{template.used_count ?? 0}</span>
                            </div>
                            <button
                              onClick={(e) => handleLikeTemplate(template.id, e)}
                              className="flex items-center gap-1 hover:text-pink-500 transition-colors"
                            >
                              <Heart className="w-3 h-3 fill-current text-rose-400/70" />
                              <span className="font-semibold">{template.likes ?? 0}</span>
                            </button>
                          </div>
                        </div>

                        <button
                          onClick={() => handleUseTemplate(template.id)}
                          className="w-full py-2 text-blue-600 dark:text-blue-400 text-[12px] font-bold uppercase tracking-widest rounded-lg border border-blue-300/60 dark:border-blue-700/50 bg-white/60 dark:bg-blue-950/30 hover:bg-blue-500 hover:text-white hover:border-blue-500 transition-all"
                        >
                          {t("templates.useTemplate")}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {sections.length === 0 && (
              <div className="text-center py-24 text-muted-foreground">
                <p className="text-[15px]">{t("templates.noResults")}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
