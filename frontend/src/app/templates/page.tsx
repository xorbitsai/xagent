"use client";

import { useI18n } from "@/contexts/i18n-context";
import {
  Play,
  Heart,
  Loader2,
  Clock,
  ChevronRight
} from "lucide-react";
import Image from "next/image";
import { useState, useEffect } from "react";
import { cn, getApiUrl } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { apiRequest } from "@/lib/api-wrapper";
import { SearchInput } from "@/components/ui/search-input";
import type { Template } from "@/types/template";

// Category section types
interface CategorySection {
  id: string;
  title: string;
  templates: Template[];
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
    { id: "Featured", label: t("templates.categoryTitles.featured") },
    { id: "Sales", label: t("templates.categoryTitles.sales") },
    { id: "Marketing", label: t("templates.categoryTitles.marketing") },
    { id: "Support", label: t("templates.categoryTitles.support") },
  ];

  // Category display configuration
  const categoryConfig: Record<string, { title: string }> = {
    Featured: {
      title: t("templates.categoryTitles.featured"),
    },
    Sales: {
      title: t("templates.categoryTitles.sales"),
    },
    Marketing: {
      title: t("templates.categoryTitles.marketing"),
    },
    Support: {
      title: t("templates.categoryTitles.support"),
    },
  };

  // Fetch templates from API
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoading(true);
        const response = await apiRequest(
          `${getApiUrl()}/api/templates/?lang=${locale}`
        );
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

  // Group templates by category
  const groupTemplatesByCategory = (
    templatesList: Template[],
    includeFeaturedSection: boolean,
    includeCategorySections: boolean,
  ): CategorySection[] => {
    const grouped: Record<string, Template[]> = {};

    templatesList.forEach((template) => {
      if (includeFeaturedSection && template.featured) {
        if (!grouped.Featured) {
          grouped.Featured = [];
        }
        grouped.Featured.push(template);
      }
      if (!includeCategorySections) {
        return;
      }
      if (!grouped[template.category]) {
        grouped[template.category] = [];
      }
      grouped[template.category].push(template);
    });

    return Object.entries(grouped)
      .map(([category, templates]) => ({
        id: category.toLowerCase().replace(/\s+/g, "-"),
        title: categoryConfig[category]?.title || category,
        templates,
      }))
      .sort((a, b) => {
        if (a.title === t("templates.categoryTitles.featured")) {
          return -1;
        }
        if (b.title === t("templates.categoryTitles.featured")) {
          return 1;
        }
        return a.title.localeCompare(b.title);
      });
  };

  // Filter and group templates
  const filteredTemplates = templates.filter((template) => {
    const matchesCategory =
      selectedCategory === "All" ||
      (selectedCategory === "Featured" && Boolean(template.featured)) ||
      template.category === selectedCategory;
    const matchesSearch =
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const filteredSections = groupTemplatesByCategory(
    filteredTemplates,
    true,
    selectedCategory !== "Featured",
  );

  // Handle use template
  const handleUseTemplate = async (templateId: string) => {
    // Record usage
    try {
      await apiRequest(`${getApiUrl()}/api/templates/${templateId}/use`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to record template usage:", error);
    }

    // Navigate to build/new page with template parameter
    router.push(`/build/new?template=${templateId}`);
  };

  // Handle like template
  const handleLikeTemplate = async (
    templateId: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/templates/${templateId}/like`,
        { method: "POST" },
      );
      if (response.ok) {
        // Refresh templates to get updated stats
        const templatesResponse = await apiRequest(
          `${getApiUrl()}/api/templates/?lang=${locale}`,
        );
        if (templatesResponse.ok) {
          const data = await templatesResponse.json();
          setTemplates(data);
        }
      }
    } catch (error) {
      console.error("Failed to like template:", error);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background/50">
      {/* Header */}
      <div className="w-full px-8 pt-8 pb-4">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h1 className="text-3xl font-bold mb-1 text-foreground">
              {t("templates.title")}
            </h1>
            <p className="text-muted-foreground">{t("templates.subtitle")}</p>
          </div>
          <SearchInput
            placeholder={t("templates.searchPlaceholder")}
            value={searchQuery}
            onChange={setSearchQuery}
            containerClassName="w-80"
            className="bg-transparent border-border rounded-lg focus:bg-background transition-all"
          />
        </div>
        <hr className="border-border/60" />
      </div>

      {/* Category Filter */}
      <div className="w-full px-8 pb-6 flex justify-between items-center gap-2 overflow-x-auto scrollbar-hide">
        <div className="flex gap-2">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={cn(
                "px-4 py-1.5 rounded-full text-sm font-medium transition-all whitespace-nowrap border",
                selectedCategory === category.id
                  ? "bg-primary text-primary-foreground border-primary shadow-md"
                  : "bg-transparent text-muted-foreground border-border hover:bg-secondary hover:text-foreground",
              )}
            >
              {category.label}
            </button>
          ))}
        </div>
        <div className="px-4 py-1.5 rounded-full text-sm font-medium text-primary bg-primary/10 border border-primary/20 whitespace-nowrap flex-shrink-0">
          {filteredTemplates.length} {filteredTemplates.length === 1 ? "template" : "templates"}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="w-full px-8 pb-8 space-y-10">
            {filteredSections.map((section) => (
              <div key={section.id} className="animate-fade-in">
                {/* Section Header */}
                <div className="flex items-center gap-4 mb-6">
                  <h2 className="text-primary font-bold text-sm tracking-widest uppercase whitespace-nowrap">
                    {section.title}
                  </h2>
                  <div className="h-[1px] flex-grow bg-border/60" />
                </div>

                {/* Templates Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {section.templates.map((template) => {
                    return (
                      <div
                        key={template.id}
                        className="flex flex-col bg-card rounded-2xl border border-border/60 shadow-sm hover:shadow-md transition-shadow p-6 group"
                      >
                        {/* Card Header: Category & Setup Time */}
                        <div className="flex justify-between items-center mb-4">
                          <span className="text-xs font-bold text-primary tracking-wide uppercase">
                            {categories.find((cat) => cat.id === template.category)?.label || ""}
                          </span>
                          <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
                            <Clock className="w-3.5 h-3.5" />
                            <span>{template.setup_time || "5 min setup"}</span>
                          </div>
                        </div>

                        {/* Title */}
                        <h3 className="font-bold text-xl mb-4 text-foreground group-hover:text-primary transition-colors line-clamp-1">
                          {template.name}
                        </h3>

                        {/* Description/Features */}
                        <div className="flex-1 space-y-2 mb-6">
                          {(template.features && template.features.length > 0) ? (
                            template.features.map((feature, idx) => (
                              <div key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                                <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                                <span className="line-clamp-2">{feature}</span>
                              </div>
                            ))
                          ) : (
                            <div className="flex items-start gap-2 text-sm text-muted-foreground">
                              <ChevronRight className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                              <span className="line-clamp-4">{template.description}</span>
                            </div>
                          )}
                        </div>

                        {/* Divider */}
                        <div className="h-[1px] bg-border/60 mb-4" />

                        {/* Footer: Connections, Stats & Action */}
                        <div className="mt-auto">
                          <div className="flex items-center justify-between text-sm text-muted-foreground mb-4">
                            <div className="flex items-center">
                              {template.connections && template.connections.length > 0 ? (
                                <div className="flex gap-1.5">
                                  {template.connections.slice(0, 4).map((conn, idx) => (
                                    <div key={idx} className="w-8 h-8 rounded-lg bg-background border border-border flex items-center justify-center overflow-hidden transition-transform">
                                      {conn.logo ? <img
                                        src={conn.logo}
                                        alt={conn.name}
                                        width={20}
                                        height={20}
                                        className="object-contain"
                                        onError={(e) => {
                                          // Fallback to initial letters if icon is not found
                                          e.currentTarget.style.display = 'none';
                                          e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                        }}
                                      /> :
                                        <span className="text-[10px] font-medium hidden text-primary/70">{conn.name.substring(0, 2).toUpperCase()}</span>
                                      }
                                    </div>
                                  ))}
                                  {template.connections.length > 4 && (
                                    <div className="w-8 h-8 rounded-lg bg-muted border border-border flex items-center justify-center text-muted-foreground">
                                      <span className="text-[10px] font-medium">+{template.connections.length - 4}</span>
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <div className="h-8" />
                              )}
                            </div>

                            <div className="flex items-center gap-3">
                              <div className="flex items-center gap-1.5">
                                <Play className="w-3.5 h-3.5 fill-current text-primary/70" />
                                <span className="font-semibold text-foreground/80">{template.used_count}</span>
                              </div>
                              <button
                                onClick={(e) => handleLikeTemplate(template.id, e)}
                                className="flex items-center gap-1.5 hover:text-pink-500 transition-colors"
                              >
                                <Heart className="w-3.5 h-3.5 fill-current text-blue-400/70" />
                                <span className="font-semibold text-foreground/80">{template.likes}</span>
                              </button>
                            </div>
                          </div>

                          <button
                            onClick={() => handleUseTemplate(template.id)}
                            className="w-full py-2.5 text-primary text-sm font-bold uppercase tracking-wide rounded-lg border border-primary/20 hover:bg-primary/5 transition-colors"
                          >
                            {t("templates.useTemplate")}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {filteredSections.length === 0 && !loading && (
              <div className="text-center py-20 text-muted-foreground">
                <p>{t("templates.noResults")}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
