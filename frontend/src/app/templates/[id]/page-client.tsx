"use client";

import React, { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import { hireAgentFromTemplate } from "@/lib/hire-agent";
import { PersonaAvatar } from "@/components/templates/persona-avatar";
import type { TemplateDetail } from "@/types/template";

const TOOL_CATEGORY_KEYS: Record<string, string> = {
  basic: "builds.configForm.tools.categories.basic",
  web_search: "builds.configForm.tools.categories.webSearch",
  file: "builds.configForm.tools.categories.file",
  vision: "builds.configForm.tools.categories.vision",
  image: "builds.configForm.tools.categories.image",
  video: "builds.configForm.tools.categories.video",
  audio: "builds.configForm.tools.categories.audio",
  knowledge: "builds.configForm.tools.categories.knowledge",
  browser: "builds.configForm.tools.categories.browser",
  ppt: "builds.configForm.tools.categories.ppt",
  office: "builds.configForm.tools.categories.office",
  database: "builds.configForm.tools.categories.database",
  skill: "builds.configForm.tools.categories.skill",
  ssh: "builds.configForm.tools.categories.ssh",
};

function capitalize(value: string): string {
  return value.length > 0 ? value[0].toUpperCase() + value.slice(1) : value;
}

type LoadStatus = "loading" | "error" | "ready";

export default function TemplateDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { t, tDynamic, locale } = useI18n();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;

  const [template, setTemplate] = useState<TemplateDetail | null>(null);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [hiring, setHiring] = useState(false);
  // Guards handlePrimaryAction against a second invocation (double-click,
  // a repeated Enter keypress) firing before React commits `hiring: true` -
  // a ref updates synchronously, a state setter does not.
  const hiringRef = useRef(false);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setStatus("loading");

    (async () => {
      try {
        const response = await apiRequest(
          `${getApiUrl()}/api/templates/${id}?lang=${locale}`
        );
        if (cancelled) return;
        if (!response.ok) {
          throw new Error(`Failed to load template (${response.status})`);
        }
        const data = (await response.json()) as TemplateDetail;
        if (cancelled) return;
        setTemplate(data);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [id, locale]);

  // A workforce-type template (or any agent-type template with no
  // marketplace persona) has no meaningful detail view here - the
  // marketplace list never links to this page for one, so reaching this
  // state means a stale/direct URL. Checking `type` explicitly (not just
  // persona-is-null) matches the same guard library-template-card.tsx uses
  // before it will touch persona at all.
  const persona = template && template.type !== "workforce" ? template.persona : null;
  const isReady = status === "ready" && template !== null && persona !== null;

  const getToolLabel = (category: string): string => {
    const key = TOOL_CATEGORY_KEYS[category];
    return key ? tDynamic(key, capitalize(category)) : capitalize(category);
  };

  const handleBack = () => router.push("/templates");

  const handlePrimaryAction = async () => {
    if (!template || !persona || hiringRef.current) return;

    if (template.hired && template.hired_agent_id) {
      router.push(`/agent/${template.hired_agent_id}`);
      return;
    }

    hiringRef.current = true;
    setHiring(true);
    try {
      const result = await hireAgentFromTemplate(template.id, persona, {
        beforeWeStart: t("templates.marketplace.beforeWeStart"),
        closingNote: t("templates.marketplace.hireClosingNote"),
      });
      if (!isMountedRef.current) return;
      router.push(`/task/${result.taskId}`);
    } catch {
      if (!isMountedRef.current) return;
      toast.error(t("templates.marketplace.hireFailed", { name: persona.name }));
      hiringRef.current = false;
      setHiring(false);
    }
  };

  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!isReady || !template || !persona) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
        <p className="text-muted-foreground">{t("templates.marketplace.notFound")}</p>
        <Button variant="outline" onClick={handleBack}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          {t("templates.marketplace.back")}
        </Button>
      </div>
    );
  }

  const toolCategories = (template.agent_config?.tool_categories || []).filter(
    (category) => !category.startsWith("mcp:")
  );
  const connectedApps = template.connections || [];
  const skills = template.agent_config?.skills || [];
  const samplePrompts = template.sample_prompts || [];

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <div className="mx-auto w-full max-w-5xl px-6 py-6 md:px-8">
        <button
          type="button"
          onClick={handleBack}
          className="mb-6 flex items-center gap-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("templates.marketplace.back")}
        </button>

        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <PersonaAvatar persona={persona} sizeClassName="h-16 w-16" textClassName="text-xl" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">{persona.name}</h1>
              <p className="text-muted-foreground">{persona.role}</p>
            </div>
          </div>
          <Button onClick={handlePrimaryAction} disabled={hiring} size="lg">
            {hiring ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t("templates.marketplace.hiring")}
              </>
            ) : template.hired ? (
              t("templates.marketplace.chat")
            ) : (
              t("templates.marketplace.hire", { name: persona.name })
            )}
          </Button>
        </div>

        <p className="mb-8 max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
          {template.description}
        </p>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          <div className="flex flex-col gap-8 lg:col-span-2">
            {template.features.length > 0 && (
              <section>
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {t("templates.marketplace.whatItDoes")}
                </h2>
                <ul className="flex flex-col gap-2">
                  {template.features.map((feature, index) => (
                    <li
                      key={index}
                      className="flex gap-2 text-[14px] leading-relaxed text-foreground/90"
                    >
                      <span className="flex-none text-muted-foreground/60">›</span>
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {samplePrompts.length > 0 && (
              <section>
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {t("templates.marketplace.tryAsking")}
                </h2>
                <div className="flex flex-col gap-2">
                  {samplePrompts.map((prompt, index) => (
                    <div
                      key={index}
                      className="rounded-[10px] border border-border bg-muted/40 px-3.5 py-2.5 text-[13.5px] text-foreground/90"
                    >
                      {prompt.prompt}
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>

          <div className="flex flex-col gap-5 rounded-[14px] border border-border bg-card p-5">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {t("templates.marketplace.whatsIncluded")}
            </h2>

            {template.agent_config?.execution_mode && (
              <div>
                <div className="text-[11.5px] font-medium text-muted-foreground">
                  {t("templates.marketplace.thinking")}
                </div>
                <div className="text-[13.5px] font-medium text-foreground">
                  {capitalize(template.agent_config.execution_mode)}
                </div>
              </div>
            )}

            {toolCategories.length > 0 && (
              <div>
                <div className="mb-1.5 text-[11.5px] font-medium text-muted-foreground">
                  {t("templates.marketplace.tools")}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {toolCategories.map((category) => (
                    <span
                      key={category}
                      className="rounded-full bg-muted px-2.5 py-1 text-[11.5px] font-medium text-foreground/80"
                    >
                      {getToolLabel(category)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {connectedApps.length > 0 && (
              <div>
                <div className="mb-1.5 text-[11.5px] font-medium text-muted-foreground">
                  {t("templates.marketplace.connectedApps")}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {connectedApps.map((app) => (
                    <span
                      key={app.name}
                      className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-[11.5px] font-medium text-foreground/80"
                    >
                      {app.logo ? (
                        <img src={app.logo} alt="" className="h-3.5 w-3.5" />
                      ) : null}
                      {app.name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {skills.length > 0 && (
              <div>
                <div className="mb-1.5 text-[11.5px] font-medium text-muted-foreground">
                  {t("templates.marketplace.skills")}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {skills.map((skill) => (
                    <span
                      key={skill}
                      className="rounded-full bg-muted px-2.5 py-1 text-[11.5px] font-medium text-foreground/80"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {!template.hired && (
          <button
            type="button"
            onClick={() => router.push(`/build/new?template=${template.id}`)}
            className="mt-8 text-[13px] font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          >
            {t("templates.marketplace.customizeBeforeHiring")}
          </button>
        )}
      </div>
    </div>
  );
}
