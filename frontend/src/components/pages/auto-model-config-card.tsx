"use client";

import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Check, Settings2, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { MultiSelect } from "@/components/ui/multi-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select-radix";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/components/ui/sonner";
import { useI18n } from "@/contexts/i18n-context";
import { apiRequest } from "@/lib/api-wrapper";
import { cn, getApiUrl } from "@/lib/utils";

import type { Model } from "./models";

interface RouterProfile {
  id: string;
  provider?: string;
  aliases?: string[] | null;
  input_modalities: string[];
  context_window?: number;
}

interface AutoCandidate {
  routing_model_id: string;
  target_model_id: number;
  target_model: Model;
}

interface AutoConfig {
  configured: boolean;
  fallback_model_id?: number;
  auto_model?: Model;
  candidates: AutoCandidate[];
}

interface Props {
  models: Model[];
  generalDefault?: Model;
  onSuccess: () => Promise<void>;
}

function normalizedValues(model: Model): string[] {
  const raw = model.model_name.trim().toLowerCase();
  const provider = model.model_provider.trim().toLowerCase();
  const values = new Set([raw, model.model_id.trim().toLowerCase()]);
  const providerAliases: Record<string, string[]> = {
    claude: ["anthropic"],
    gemini: ["google"],
    zhipu: ["z-ai"],
    "zai-coding-plan": ["z-ai"],
    "zhipuai-coding-plan": ["z-ai"],
    "kimi-for-coding": ["moonshotai"],
  };
  for (const prefix of [provider, ...(providerAliases[provider] || [])]) {
    values.add(`${prefix}/${raw}`);
  }
  return [...values];
}

export function guessProfile(
  model: Model,
  profiles: RouterProfile[],
): string | undefined {
  const modelValues = new Set(normalizedValues(model));
  return profiles.find((profile) => {
    const profileValues = [profile.id, ...(profile.aliases || [])].map((value) =>
      value.trim().toLowerCase(),
    );
    return profileValues.some((value) => modelValues.has(value));
  })?.id;
}

async function errorMessage(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : fallback;
  } catch {
    return fallback;
  }
}

export function AutoModelConfigCard({
  models,
  generalDefault,
  onSuccess,
}: Props) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [routerAvailable, setRouterAvailable] = useState<boolean | null>(null);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<AutoConfig | null>(null);
  const [profiles, setProfiles] = useState<RouterProfile[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [bindings, setBindings] = useState<Record<string, string>>({});
  const [fallbackId, setFallbackId] = useState<string>("");
  const [setAsDefault, setSetAsDefault] = useState(false);

  const availableModels = useMemo(
    () =>
      models.filter(
        (model) =>
          model.category === "llm" &&
          model.is_active &&
          !(
            model.model_name.trim().toLowerCase() === "auto" &&
            ["router", "openrouter"].includes(
              model.model_provider.toLowerCase(),
            )
          ),
      ),
    [models],
  );

  const selectedModels = useMemo(
    () =>
      selectedIds
        .map((id) => availableModels.find((model) => String(model.id) === id))
        .filter((model): model is Model => Boolean(model)),
    [availableModels, selectedIds],
  );

  const hydrateForm = (
    nextConfig: AutoConfig,
    nextProfiles: RouterProfile[],
  ) => {
    setConfig(nextConfig);
    setSelectedIds(
      nextConfig.candidates.map((candidate) =>
        String(candidate.target_model_id),
      ),
    );
    setBindings(
      Object.fromEntries(
        nextConfig.candidates.map((candidate) => [
          String(candidate.target_model_id),
          candidate.routing_model_id,
        ]),
      ),
    );
    setFallbackId(
      nextConfig.fallback_model_id ? String(nextConfig.fallback_model_id) : "",
    );
    setSetAsDefault(
      Boolean(
        nextConfig.auto_model &&
        generalDefault?.id === nextConfig.auto_model.id,
      ),
    );
    setProfiles(nextProfiles);
  };

  const loadConfig = async () => {
    setLoading(true);
    try {
      const [configResponse, profilesResponse] = await Promise.all([
        apiRequest(`${getApiUrl()}/api/models/auto-config`, { headers: {} }),
        apiRequest(`${getApiUrl()}/api/models/auto-config/profiles`, {
          headers: {},
        }),
      ]);
      if (!configResponse.ok) {
        throw new Error(
          await errorMessage(
            configResponse,
            t("models.auto.errors.loadFailed"),
          ),
        );
      }
      if (profilesResponse.status === 503) {
        setRouterAvailable(false);
        return;
      }
      if (!profilesResponse.ok) {
        throw new Error(
          await errorMessage(
            profilesResponse,
            t("models.auto.errors.profilesFailed"),
          ),
        );
      }
      hydrateForm(await configResponse.json(), await profilesResponse.json());
      setRouterAvailable(true);
    } catch (error) {
      setRouterAvailable(true);
      toast.error(
        error instanceof Error
          ? error.message
          : t("models.auto.errors.loadFailed"),
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (config?.auto_model) {
      setSetAsDefault(generalDefault?.id === config.auto_model.id);
    }
  }, [config?.auto_model, generalDefault?.id]);

  const handleSelectedModels = (ids: string[]) => {
    setSelectedIds(ids);
    const nextBindings: Record<string, string> = {};
    for (const id of ids) {
      const existing = bindings[id];
      const model = availableModels.find(
        (candidate) => String(candidate.id) === id,
      );
      nextBindings[id] =
        existing || (model ? guessProfile(model, profiles) : undefined) || "";
    }
    setBindings(nextBindings);
    if (!ids.includes(fallbackId)) {
      setFallbackId(ids[0] || "");
    }
  };

  const duplicateProfiles = useMemo(() => {
    const counts = Object.values(bindings).reduce<Record<string, number>>(
      (acc, value) => {
        if (value) acc[value] = (acc[value] || 0) + 1;
        return acc;
      },
      {},
    );
    return new Set(
      Object.entries(counts)
        .filter(([, count]) => count > 1)
        .map(([id]) => id),
    );
  }, [bindings]);

  const canSave =
    selectedModels.length > 0 &&
    selectedModels.every((model) => Boolean(bindings[String(model.id)])) &&
    duplicateProfiles.size === 0 &&
    Boolean(fallbackId);

  const save = async () => {
    if (!canSave) return;
    setSaving(true);
    try {
      const response = await apiRequest(
        `${getApiUrl()}/api/models/auto-config`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fallback_model_id: Number(fallbackId),
            set_as_default: setAsDefault,
            candidates: selectedModels.map((model) => ({
              target_model_id: model.id,
              routing_model_id: bindings[String(model.id)],
            })),
          }),
        },
      );
      if (!response.ok) {
        throw new Error(
          await errorMessage(response, t("models.auto.errors.saveFailed")),
        );
      }
      const saved: AutoConfig = await response.json();
      if (!saved.auto_model) {
        throw new Error(t("models.auto.errors.saveFailed"));
      }

      hydrateForm(saved, profiles);
      await onSuccess();
      setOpen(false);
      toast.success(t("models.auto.saved"));
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : t("models.auto.errors.saveFailed"),
      );
    } finally {
      setSaving(false);
    }
  };

  if (routerAvailable !== true) {
    return null;
  }

  return (
    <>
      <Card className="mb-8 overflow-hidden border-primary/25 bg-gradient-to-br from-primary/[0.07] via-background to-background">
        <div className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between">
          <div className="flex gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <BrainCircuit className="h-6 w-6" />
            </div>
            <div>
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-semibold">
                  {t("models.auto.title")}
                </h2>
                <Badge variant={config?.configured ? "default" : "secondary"}>
                  {config?.configured
                    ? t("models.auto.configured")
                    : t("models.auto.notConfigured")}
                </Badge>
                {config?.auto_model &&
                  generalDefault?.id === config.auto_model.id && (
                    <Badge variant="outline" className="gap-1">
                      <Check className="h-3 w-3" />{" "}
                      {t("models.auto.defaultBadge")}
                    </Badge>
                  )}
              </div>
              <p className="max-w-2xl text-sm text-muted-foreground">
                {t("models.auto.description")}
              </p>
              {config?.configured && (
                <p className="mt-2 text-xs text-muted-foreground">
                  {t("models.auto.summary", {
                    count: config.candidates.length,
                  })}
                </p>
              )}
            </div>
          </div>
          <Button
            variant={config?.configured ? "outline" : "default"}
            onClick={() => setOpen(true)}
            disabled={loading}
            className="shrink-0 gap-2"
          >
            {config?.configured ? (
              <Settings2 className="h-4 w-4" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            {config?.configured
              ? t("models.auto.edit")
              : t("models.auto.configure")}
          </Button>
        </div>
      </Card>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("models.auto.dialogTitle")}</DialogTitle>
            <DialogDescription>
              {t("models.auto.dialogDescription")}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-2">
            <div className="space-y-2">
              <Label>{t("models.auto.candidateModels")}</Label>
              <MultiSelect
                values={selectedIds}
                onValuesChange={handleSelectedModels}
                options={availableModels.map((model) => ({
                  value: String(model.id),
                  label: model.model_name,
                  description: model.model_provider,
                }))}
                searchable
                placeholder={t("models.auto.selectModels")}
              />
              {availableModels.length === 0 && (
                <p className="text-xs text-amber-600">
                  {t("models.auto.noModels")}
                </p>
              )}
            </div>

            {selectedModels.length > 0 && (
              <div className="space-y-3">
                <Label>{t("models.auto.profileBindings")}</Label>
                {selectedModels.map((model) => {
                  const binding = bindings[String(model.id)] || "";
                  return (
                    <div
                      key={model.id}
                      className="grid gap-2 rounded-lg border p-3 md:grid-cols-[1fr_1.4fr] md:items-center"
                    >
                      <div>
                        <div className="text-sm font-medium">
                          {model.model_name}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {model.model_provider}
                        </div>
                      </div>
                      <Select
                        value={binding}
                        onValueChange={(value) =>
                          setBindings((current) => ({
                            ...current,
                            [String(model.id)]: value,
                          }))
                        }
                      >
                        <SelectTrigger
                          className={cn(
                            duplicateProfiles.has(binding) &&
                              "border-destructive",
                          )}
                        >
                          <SelectValue
                            placeholder={t("models.auto.selectProfile")}
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {profiles.map((profile) => (
                            <SelectItem key={profile.id} value={profile.id}>
                              {profile.id}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  );
                })}
                {duplicateProfiles.size > 0 && (
                  <p className="text-xs text-destructive">
                    {t("models.auto.duplicateProfile")}
                  </p>
                )}
              </div>
            )}

            <div className="space-y-2">
              <Label>{t("models.auto.fallback")}</Label>
              <Select
                value={fallbackId}
                onValueChange={setFallbackId}
                disabled={!selectedModels.length}
              >
                <SelectTrigger>
                  <SelectValue placeholder={t("models.auto.selectFallback")} />
                </SelectTrigger>
                <SelectContent>
                  {selectedModels.map((model) => (
                    <SelectItem key={model.id} value={String(model.id)}>
                      {model.model_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {t("models.auto.fallbackHelp")}
              </p>
            </div>

            <div className="flex items-center justify-between gap-4 rounded-lg border p-4">
              <div>
                <Label htmlFor="auto-default">
                  {t("models.auto.setDefault")}
                </Label>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("models.auto.setDefaultHelp")}
                </p>
              </div>
              <Switch
                id="auto-default"
                checked={setAsDefault}
                onCheckedChange={setSetAsDefault}
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={saving}
            >
              {t("common.cancel")}
            </Button>
            <Button onClick={save} disabled={!canSave || saving}>
              {saving ? t("models.auto.saving") : t("models.auto.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
