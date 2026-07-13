
import React, { useEffect, useState } from 'react';
import { getApiUrl } from '@/lib/utils';
import { apiRequest } from '@/lib/api-wrapper';
import { ChevronDown, Sparkles } from 'lucide-react';
import { useI18n, type Locale } from '@/contexts/i18n-context';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

interface TokenUsageDisplayProps {
  taskId: number | null;
  isRunning: boolean;
  className?: string;
}

interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  llm_calls: number;
  model_usage: ModelTokenUsage[];
}

interface ModelTokenUsage {
  model_id: string;
  model_name: string;
  input_tokens: number;
  output_tokens: number;
}

// Build formatters lazily per locale so this file stays valid regardless of the
// locale set (the SaaS overlay adds more locales than the standalone build).
const compactTokenFormatters = new Map<Locale, Intl.NumberFormat>();
const exactTokenFormatters = new Map<Locale, Intl.NumberFormat>();

function getFormatter(
  cache: Map<Locale, Intl.NumberFormat>,
  locale: Locale,
  options?: Intl.NumberFormatOptions,
): Intl.NumberFormat {
  let formatter = cache.get(locale);
  if (!formatter) {
    formatter = new Intl.NumberFormat(locale, options);
    cache.set(locale, formatter);
  }
  return formatter;
}

function normalizeTokenCount(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}

export function formatTokenCount(value: number, locale: Locale = 'en'): string {
  return getFormatter(compactTokenFormatters, locale, {
    notation: 'compact',
    compactDisplay: 'short',
    maximumFractionDigits: 2,
  })
    .format(normalizeTokenCount(value))
    .toLowerCase();
}

export function formatExactTokenCount(value: number, locale: Locale = 'en'): string {
  return getFormatter(exactTokenFormatters, locale).format(normalizeTokenCount(value));
}

export function TokenUsageDisplay({ taskId, isRunning, className }: TokenUsageDisplayProps) {
  const [usage, setUsage] = useState<TokenUsage | null>(null);
  const { locale, t } = useI18n();

  useEffect(() => {
    if (!taskId) return;

    let isMounted = true;
    let intervalId: NodeJS.Timeout;

    const fetchUsage = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/chat/task/${taskId}`);
        if (response.ok && isMounted) {
          const data = await response.json();
          setUsage({
            input_tokens: data.input_tokens || 0,
            output_tokens: data.output_tokens || 0,
            total_tokens: data.total_tokens || 0,
            llm_calls: data.llm_calls || 0,
            model_usage: Array.isArray(data.model_usage) ? data.model_usage : [],
          });
        }
      } catch (error) {
        console.error('Failed to fetch token usage:', error);
      }
    };

    // Initial fetch
    fetchUsage();

    // Poll if running
    if (isRunning) {
      intervalId = setInterval(fetchUsage, 15000); // Poll every 15 seconds
    }

    return () => {
      isMounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [taskId, isRunning]);

  if (!usage) return null;

  const attributedModelCount = usage.model_usage.filter((model) => Boolean(model.model_id)).length;
  const unattributedUsageCount = usage.model_usage.length - attributedModelCount;
  const modelUsageLabel = attributedModelCount === 0
    ? t('chatPage.tokenUsage.unattributedCount', { count: unattributedUsageCount })
    : unattributedUsageCount > 0
      ? t(
          attributedModelCount === 1
            ? 'chatPage.tokenUsage.oneModelWithUnattributed'
            : 'chatPage.tokenUsage.modelsWithUnattributed',
          { count: attributedModelCount, unattributed: unattributedUsageCount },
        )
      : t(
          attributedModelCount === 1
            ? 'chatPage.tokenUsage.oneModel'
            : 'chatPage.tokenUsage.models',
          { count: attributedModelCount },
        );

  return (
    <div className={`inline-flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border bg-card/80 px-3 py-2 text-xs sm:text-sm ${className || ""}`}>
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <Sparkles className="w-4 h-4 text-indigo-500" />
        <span className="font-medium text-foreground" title={formatExactTokenCount(usage.input_tokens, locale)}>
          {formatTokenCount(usage.input_tokens, locale)}
        </span>
        <span
          className="text-muted-foreground"
          title={t('chatPage.tokenUsage.input')}
        >
          {t('chatPage.tokenUsage.inputShort')}
        </span>
      </span>
      <span className="flex items-center gap-1.5 whitespace-nowrap">
        <span className="font-medium text-foreground" title={formatExactTokenCount(usage.output_tokens, locale)}>
          {formatTokenCount(usage.output_tokens, locale)}
        </span>
        <span
          className="text-muted-foreground"
          title={t('chatPage.tokenUsage.output')}
        >
          {t('chatPage.tokenUsage.outputShort')}
        </span>
      </span>
      {usage.model_usage.length > 0 && (
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <span>{modelUsageLabel}</span>
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="end"
            className="w-[28rem] max-w-[calc(100vw-2rem)] p-0"
          >
            <div className="border-b px-3 py-2.5 text-sm font-medium">
              {t('chatPage.tokenUsage.byModel')}
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_5rem_5rem] gap-x-4 gap-y-2 p-3 text-xs">
              <span className="text-muted-foreground">{t('chatPage.tokenUsage.model')}</span>
              <span className="text-right text-muted-foreground">
                {t('chatPage.tokenUsage.inputShort')}
              </span>
              <span className="text-right text-muted-foreground">
                {t('chatPage.tokenUsage.outputShort')}
              </span>
              {usage.model_usage.map((model) => (
                <React.Fragment key={JSON.stringify([model.model_id, model.model_name])}>
                  <span className="min-w-0" title={model.model_name || model.model_id}>
                    <span className="block truncate font-medium">
                      {model.model_name || model.model_id || t('chatPage.tokenUsage.unknownModel')}
                    </span>
                    {model.model_id && model.model_name && model.model_id !== model.model_name && (
                      <span className="block truncate text-[10px] text-muted-foreground">
                        {model.model_id}
                      </span>
                    )}
                    {!model.model_id && model.model_name && (
                      <span className="block truncate text-[10px] text-muted-foreground">
                        {t('chatPage.tokenUsage.unattributed')}
                      </span>
                    )}
                  </span>
                  <span className="text-right tabular-nums" title={formatExactTokenCount(model.input_tokens, locale)}>
                    {formatTokenCount(model.input_tokens, locale)}
                  </span>
                  <span className="text-right tabular-nums" title={formatExactTokenCount(model.output_tokens, locale)}>
                    {formatTokenCount(model.output_tokens, locale)}
                  </span>
                </React.Fragment>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
}
