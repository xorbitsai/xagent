import { Button } from "@/components/ui/button"
import { Code, Eye, Download, ExternalLink } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { isToggleableFile } from "@/lib/utils"

interface FilePreviewActionButtonsProps {
  viewMode: 'preview' | 'code'
  onViewModeChange: (mode: 'preview' | 'code') => void
  fileName: string
  onDownload: () => void
  onOpenInNewWindow?: () => void
  showText?: boolean
}

export function FilePreviewActionButtons({
  viewMode,
  onViewModeChange,
  fileName,
  onDownload,
  onOpenInNewWindow,
  showText = true
}: FilePreviewActionButtonsProps) {
  const { t } = useI18n()

  const isToggleable = isToggleableFile(fileName)

  return (
    <div className="flex items-center gap-2">
      {isToggleable && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => onViewModeChange(viewMode === 'preview' ? 'code' : 'preview')}
          className="flex items-center gap-2"
        >
          {viewMode === 'preview' ? (
            <>
              <Code className="h-4 w-4" />
              {showText && t('common.code')}
            </>
          ) : (
            <>
              <Eye className="h-4 w-4" />
              {showText && t('common.preview')}
            </>
          )}
        </Button>
      )}

      {onOpenInNewWindow && (
        <Button
          variant="outline"
          size="sm"
          onClick={onOpenInNewWindow}
          className={showText ? "flex items-center gap-2" : "h-8 w-8 p-0"}
          title={t('files.previewDialog.buttons.openInNewWindow')}
        >
          <ExternalLink className="h-4 w-4" />
          {showText && t('files.previewDialog.buttons.openInNewWindow')}
        </Button>
      )}

      <Button
        variant="outline"
        size="sm"
        onClick={onDownload}
        className={showText ? "flex items-center gap-2" : "h-8 w-8 p-0"}
        title={t('files.previewDialog.buttons.download')}
        aria-label={t('files.previewDialog.buttons.download')}
      >
        <Download className="h-4 w-4" />
        {showText && t('files.previewDialog.buttons.download')}
      </Button>
    </div>
  )
}
