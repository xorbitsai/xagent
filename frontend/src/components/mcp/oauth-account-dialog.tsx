"use client"

import React, { useCallback, useId, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useI18n } from "@/contexts/i18n-context"
import { normalizeOAuthAccount, type OAuthAccountInput } from "@/lib/oauth-account"

export interface OAuthAccountRequest {
  /** What is being connected (an app id, or a provider for a batch login),
   * so a retry pre-fills the label last submitted for the same thing. */
  key: string
  name: string
  icon?: string
  input: OAuthAccountInput
  /** Receives the normalized label. Called synchronously from the submit
   * handler, so a popup opened here keeps the click's user activation and
   * is not blocked. */
  onSubmit: (account: string) => void
}

interface PendingRequest {
  request: OAuthAccountRequest
  initialValue: string
  retry: boolean
  seq: number
}

/**
 * The input step that collects the account identifier before a builtin
 * OAuth popup for a provider declaring `oauth_account_input`. Render
 * `accountDialog` once, and call `requestAccount` instead of opening the
 * popup directly. Pass `retry` when the previous popup closed without
 * connecting: any error the login route returned was shown in that popup
 * (there is no channel back to the opener), so the step reopens with the
 * last label and a hint to check it.
 */
export function useOAuthAccountStep() {
  const [pending, setPending] = useState<PendingRequest | null>(null)
  const lastAccounts = useRef(new Map<string, string>())
  const seq = useRef(0)

  const requestAccount = useCallback(
    (request: OAuthAccountRequest, options: { retry?: boolean } = {}) => {
      seq.current += 1
      setPending({
        request,
        initialValue: options.retry ? lastAccounts.current.get(request.key) ?? "" : "",
        retry: !!options.retry,
        seq: seq.current,
      })
    },
    [],
  )

  const accountDialog = (
    <OAuthAccountDialog
      // Remount per request so the input starts from that request's value.
      key={pending?.seq ?? 0}
      pending={pending}
      onCancel={() => setPending(null)}
      onSubmit={(account) => {
        if (!pending) return
        lastAccounts.current.set(pending.request.key, account)
        setPending(null)
        pending.request.onSubmit(account)
      }}
    />
  )

  return { requestAccount, accountDialog }
}

function OAuthAccountDialog({
  pending,
  onCancel,
  onSubmit,
}: {
  pending: PendingRequest | null
  onCancel: () => void
  onSubmit: (account: string) => void
}) {
  const { t } = useI18n()
  const [value, setValue] = useState(pending?.initialValue ?? "")
  const [invalid, setInvalid] = useState(false)
  const inputId = useId()
  const helpId = useId()
  const errorId = useId()
  const request = pending?.request

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!request) return
    const account = normalizeOAuthAccount(value, request.input.suffix)
    if (!account) {
      setInvalid(true)
      return
    }
    onSubmit(account)
  }

  return (
    <Dialog open={!!pending} onOpenChange={(nextOpen) => { if (!nextOpen) onCancel() }}>
      <DialogContent className="sm:max-w-md" aria-describedby={undefined}>
        {request && (
          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <DialogTitle className="flex items-center gap-2">
              {request.icon && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={request.icon} alt="" className="h-5 w-5" />
              )}
              {t("tools.mcp.dialog.connect")} {request.name}
            </DialogTitle>
            <div className="space-y-1.5">
              <Label htmlFor={inputId}>{request.input.label}</Label>
              <div className="flex items-center gap-2">
                <Input
                  id={inputId}
                  autoFocus
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  placeholder={request.input.placeholder}
                  value={value}
                  aria-invalid={invalid}
                  aria-describedby={invalid ? errorId : request.input.help ? helpId : undefined}
                  onChange={(event) => {
                    setValue(event.target.value)
                    setInvalid(false)
                  }}
                />
                {request.input.suffix && (
                  <span className="shrink-0 text-sm text-muted-foreground">{request.input.suffix}</span>
                )}
              </div>
              {request.input.help && (
                <p id={helpId} className="text-xs text-muted-foreground">{request.input.help}</p>
              )}
              {invalid ? (
                <p id={errorId} role="alert" className="text-xs text-destructive">
                  {t("tools.mcp.accountStep.invalid")}
                </p>
              ) : pending.retry ? (
                <p className="text-xs text-muted-foreground">{t("tools.mcp.accountStep.retryHint")}</p>
              ) : null}
            </div>
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={onCancel}>
                {t("common.cancel")}
              </Button>
              <Button type="submit">{t("tools.mcp.accountStep.continue")}</Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
