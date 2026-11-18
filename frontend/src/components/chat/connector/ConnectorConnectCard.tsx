import React, { useCallback, useEffect, useId, useRef, useState } from "react"
import { useI18n } from "@/contexts/i18n-context"
import type { TranslationKey } from "@/i18n/translations"
import { getConnectorState, resumeConnector, saveConnector, skipConnector } from "@/lib/connector-connect"
import type { ConnectorConnectResult, ConnectorErrorCode } from "@/lib/page-extension-contracts"
import { ConnectorLogo, CARD, CARD_W } from "./ConnectorLogo"
import { ConnectorConnectedBody, LockIcon } from "./ConnectorConnectedBody"
import type { ConnectorInfo } from "./types"

type ConnectorCardState = "idle" | "saving" | "connected" | "failed" | "skipped" | "unavailable" | "stale"

const ROUTE_KEYS = { native: "connector.connect.routeNative", mcp: "connector.connect.routeMcp", api: "connector.connect.routeApi" } as const
const STATE_KEYS: Record<ConnectorCardState, TranslationKey> = {
  idle: "connector.connect.stateIdle",
  saving: "connector.connect.stateSaving",
  connected: "connector.connect.stateConnected",
  failed: "connector.connect.stateFailed",
  skipped: "connector.connect.stateSkipped",
  unavailable: "connector.connect.stateUnavailable",
  stale: "connector.connect.stateStale",
}
const ERROR_KEYS: Record<ConnectorErrorCode, TranslationKey> = {
  auth_failed: "connector.connect.errorAuthFailed",
  unreachable: "connector.connect.errorUnreachable",
  timeout: "connector.connect.errorTimeout",
  no_tools: "connector.connect.errorNoTools",
  invalid_field: "connector.connect.errorInvalidField",
}

interface Props {
  connector: ConnectorInfo
  /**
   * Whether the task is waiting on exactly this interaction. Only then can the
   * card be acted on; an unsettled card that is not pending (history, a task
   * that moved on) is read-only. The card never sends a chat message: the host
   * settles the interaction when a save or skip succeeds and the task continues
   * through the normal event stream.
   */
  pending: boolean
}

export function ConnectorConnectCard({ connector, pending }: Props) {
  const { t } = useI18n()
  const errorId = useId()
  const [state, setState] = useState<ConnectorCardState>("idle")
  const [skipping, setSkipping] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<ConnectorConnectResult | null>(null)
  const [failure, setFailure] = useState<{ code: ConnectorErrorCode | null; field: string | null } | null>(null)
  const [skipError, setSkipError] = useState(false)
  // The card is settled but the host has not (or may not have) continued the task.
  const [needsContinue, setNeedsContinue] = useState(false)
  const [continuing, setContinuing] = useState(false)
  const [continueError, setContinueError] = useState(false)
  // Field keys whose value is empty after trimming (a whitespace-only entry passes `required`).
  const [emptyKeys, setEmptyKeys] = useState<string[]>([])
  // Set once the user starts a save or skip, so a late restore cannot overwrite it.
  const actedRef = useRef(false)
  const mountedRef = useRef(true)
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false } }, [])

  // Reload: restore the card from the server record. Failure here is not an error state.
  useEffect(() => {
    let cancelled = false
    getConnectorState(connector.interaction_id)
      .then((res) => {
        if (cancelled || actedRef.current) return
        if (res.state === "unavailable") {
          setState("unavailable")
        } else if (res.state === "connected") {
          setResult(res); setState("connected"); setNeedsContinue(true)
        } else if (res.state === "skipped") {
          setState("skipped"); setNeedsContinue(true)
        }
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [connector.interaction_id])

  const busy = state === "saving" || skipping
  const locked = busy || !pending
  const ready = connector.fields.every((f) => (values[f.key] ?? "").trim() !== "")

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (locked) return
    if (!ready) {
      setEmptyKeys(connector.fields.filter((f) => (values[f.key] ?? "").trim() === "").map((f) => f.key))
      return
    }
    actedRef.current = true
    const body = Object.fromEntries(connector.fields.map((f) => [f.key, (values[f.key] ?? "").trim()]))
    setState("saving")
    setFailure(null)
    try {
      const res = await saveConnector(connector.interaction_id, connector.app_id, body)
      if (!mountedRef.current) return
      if (res.state === "unavailable") {
        setState("unavailable")
      } else if (res.state === "connected") {
        setValues({})
        setResult(res)
        setState("connected")
        setNeedsContinue(res.taskResumed !== true)
      } else {
        setFailure({ code: res.errorCode, field: res.errorField })
        setState("failed")
      }
    } catch {
      if (!mountedRef.current) return
      setFailure({ code: null, field: null })
      setState("failed")
    }
  }

  const skip = async () => {
    if (locked) return
    actedRef.current = true
    setSkipError(false)
    setSkipping(true)
    try {
      const res = await skipConnector(connector.interaction_id)
      if (!mountedRef.current) return
      if (res.state === "unavailable") {
        setState("unavailable")
      } else if (res.state !== "skipped") {
        // A failure the host described (or an unexpected state) is not a skip.
        setSkipError(true)
      } else {
        setValues({})
        setState("skipped")
        setNeedsContinue(res.taskResumed !== true)
      }
    } catch {
      if (mountedRef.current) setSkipError(true)
    } finally {
      if (mountedRef.current) setSkipping(false)
    }
  }

  const fieldKnown = !!failure?.field && connector.fields.some((f) => f.key === failure.field)
  // Asks the host to continue the task; idempotent there, so only in-flight is guarded.
  const continueTask = async () => {
    if (continuing) return
    setContinuing(true)
    setContinueError(false)
    try {
      const res = await resumeConnector(connector.interaction_id)
      if (!mountedRef.current) return
      if (res.state !== "unavailable" && res.taskResumed === true) setNeedsContinue(false)
      else setContinueError(true)
    } catch {
      if (mountedRef.current) setContinueError(true)
    } finally {
      if (mountedRef.current) setContinuing(false)
    }
  }

  // An unsettled card the task is no longer waiting on (the batch was resumed by
  // another reply, or the task moved on) is not needed any more.
  const shownState: ConnectorCardState = !pending && (state === "idle" || state === "failed") ? "stale" : state
  const errorText = failure ? t(failure.code ? ERROR_KEYS[failure.code] : "connector.connect.errorGeneric") : null
  const stateColor = shownState === "connected" ? "text-[#15804A]" : shownState === "failed" ? "text-[#C0322E]" : "text-[#6E6E7A]"
  const body = shownState === "skipped" ? null : shownState === "stale" ? (
    <p className="m-0 text-[12.5px] leading-[1.5] text-[#3D3D46]">{t("connector.connect.stale")}</p>
  ) : state === "unavailable" ? (
    <p role="status" className="m-0 text-[12.5px] leading-[1.5] text-[#3D3D46]">{t("connector.connect.unavailable")}</p>
  ) : state === "connected" ? (
    <ConnectorConnectedBody
      secretMasked={result?.secretMasked ?? null}
      agentName={result?.agentName ?? connector.agent_name ?? null}
      scope={result?.scope ?? "local"}
      tools={result?.tools ?? []}
    />
  ) : (
    <>
      {connector.key_label && (
        <b className="mb-[6px] block text-[12px] font-[650] text-[#3D3D46]">{t("connector.connect.howTo", { what: connector.key_label })}</b>
      )}
      {connector.how.length > 0 && (
        <ol className="mb-[12px] mt-0 list-decimal pl-[19px] text-[12.5px] leading-[1.5] text-[#3D3D46] [&>li+li]:mt-[3px]">
          {connector.how.map((step, i) => <li key={i}>{step}</li>)}
        </ol>
      )}
      <form
        autoComplete="off"
        onSubmit={submit}
        className="grid gap-[10px] rounded-[12px] border border-[#E7E7EC] bg-[#FAFAFA] p-[12px]"
      >
        {connector.fields.map((field) => {
          const isEmpty = emptyKeys.includes(field.key)
          const fieldError = isEmpty || (fieldKnown && failure?.field === field.key)
          return (
            <div key={field.key} className="grid gap-[5px]">
            <label className="grid gap-[5px] text-[12px] font-[550] text-[#3D3D46]">
              {field.label}
              <input
                name={field.key}
                type={field.secret ? "password" : "text"}
                placeholder={field.placeholder}
                // "off" is ignored for password inputs; "new-password" keeps
                // password managers from offering to save or fill an API key.
                autoComplete={field.secret ? "new-password" : "off"}
                spellCheck={false}
                required
                disabled={locked}
                value={values[field.key] ?? ""}
                aria-invalid={fieldError || undefined}
                aria-describedby={fieldError ? `${errorId}-${field.key}` : undefined}
                onChange={(e) => { setValues((v) => ({ ...v, [field.key]: e.target.value })); setEmptyKeys((k) => k.filter((x) => x !== field.key)) }}
                className={`w-full rounded-[10px] border bg-white px-[11px] py-[9px] text-[14px] text-[#1A1A1F] outline-none placeholder:text-[#6E6E7A] focus:border-[#BFC6FA] focus:shadow-[0_0_0_3px_#EEF1FE] ${fieldError ? "border-[#C0322E]" : "border-[#E7E7EC]"}`}
              />
            </label>
              {fieldError && (isEmpty || errorText) && <span id={`${errorId}-${field.key}`} role="alert" className="text-[11.5px] text-[#C0322E]">{isEmpty ? t("connector.connect.fieldRequired") : errorText}</span>}
            </div>
          )
        })}
        <div className="flex items-start gap-[7px] text-[11.5px] leading-[1.45] text-[#6B6B76]">
          <LockIcon size={14} className="mt-[1px] shrink-0 basis-[14px] fill-none stroke-[#15804A] stroke-2 [stroke-linecap:round] [stroke-linejoin:round]" />
          <span>{t("connector.connect.lock")}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="submit"
            disabled={locked}
            className="inline-flex items-center justify-center gap-[7px] rounded-[10px] bg-[#2536E0] px-[15px] py-[9px] text-[13px] font-[600] text-white disabled:opacity-60"
          >
            {state === "saving" ? t("connector.connect.saving") : t("connector.connect.save")}
          </button>
          <button type="button" disabled={locked} onClick={skip} className="rounded-[9px] px-[11px] py-[8px] text-[12.5px] text-[#6B6B76] hover:bg-[#F5F5F7]">
            {t("connector.connect.skip")}
          </button>
        </div>
        {failure && !fieldKnown && errorText && <p role="alert" className="m-0 text-[11.5px] text-[#C0322E]">{errorText}</p>}
        {skipError && <p role="alert" className="m-0 text-[11.5px] text-[#C0322E]">{t("connector.connect.skipFailed")}</p>}
      </form>
    </>
  )

  return (
    <div className={`${CARD} ${CARD_W}`} data-testid="connector-connect-card" data-s={shownState}>
      <div className="flex items-center gap-[11px] px-[15px] py-[13px]">
        <ConnectorLogo logo={connector.logo} />
        <div className="min-w-0 flex-1">
          <b className="flex items-center gap-[7px] text-[14px] font-[620] tracking-[-0.01em]">
            {connector.name}
            <span className="mt-[2px] block rounded-full bg-[#F5F5F7] px-[7px] py-[2px] text-[12px] font-[700] uppercase leading-[1.4] tracking-[0.05em] text-[#6B6B76]">{t(ROUTE_KEYS[connector.route])}</span>
          </b>
          {connector.what && <span className="mt-[2px] block text-[12px] leading-[1.4] text-[#6B6B76]">{t("connector.connect.lets", { what: connector.what })}</span>}
        </div>
        <span role="status" className={`shrink-0 text-[11.5px] font-[600] ${stateColor}`}>{t(STATE_KEYS[shownState])}</span>
      </div>
      {body && <div className="border-t border-[#F0F0F4] px-[15px] pb-[14px] pt-[12px]">{body}</div>}
      {needsContinue && pending && (state === "connected" || state === "skipped") && (
        <div className="flex flex-wrap items-center gap-2 border-t border-[#F0F0F4] px-[15px] py-[10px] text-[12.5px] text-[#3D3D46]">
          <span role={continueError ? "alert" : undefined}>{t(continueError ? "connector.connect.continueFailed" : "connector.connect.continueHint")}</span>
          <button
            type="button"
            disabled={continuing}
            onClick={() => void continueTask()}
            className="rounded-[9px] bg-[#2536E0] px-[11px] py-[6px] text-[12.5px] font-[600] text-white disabled:opacity-60"
          >
            {t("connector.connect.continue")}
          </button>
        </div>
      )}
    </div>
  )
}
