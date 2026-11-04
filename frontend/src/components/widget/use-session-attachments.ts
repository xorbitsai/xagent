"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { ComposerAttachmentStatus } from "@/components/chat/ChatInput"
import { readSendErrorCode } from "@/components/chat/clarification-delivery"
import { toast } from "@/components/ui/sonner"
import type { SessionAttachmentsConfig, useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import {
  clientErrorTranslationKey,
  type ClientErrorCode,
} from "@/lib/client-errors"
import {
  capabilityRefusal,
  uploadWidgetSessionFile,
  type SessionUploadedFile,
} from "@/lib/widget-session-upload"
import type {
  WidgetSession,
  WidgetSessionFileUploads,
  WidgetSessionStatus,
} from "./use-widget-session"

// What the page needs from the bridge to upload files before a send.
export interface SessionUploadControls {
  ensureFreshSession: () => Promise<WidgetSession>
  refreshSession: (rejectedToken: string) => Promise<WidgetSession>
  waitForOpenConnection: () => Promise<void>
}

type AppControls = ReturnType<typeof useApp>

// Whether sendMessage itself refuses a turn in each conversation phase; an
// upload started before such a turn would be wasted. Every phase is listed, so
// a phase the provider adds or renames fails to compile until it is decided
// here.
const SEND_BLOCKED_IN_PHASE: Readonly<Record<AppControls["sessionConversationState"], boolean>> = {
  unbound: false,
  bound: false,
  reset_requested: true,
  replacement_ready: false,
  replacement_sending: true,
  replacement_awaiting_task: true,
  reload_required: true,
}

const isAbortError = (error: unknown): boolean =>
  typeof error === "object" && error !== null
  && (error as { name?: unknown }).name === "AbortError"

// The error that ends a send whose file the visitor removed. The visitor did it
// on purpose, so the composer must not announce a failure.
const removedFileError = (): Error =>
  Object.assign(
    new DOMException("A file was removed before the message was sent.", "AbortError"),
    { notificationHandled: true },
  )

// Resolves with `promise`, or rejects as soon as `signal` aborts. Only this
// caller's wait ends; `promise` itself keeps running.
const raceSend = <T,>(promise: Promise<T>, signal: AbortSignal): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(removedFileError())
    if (signal.aborted) {
      onAbort()
      return
    }
    signal.addEventListener("abort", onAbort, { once: true })
    promise.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value) },
      (error) => { signal.removeEventListener("abort", onAbort); reject(error) },
    )
  })

const userFacingError = (message: string, errorCode?: ClientErrorCode): Error =>
  Object.assign(new Error(message), {
    userFacing: true,
    ...(errorCode ? { errorCode } : {}),
  })

const isUserFacingError = (error: unknown): error is Error =>
  error instanceof Error && (error as { userFacing?: unknown }).userFacing === true

// A send with files that is still running. Removing any of its files aborts
// `controller`, which ends the send and the upload it is waiting on.
interface AttachmentSend {
  files: readonly File[]
  controller: AbortController
}

export const toAttachments = (
  fileUploads: WidgetSessionFileUploads | undefined,
): SessionAttachmentsConfig | undefined => fileUploads
  ? {
    accept: fileUploads.allowedExtensions.join(","),
    maxFiles: fileUploads.maxFilesPerMessage,
    maxFileBytes: fileUploads.maxFileBytes,
  }
  : undefined

export interface SessionAttachmentsOptions {
  // Present only while the session server advertises file uploads.
  attachments: SessionAttachmentsConfig | undefined
  uploads: SessionUploadControls
  status: WidgetSessionStatus
  // Whether a conversation is established, so the panel is the composer.
  conversationEstablished: boolean
  sendMessage: AppControls["sendMessage"]
  sessionConversationState: AppControls["sessionConversationState"]
  isSessionInteractionLocked: boolean
  isConversationResetPending: boolean
}

// The files a visitor selects in the session chat, their uploads and the sends
// that carry them. The page calls this once and hands the result to the start
// page and the conversation panel; both surfaces share the upload state, and
// each keeps its own selection here, so neither is lost when a surface
// unmounts.
export function useSessionAttachments({
  attachments,
  uploads,
  status,
  conversationEstablished,
  sendMessage,
  sessionConversationState,
  isSessionInteractionLocked,
  isConversationResetPending,
}: SessionAttachmentsOptions) {
  const { t } = useI18n()
  const [startFiles, setStartFiles] = useState<File[]>([])
  const [attachmentStatuses, setAttachmentStatuses] =
    useState<ReadonlyMap<File, ComposerAttachmentStatus>>(() => new Map())
  // Counts attachment work: each upload, and each send with files from its
  // first upload until its chat frame has gone out and been accepted. While it
  // is above zero the start page and "new conversation" are locked and every
  // file retry is disabled; the conversation panel's composer is locked only
  // while the task is pending or there is no task.
  const [attachmentWork, setAttachmentWork] = useState(0)
  const isAttachmentBusy = attachmentWork > 0
  // Keyed by the File the visitor picked, so a message sent again with the
  // same files (edited or not) reuses what already reached the server.
  const uploadedFilesRef = useRef(new WeakMap<File, SessionUploadedFile>())
  const uploadControllersRef = useRef(new Map<File, AbortController>())
  // One entry per send with files that is still running; removing any of its
  // files aborts that send and the upload it is waiting on.
  const activeSendsRef = useRef(new Set<AttachmentSend>())
  // The conversation panel's files live here rather than in the panel, which
  // is unmounted while the session is unavailable, so they are still selected
  // when it comes back, as the start page's files are.
  const [panelFiles, setPanelFiles] = useState<File[]>([])
  const panelFilesRef = useRef<File[]>([])
  // The capability of the latest accepted session. A send reads it right
  // before its chat frame goes out, because the `attachments` it captured
  // when it started may belong to a session replaced since.
  const latestAttachmentsRef = useRef(attachments)
  latestAttachmentsRef.current = attachments

  const setAttachmentStatus = useCallback((
    file: File,
    next: ComposerAttachmentStatus | null,
  ) => {
    setAttachmentStatuses((current) => {
      const updated = new Map(current)
      if (next === null) updated.delete(file)
      else updated.set(file, next)
      return updated
    })
  }, [])

  // Forgets everything held for a file the visitor no longer wants: cancels
  // its upload, including one still waiting for a token, ends every send that
  // includes it together with the upload that send is waiting on, and drops
  // what was already uploaded. A request that had reached the server may
  // still leave the file stored there; it is never attached to a message.
  const releaseAttachment = useCallback((file: File) => {
    uploadControllersRef.current.get(file)?.abort()
    uploadControllersRef.current.delete(file)
    activeSendsRef.current.forEach((send) => {
      if (send.files.includes(file)) send.controller.abort()
    })
    uploadedFilesRef.current.delete(file)
    setAttachmentStatus(file, null)
  }, [setAttachmentStatus])

  const releaseRemovedAttachments = useCallback((previous: File[], next: File[]) => {
    const kept = new Set(next)
    previous.filter((file) => !kept.has(file)).forEach(releaseAttachment)
  }, [releaseAttachment])

  // Cancels every upload and send. A file still selected goes back to not
  // uploaded: the registry is cleared here, so the upload's own abort handling
  // cannot tell the file is still selected, and the card would otherwise stay
  // at "Uploading…" if the session comes back.
  const abortAllUploads = useCallback(() => {
    uploadControllersRef.current.forEach((controller, file) => {
      controller.abort()
      setAttachmentStatus(file, null)
    })
    uploadControllersRef.current.clear()
    activeSendsRef.current.forEach((send) => send.controller.abort())
  }, [setAttachmentStatus])

  useEffect(() => abortAllUploads, [abortAllUploads])
  useEffect(() => {
    if (status === "terminal" || status === "degraded") abortAllUploads()
  }, [abortAllUploads, status])

  const handleStartFilesChange = useCallback((next: File[]) => {
    releaseRemovedAttachments(startFiles, next)
    setStartFiles(next)
  }, [releaseRemovedAttachments, startFiles])

  const handlePanelFilesChange = useCallback((next: File[]) => {
    releaseRemovedAttachments(panelFilesRef.current, next)
    panelFilesRef.current = next
    setPanelFiles(next)
  }, [releaseRemovedAttachments])

  // The panel's files belong to the conversation: they are released once it
  // returns to the start page.
  useEffect(() => {
    if (!conversationEstablished && panelFilesRef.current.length > 0) {
      handlePanelFilesChange([])
    }
  }, [conversationEstablished, handlePanelFilesChange])

  // Checks files against a capability the session server advertised:
  // extension, per-file size and per-message count. With no capability
  // advertised, no file may be uploaded or sent.
  const assertCapabilityAccepts = useCallback((
    files: readonly File[],
    limits: SessionAttachmentsConfig | undefined,
  ) => {
    if (!limits) throw userFacingError(t("widgetSession.attachments.notAvailable"))
    for (const file of files) {
      const refusal = capabilityRefusal(file, limits)
      if (refusal === "type") {
        throw userFacingError(t("widgetSession.attachments.typeNotSupported"))
      }
      if (refusal === "size") {
        throw userFacingError(t(clientErrorTranslationKey("upload_too_large")), "upload_too_large")
      }
    }
    if (files.length > limits.maxFiles) {
      throw userFacingError(
        t("widgetSession.attachments.tooManyFiles", { count: limits.maxFiles }),
      )
    }
  }, [t])

  // Uploads one file with the session token. A send passes itself as `send`:
  // aborting the send aborts this upload, and every token the upload uses
  // must come from a session whose capability still accepts all of the
  // send's files. A retry passes no `send`, so only removing its own file,
  // the session becoming unavailable or ending, or the page unmounting stops
  // it, and its tokens are checked against that one file. The abort
  // controller is registered before the first token wait, so every wait of
  // the upload can be cancelled.
  // Uploading counts as attachment work from the call, token wait included,
  // and the file cards' retry buttons are disabled while any upload or send
  // with files is in flight. The conversation panel's composer stays open
  // while the task is running, paused, waiting for an answer, completed or
  // failed, but it refuses to submit while a file shows as uploading.
  const uploadAttachment = useCallback(async (
    file: File,
    send?: AttachmentSend,
  ): Promise<SessionUploadedFile> => {
    const controller = new AbortController()
    uploadControllersRef.current.set(file, controller)
    const abortWithSend = () => controller.abort()
    send?.controller.signal.addEventListener("abort", abortWithSend, { once: true })
    if (send?.controller.signal.aborted) controller.abort()
    setAttachmentWork((count) => count + 1)
    setAttachmentStatus(file, "uploading")
    const checkedFiles: readonly File[] = send ? send.files : [file]
    // The upload helper reports any failure of the token callbacks as
    // upload_failed, so a refusal is kept here and rethrown as it was raised.
    const refusal: { error?: Error } = {}
    const acceptedToken = (session: WidgetSession): string => {
      try {
        assertCapabilityAccepts(checkedFiles, toAttachments(session.capabilities?.fileUploads))
      } catch (error) {
        refusal.error = error as Error
        throw error
      }
      return session.token
    }
    try {
      const uploaded = await uploadWidgetSessionFile({
        origin: window.location.origin,
        getToken: async () => acceptedToken(await uploads.ensureFreshSession()),
        refreshToken: async (rejected) => acceptedToken(await uploads.refreshSession(rejected)),
        file,
        signal: controller.signal,
      })
      uploadedFilesRef.current.set(file, uploaded)
      setAttachmentStatus(file, "uploaded")
      return uploaded
    } catch (error) {
      if (controller.signal.aborted) {
        // A removed file has no card left to mark. A file still selected,
        // whose upload ended with its send, goes back to not uploaded so it
        // can be sent again. Neither needs a "not sent" notice.
        if (uploadControllersRef.current.get(file) === controller) {
          setAttachmentStatus(file, null)
        }
        throw typeof error === "object" && error !== null
          ? Object.assign(error, { notificationHandled: true })
          : error
      }
      if (refusal.error) {
        // Nothing of this file was uploaded; it stays selected, so the visitor
        // can remove it or send again.
        setAttachmentStatus(file, null)
        throw refusal.error
      }
      setAttachmentStatus(file, "failed")
      throw error
    } finally {
      send?.controller.signal.removeEventListener("abort", abortWithSend)
      if (uploadControllersRef.current.get(file) === controller) {
        uploadControllersRef.current.delete(file)
      }
      setAttachmentWork((count) => count - 1)
    }
  }, [assertCapabilityAccepts, setAttachmentStatus, uploads])

  // A retry uploads only this file and belongs to no send. Its upload counts
  // as attachment work from the click, token wait included, so the other
  // cards' retry buttons stay disabled until it settles.
  const handleRetryAttachment = useCallback(async (file: File) => {
    try {
      await uploadAttachment(file)
    } catch (error) {
      if (isAbortError(error)) return
      const code = readSendErrorCode(error)
      toast.error(code
        ? t(clientErrorTranslationKey(code))
        : isUserFacingError(error)
          ? error.message
          : t(clientErrorTranslationKey("upload_failed")))
    }
  }, [t, uploadAttachment])

  // The checks that cost nothing: a send that is certain to be refused is
  // refused before any file leaves the browser.
  const assertAttachmentsSendable = useCallback((
    files: File[],
    config: Record<string, unknown> | undefined,
    limits: SessionAttachmentsConfig,
  ) => {
    const metadata = config?.metadata as { request_id?: unknown } | undefined
    if (typeof metadata?.request_id === "string" && metadata.request_id) {
      throw userFacingError(t("widgetSession.attachments.noFilesOnAnswer"))
    }
    assertCapabilityAccepts(files, limits)
  }, [assertCapabilityAccepts, t])

  // Sends a message with the files the visitor attached. Files go to the
  // session first, one after another, and the chat frame then names them by
  // id. The session can be replaced while this runs, so each upload checks
  // the capability of the session its token comes from, and the files are
  // checked once more against the latest capability right before the frame
  // goes out. A failed check fails the send; the text never goes out alone.
  const sendWithAttachments = useCallback(async (
    message: string,
    config?: Record<string, unknown>,
    files: File[] = [],
  ) => {
    if (files.length === 0 || !attachments) {
      await sendMessage(message, config, files)
      return
    }
    if (
      SEND_BLOCKED_IN_PHASE[sessionConversationState]
      || isSessionInteractionLocked
      || isConversationResetPending
    ) {
      // sendMessage would refuse the turn in these states, but calling it
      // with the files dropped could send the text alone if its own view of
      // the state is newer than this render's, so the send stops here.
      throw new Error("Message not sent: the conversation is not ready for attachments.")
    }
    assertAttachmentsSendable(files, config, attachments)

    setAttachmentWork((count) => count + 1)
    const send: AttachmentSend = { files, controller: new AbortController() }
    activeSendsRef.current.add(send)
    try {
      const uploaded: SessionUploadedFile[] = []
      for (const file of files) {
        if (send.controller.signal.aborted) throw removedFileError()
        const known = uploadedFilesRef.current.get(file)
        uploaded.push(known ?? await uploadAttachment(file, send))
      }
      await raceSend(uploads.waitForOpenConnection(), send.controller.signal)
      // Removing a file aborts the send above; this check is the backstop for
      // a file that was released without reaching the abort.
      if (files.some((file) => !uploadedFilesRef.current.has(file))) {
        throw removedFileError()
      }
      // The open socket belongs to the latest accepted session, whose
      // capability this ref holds.
      assertCapabilityAccepts(files, latestAttachmentsRef.current)

      try {
        await sendMessage(message, config, [], { uploadedFiles: uploaded })
      } catch (error) {
        if (readSendErrorCode(error) === "message_attachment_unavailable") {
          // The server no longer has one of these files: upload them all again.
          files.forEach((file) => {
            uploadedFilesRef.current.delete(file)
            setAttachmentStatus(file, null)
          })
        }
        throw error
      }
      files.forEach((file) => {
        uploadedFilesRef.current.delete(file)
        setAttachmentStatus(file, null)
      })
    } finally {
      activeSendsRef.current.delete(send)
      setAttachmentWork((count) => count - 1)
    }
  }, [
    assertAttachmentsSendable,
    assertCapabilityAccepts,
    attachments,
    isConversationResetPending,
    isSessionInteractionLocked,
    sendMessage,
    sessionConversationState,
    setAttachmentStatus,
    uploadAttachment,
    uploads,
  ])

  // A send from the start page. Files it did not carry stay selected,
  // including files hidden while no capability is advertised.
  const sendStartMessage = useCallback(async (
    message: string,
    files: File[],
    config?: Record<string, unknown>,
  ) => {
    await sendWithAttachments(message, config, files)
    setStartFiles((current) => current.filter((file) => !files.includes(file)))
  }, [sendWithAttachments])

  return {
    startFiles,
    onStartFilesChange: handleStartFilesChange,
    sendStartMessage,
    panelFiles,
    onPanelFilesChange: handlePanelFilesChange,
    sendWithAttachments,
    statuses: attachmentStatuses,
    isBusy: isAttachmentBusy,
    retryAttachment: handleRetryAttachment,
  }
}
