const MESSAGE_TARGET = "xagent-media-offscreen"
const MAX_MEDIA_BYTES = 32 * 1024 * 1024

interface CaptureMediaMessage {
  target: typeof MESSAGE_TARGET
  type: "capture_media"
  streamId: string
  mediaKind: "audio" | "video"
  durationMs: number
}

interface CaptureMediaArtifact {
  data_base64: string
  mime_type: string
  media_kind: "audio" | "video"
  duration_ms: number
}

chrome.runtime.onMessage.addListener(
  (rawMessage: unknown, _sender, sendResponse) => {
    if (!isCaptureMediaMessage(rawMessage)) return false
    void captureMedia(rawMessage)
      .then((artifact) => sendResponse({ success: true, artifact }))
      .catch((error: unknown) =>
        sendResponse({ success: false, error: errorMessage(error) }),
      )
    return true
  },
)

async function captureMedia(
  message: CaptureMediaMessage,
): Promise<CaptureMediaArtifact> {
  const durationMs = Math.min(30_000, Math.max(1_000, message.durationMs))
  const mandatoryAudio = {
    chromeMediaSource: "tab",
    chromeMediaSourceId: message.streamId,
  }
  const mandatoryVideo = {
    ...mandatoryAudio,
    maxWidth: 1920,
    maxHeight: 1080,
    maxFrameRate: 30,
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: mandatoryAudio } as unknown as MediaTrackConstraints,
    video:
      message.mediaKind === "video"
        ? ({ mandatory: mandatoryVideo } as unknown as MediaTrackConstraints)
        : false,
  })
  let audioContext: AudioContext | null = null
  try {
    if (stream.getAudioTracks().length === 0) {
      throw new Error("The approved tab did not provide an audio track.")
    }
    // Capturing a tab otherwise mutes its local playback. Route the captured
    // audio back to the user's speakers while recording.
    audioContext = new AudioContext()
    if (audioContext.state === "suspended") await audioContext.resume()
    audioContext.createMediaStreamSource(stream).connect(audioContext.destination)

    const mimeType = preferredMimeType(message.mediaKind)
    const recorder = new MediaRecorder(stream, {
      mimeType,
      audioBitsPerSecond: 96_000,
      ...(message.mediaKind === "video"
        ? { videoBitsPerSecond: 1_500_000 }
        : {}),
    })
    const chunks: Blob[] = []
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data)
    })
    const stopped = new Promise<void>((resolve, reject) => {
      recorder.addEventListener("stop", () => resolve(), { once: true })
      recorder.addEventListener(
        "error",
        (event) => reject(event.error),
        { once: true },
      )
    })
    recorder.start(1_000)
    await delay(durationMs)
    recorder.stop()
    await stopped

    const blob = new Blob(chunks, { type: recorder.mimeType || mimeType })
    if (blob.size === 0) {
      throw new Error("The approved tab produced no media data.")
    }
    if (blob.size > MAX_MEDIA_BYTES) {
      throw new Error(
        "The captured media exceeds 32 MiB. Capture a shorter segment.",
      )
    }
    return {
      data_base64: arrayBufferToBase64(await blob.arrayBuffer()),
      mime_type: blob.type,
      media_kind: message.mediaKind,
      duration_ms: durationMs,
    }
  } finally {
    for (const track of stream.getTracks()) track.stop()
    await audioContext?.close()
  }
}

function preferredMimeType(mediaKind: "audio" | "video"): string {
  const candidates =
    mediaKind === "audio"
      ? ["audio/webm;codecs=opus", "audio/webm"]
      : [
          "video/webm;codecs=vp9,opus",
          "video/webm;codecs=vp8,opus",
          "video/webm",
        ]
  const supported = candidates.find((candidate) =>
    MediaRecorder.isTypeSupported(candidate),
  )
  if (!supported) {
    throw new Error(`Chrome cannot encode ${mediaKind} as WebM.`)
  }
  return supported
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  const parts: string[] = []
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    parts.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)),
    )
  }
  return btoa(parts.join(""))
}

function isCaptureMediaMessage(
  value: unknown,
): value is CaptureMediaMessage {
  if (typeof value !== "object" || value === null) return false
  const message = value as Partial<CaptureMediaMessage>
  return (
    message.target === MESSAGE_TARGET &&
    message.type === "capture_media" &&
    typeof message.streamId === "string" &&
    (message.mediaKind === "audio" || message.mediaKind === "video") &&
    typeof message.durationMs === "number"
  )
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, durationMs))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
