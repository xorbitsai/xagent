# Voice mode in task chat

Voice mode uses the authenticated [streaming speech transports](streaming-speech.md)
for a continuous conversation in private task chat. Save an ElevenLabs
`scribe_v2_realtime` model with the `asr` ability and an ElevenLabs speech model
with the `tts` ability before starting.

The entry remains visible but disabled while models are being checked or a
required model is missing. Hover or focus the disabled entry to see whether it
needs realtime ASR with server VAD, streaming TTS, or a selected chat model/agent.
Inactive models and file-only ASR do not qualify. Returning to the app refreshes
availability; a failed model-list request is reported separately from missing
configuration. Public widget/share capabilities remain opt-in.

Choose the headphone **Voice mode** button next to the microphone on the task
home page or in an existing private task's composer.
Choose your listening model, speaking model and voice in **Settings → Voice**.
These preferences are saved to your account, separately from the agent's writing
style. Automatic uses the first available compatible model; the model's default
voice is used unless you select a named voice from its provider. Voice lists are
fetched by the backend using the accessible model, never a browser API key.
An unavailable saved model requires choosing a replacement in Settings.
The conversation dialog uses these preferences; press **Start conversation**.
Microphone permission is requested only when starting. The call shows live
transcription and Listening, Thinking, or Speaking status; microphone mute and
end-call are the only in-call controls. Closing the dialog, leaving the task,
logging out, or losing the chat connection ends the call and releases resources.
Public widget/share conversations do not enable this private voice entry.
Each ASR session lasts at most ten minutes, including muted time; an idle
transport also times out after 30 seconds. The dialog identifies session expiry.
Use **Start conversation** to open a fresh session and continue.

On the home page, the first utterance creates the task using the composer's
selected model, mode, runtime and agent. Opening voice mode sends no message and
does not clear the text draft or attach its files. The call remains active while
the created task opens, so its first reply can be spoken without restarting the
microphone. Navigating to another task/draft, leaving task pages or changing the
login cancels the call and invalidates any pending creation response.

Speech is submitted automatically after VAD finals and a short quiet interval;
adjacent segments are combined and duplicate finals are ignored. Text enters
the ordinary task message-delivery path, including waiting-question IDs and
durable client message IDs. Attachments in the text composer are not sent by
the voice call.

The microphone remains active while the agent speaks. Backend realtime ASR
provides server VAD and final transcript boundaries. Browser echo cancellation
and optional local Silero v5 VAD track speech activity. A non-empty backend
transcript confirms speech before playback is stopped, including when local
VAD misses quiet speech. VAD activity without words leaves the reply audible. An ASR
partial confirms the words before requesting interruption of an active agent run
through the existing pause path. The next utterance waits for
that run to settle before delivery, then follows the normal continuation path.
Interruption is automatic and speaking requires no extra action.
Interrupted audio fetches and queued speech are discarded, and stale reply
callbacks cannot restart playback. Completed sentences from a streaming final
answer are synthesized as they arrive, while historical answers, internal
traces, code blocks, failed answers, and duplicate reply frames are not read.
A failed synthesis sentence is skipped with a visible warning while listening
continues; it is not automatically retried. Authentication or ASR session loss
ends the call with a specific message.

Capture uses an AudioWorklet and resamples the actual device rate to mono
PCM16 at 16 kHz. Playback consumes 24 kHz PCM incrementally, retaining split
samples across HTTP chunks and bounding scheduled audio ahead of playback.
Silero processes 32 ms frames with a 0.6 speech probability threshold and
160 ms minimum speech. Local activity refines turn boundaries; interruption
requires ASR text confirmation. Local VAD loads in
the background and never blocks connection/capture. If loading or inference
fails, the call continues using backend ASR/VAD and shows that mode. The ONNX
model and WASM runtime are served by the application itself,
prepared from pinned npm packages by `predev` and `prebuild`. They are loaded
only when starting a call. Local detection sends no additional browser audio to
a VAD service.

API/SDK clients use the same backend protocol: open realtime ASR with `vad: true`,
consume `partial`/`final` events, and stream TTS responses. They do not require
browser Silero, ONNX or WASM. Server VAD is supplied by the ASR adapter here; it
is not a separately configured VAD model.
