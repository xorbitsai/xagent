# Voice mode in task chat

Voice mode uses the authenticated [streaming speech transports](streaming-speech.md)
for a continuous conversation in private task chat. Save an ElevenLabs
`scribe_v2_realtime` model with the `asr` ability and an ElevenLabs speech model
with the `tts` ability before starting.

Open a private task conversation and choose **Voice mode** beside the composer.
Select a listening model and a speaking model, then **Start conversation**.
Microphone permission is requested only when starting. The call shows live
transcription and Listening, Thinking, or Speaking status; microphone mute and
end-call are the only in-call controls. Closing the dialog, leaving the task,
logging out, or losing the chat connection ends the call and releases resources.
Public widget/share conversations do not enable this private voice entry.

Speech is submitted automatically after VAD finals and a short quiet interval;
adjacent segments are combined and duplicate finals are ignored. Text enters
the ordinary task message-delivery path, including waiting-question IDs and
durable client message IDs. Attachments in the text composer are not sent by
the voice call.

The microphone remains active while the agent speaks. Browser echo
cancellation and local Silero v5 VAD confirm human speech before stopping
playback; an ASR partial confirms the words before requesting interruption of an
active agent run through the existing pause path. The next utterance waits for
that run to settle before delivery, then follows the normal continuation path.
Interruption is automatic and speaking requires no extra action.
Interrupted audio fetches and queued speech are discarded, and stale reply
callbacks cannot restart playback. Completed sentences from a streaming final
answer are synthesized as they arrive, while historical answers, internal
traces, code blocks, failed answers, and duplicate reply frames are not read.

Capture uses an AudioWorklet and resamples the actual device rate to mono
PCM16 at 16 kHz. Playback consumes 24 kHz PCM incrementally, retaining split
samples across HTTP chunks and bounding scheduled audio ahead of playback.
Silero processes 32 ms frames with a 0.6 speech probability threshold and
160 ms minimum speech. Only its confirmed speech event interrupts playback;
audio levels, isolated candidate frames, and ASR events without VAD confirmation
do not. The ONNX model and WASM runtime are served by the application itself,
prepared from pinned npm packages by `predev` and `prebuild`. They are loaded
only when starting a call. No browser audio is sent to a VAD service.
