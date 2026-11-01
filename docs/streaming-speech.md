# Streaming speech transports

These authenticated endpoints connect saved speech models to browser clients.
They provide live transcription and streamed synthesis; they do not start an
agent run, choose an agent turn boundary, or provide a voice conversation UI.

Both endpoints require an explicit, active model ID accessible to the logged-in
user. Ownership, shared-model visibility, and application model visibility hooks
apply. Provider credentials come from the saved configuration and stay on the
server. Streaming support is checked on the selected adapter.

Live ASR currently uses ElevenLabs `scribe_v2_realtime`. Save it as a custom
speech model with the `asr` ability. It is intentionally excluded from file-ASR
defaults and the existing microphone upload controls. Streaming TTS uses a saved
ElevenLabs speech model with the `tts` ability; omitted settings retain that
model's configured language, voice, format, and sample rate.

## Live transcription

Connect to `WS /api/models/speech/asr/stream` using the existing login token
carriers. Browsers offer the fixed protocol and credential protocol; only the
fixed protocol is selected by the server:

```javascript
const ws = new WebSocket(`${wsBase}/api/models/speech/asr/stream`, [
  "xagent-auth-v1",
  `xagent-auth-token.${accessToken}`,
]);
ws.onopen = () => ws.send(JSON.stringify({
  type: "start",
  model_id: "my-live-asr",
  language: "en",
  format: "pcm",
  sample_rate: 16000,
  vad: true,
}));
ws.onmessage = ({ data }) => console.log(JSON.parse(data));
```

Non-browser clients may instead send `Authorization: Bearer <access token>`.
Credentials in the URL query string are rejected. Authentication and durable
session revocation use the same owner as private chat WebSockets.

The first frame must be a `start` JSON object. `language` is optional; the
other defaults are `format: "pcm"`, `sample_rate: 16000`, and `vad: true`.
Optional `vad_silence_threshold_secs` (0.3–3) and `vad_threshold` (0.1–0.9)
are accepted only in VAD mode.

After the `session_started` event, send binary audio fragments with
`ws.send(arrayBuffer)`. PCM is raw signed 16-bit little-endian **mono**, at
8000, 16000, 22050, 24000, 44100, or 48000 Hz. `ulaw` requires 8000 Hz.
This endpoint does not accept WAV headers, WebM recordings, or stereo samples.
Each audio fragment is at most 64 KiB and each JSON control frame is at most
4 KiB. PCM fragments must contain an even number of bytes (complete 16-bit
samples); odd lengths close with 1008 before reaching the provider. Send
fragments as they are recorded; writes apply provider backpressure.

The server emits `session_started`, `partial`, and `final` JSON events with
`text`, `segment_id`, and `session_id`. A `final` commits a transcript segment,
not an agent turn. Server VAD commits automatically. With `vad: false`, send
`{"type":"commit"}` to request a commit. Sending that command in VAD mode is
an invalid request.

In manual mode, wait for the corresponding `final` after `commit` before sending
`close`. `{"type":"close"}` is an immediate abort: it cancels the event relay
and discards pending audio/events, including a committed segment whose `final`
has not arrived. Disconnecting has the same cancellation semantics. Neither
implicitly commits or drains pending finals.

Errors have the shape
`{"type":"error","code":"invalid_request","message":"Invalid speech stream request"}`.

| Code | Meaning | WebSocket close code |
| --- | --- | --- |
| `invalid_request` | Invalid start, command, frame or audio duration | 1008 |
| `model_unavailable` | Selected model is unavailable or incompatible | 1008 |
| `session_timeout` | Local receive/send idle limit or session deadline | 1000 |
| `stream_failed` | Provider failure, including an upstream timeout | 1011 |

Authentication failure during the handshake is rejected with HTTP 403.
Session revocation after acceptance closes with 4001 and may have no `error`
event. Query-string credentials are refused before the upgrade, as
described in [WebSocket authentication](websocket-header-auth.md).
Connections and total submitted audio are limited to ten minutes; no inbound
frame for 30 seconds also ends the session. Reconnect for a new session.
Terminal error/close delivery is bounded and best effort. When an outbound
transcript stalls, the server attempts to close without another error data frame.

## Streaming synthesis

`POST /api/models/speech/tts/stream` accepts JSON and the normal Bearer login:

```bash
curl --no-buffer "$API_BASE/api/models/speech/tts/stream" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"my-tts","text":"Hello there.","format":"pcm","sample_rate":24000}' \
  --output speech.pcm
```

`text` is required (1–5000 characters). Optional overrides are `voice`,
`language`, `format`, and `sample_rate`. Voice IDs must be values returned by
the provider's voice listing or cloning APIs, using 1–128 ASCII letters,
digits, underscores or hyphens. This route does not accept
reference file paths or arbitrary provider options.

The body contains audio bytes as they arrive, with no JSON or base64 envelope.
`Content-Type` identifies MP3 (`audio/mpeg`), WAV (`audio/wav`), or raw PCM
(`audio/pcm`); other codecs use `application/octet-stream`.
`X-Audio-Format` and `X-Audio-Sample-Rate` describe the actual first audio
fragment, and are exposed to browser fetch clients. HTTP chunk boundaries are
arbitrary byte fragments, not independently playable audio files. Raw PCM
requires a client PCM decoder/player; `fetch().body` can consume the stream
without waiting for the complete recording.

The route waits for first audio before starting a successful response.
Malformed request bodies (including invalid voice IDs) return 422. Unsupported
output formats/sample rates or incompatible synthesis options return 400 before
opening the provider request. A first-audio timeout returns 504, and a provider
failure before first audio returns 502. After audio has begun, failures and
timeouts interrupt the stream; clients must handle a truncated response.
Aborting the fetch closes the provider iterator and client, including while
waiting for first audio. Individual upstream waits are bounded to 30 seconds
and the complete response to ten minutes. Responses disable caching and
request proxy buffering to be disabled.

## Usage and lifecycle

ASR records submitted audio duration when its session closes, inside its opening
usage scope. TTS retains the provider adapter's accepted-text metering, including
interrupted responses. Both deliver through the existing standalone usage hook
for the requesting user, once per connection/request (`speech_transcribe` or
`speech_synthesize`). The transport does not record a second provider charge.
Database connections are released before audio streaming.

The limits above apply per connection/request. This transport currently has no
per-user or per-model concurrency cap: parallel streams, including those using
a shared model's provider credential, are permitted. Usage accounting does not
provide concurrency admission. Deployments that require such caps must enforce
them across API replicas at their admission layer.
