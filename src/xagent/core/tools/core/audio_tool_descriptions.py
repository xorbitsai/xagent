"""
Audio tool descriptions

This module contains the description templates for audio processing tools.
Extracted from audio_tool.py for better maintainability.
"""

# Description for transcribe_audio tool
TRANSCRIBE_AUDIO_DESCRIPTION = """
Transcribe audio to text using Speech-to-Text (ASR).

This tool converts spoken language in audio files into written text.
Supports multiple languages and can provide detailed timing information.

Available models (⭐[DEFAULT] marks the configured default model):
{}

**IMPORTANT: Prefer the default model marked with ⭐[DEFAULT]. Only specify model_id if the user explicitly requests a different model.**

Parameters:
- audio_file_path (required): audio file path, file_id, or URL to transcribe
- language (optional): language code (e.g., 'zh', 'en', 'yue', 'ja', 'ko')
- model_id (optional): specific ASR model to use. Omit to use the default model marked with ⭐[DEFAULT].
- verbose (optional): Set to True if you need segment details in the return value. Default: False

Language support:
- 'zh': Chinese (Mandarin)
- 'en': English
- 'yue': Cantonese
- 'ja': Japanese
- 'ko': Korean
- And more depending on model capabilities

Audio formats: wav, mp3, m4a, flac, ogg, and other common formats

Advanced features (if supported by model):
- Speaker diarization: identify different speakers
- Timestamps: get word-level or segment-level timing
- Confidence scores: get transcription confidence
- Smart segment merging: consecutive segments from same speaker are automatically merged (gap < 1s) to improve readability

Output:
- file_id: File ID for accessing the full transcription JSON file in workspace
- transcription_path: Path to saved transcription JSON file in workspace
- saved_to_workspace: Whether the transcription was saved to workspace
- segments: Detailed segment information (only present if verbose=True)
- language: Detected language code
- model_used: The actual model used for transcription
- text_length: Length of transcribed text
- segment_count: Number of segments

Note: Use read_file(file_id) to get the full transcription text.

JSON Output Format (saved to file specified by file_id):
```json
{{
  "model": "model_name",
  "language": "zh",
  "text": "Full transcribed text here...",
  "segments": [
    {{
      "text": "Segment text",
      "start": 0.0,
      "end": 2.5,
      "speaker": "spk1",
      "confidence": 0.95
    }}
  ],
  "metadata": {{
    "audio_source": "input_audio.mp3",
    "verbose_mode": true,
    "total_segments": 10
  }}
}}
```

Note: Segments are automatically merged when consecutive segments from
the same speaker are close together (< 1 second gap) to improve readability
and reduce fragmentation.
""".strip()

# Description for synthesize_speech tool
SYNTHESIZE_SPEECH_DESCRIPTION = """
Synthesize speech from text using Text-to-Speech (TTS).

This tool converts written text into natural-sounding speech audio.
Supports multiple voices, languages, and audio formats.

Available models (⭐[DEFAULT] marks the configured default model):
{}

**IMPORTANT: Prefer the default model marked with ⭐[DEFAULT]. Only specify model_id if the user explicitly requests a different model.**

Parameters:
- text (required): text content to synthesize into speech
- voice (optional): provider-specific voice identifier. Never invent or derive this value from language, gender, accent, or style. For ElevenLabs, use only an exact voice_id returned by list_tts_voices or clone_tts_voice. Omit for the configured default voice.
- language (optional): language code (e.g., 'zh', 'en', 'yue'). Auto-detected from text if not specified.
- audio_format (optional): audio output format (e.g., 'mp3', 'wav', 'pcm'). Default: 'mp3'
- sample_rate (optional): sample rate in Hz when the provider/model supports it.
- reference_audio (optional): reference audio file path or workspace file ID for voice cloning (if supported by model). When provided, it takes precedence over voice. ElevenLabs creates a temporary Instant Voice Clone, reuses it within the task, and deletes it during task cleanup.
- voice_settings (optional): provider-specific voice shaping object. Use only with models that advertise supports_voice_settings in list_audio_models.
- provider_options (optional): provider-specific synthesis options. Use list_audio_models to inspect supported_provider_options.
- model_id (optional): specific TTS model to use. Omit to use the default model marked with ⭐[DEFAULT].

Voice options depend on the model:
- Voice identifiers are provider-specific opaque values, not generic labels such as language plus gender.
- Some models support voice cloning using reference_audio
- Multilingual models can auto-detect language from text
- For providers that support dynamic voice lookup, call list_tts_voices first and pass an exact returned voice_id as voice.
- If the user requests voice attributes but provides no exact voice ID, inspect list_tts_voices metadata and select a matching returned voice_id. If no selection is needed, omit voice to use the configured default.

Provider-specific options:
- ElevenLabs supports voice_settings keys: stability, similarity_boost, style, speed, use_speaker_boost.
- ElevenLabs supports provider_options keys such as seed, previous_text, next_text, optimize_streaming_latency, apply_text_normalization, apply_language_text_normalization, pronunciation_dictionary_locators, and pronunciation_aliases.
- ElevenLabs pronunciation_aliases is a case-sensitive object that maps exact source words or phrases to text that produces the intended spoken pronunciation. It changes only the text sent to ElevenLabs; keep text and written artifacts correctly spelled. Prefer phrase-level aliases over replacing one ambiguous character globally. For example: {{"Claughton": "Cloffton", "UN": "United Nations"}}.
- Xinference currently does not expose dynamic voice listing through this tool. Some Xinference TTS models may still accept model-specific voice IDs or extra synthesis parameters.

Audio format options:
- mp3: Compressed audio, good for speech (default)
- wav: Uncompressed audio, higher quality
- pcm: Raw audio data

The generated audio file will be automatically saved to workspace.
""".strip()

# Description for synthesize_speech_json tool
SYNTHESIZE_SPEECH_JSON_DESCRIPTION = """
Batch synthesize speech from JSON structure using Text-to-Speech (TTS).

This tool converts multiple text segments into speech audio files in a single call.
Supports flexible JSON format with configurable field mapping, voice cloning, and batch processing.

Available models (⭐[DEFAULT] marks the configured default model):
{}

**IMPORTANT: Prefer the default model marked with ⭐[DEFAULT]. Only specify model_id if the user explicitly requests a different model.**

Parameters:
- json_data (optional): JSON string or dict containing synthesis configuration. Either json_data or file_id must be provided.
- file_id (optional): File ID, file path, or URL to read JSON data from. Either json_data or file_id must be provided.
- segments_field (optional): Field name containing segments array (default: "segments")
- text_field (optional): Field name containing text within each segment (default: "text")
- voice_field (optional): Field name containing voice within each segment (default: "voice")
- reference_field (optional): Field name containing reference audio file path/ID for voice cloning (default: "reference_audio")
- voice_settings_field (optional): Field name containing per-segment provider voice settings (default: "voice_settings")
- provider_options_field (optional): Field name containing per-segment provider options (default: "provider_options")
- default_voice (optional): Default voice for segments without voice specified
- default_language (optional): Default language code (auto-detect if None)
- default_voice_settings (optional): Provider voice settings applied to every segment unless overridden
- default_provider_options (optional): Provider options applied to every segment unless overridden
- audio_format (optional): Output audio format (default: 'mp3')
- sample_rate (optional): Sample rate in Hz (default: model-specific)
- model_id (optional): Specific TTS model to use. Omit to use the default model marked with ⭐[DEFAULT].
- batch_size (optional): Number of syntheses to process in parallel (1-20, default: 5)

JSON Format (nested segment structure):
```json
{{
    "segments": [
        {{"text": "你好世界", "reference_audio": "ref_voice_1"}},
        {{
            "text": "这是一个测试",
            "reference_audio": "ref_voice_2",
            "voice_settings": {{"stability": 0.45, "style": 0.2}}
        }}
    ],
    "default_provider_options": {{"seed": 1234}},
    "output_format": "mp3",
    "sample_rate": 24000
}}
```

Voice identifiers:
- Never invent voice or default_voice values from language, gender, accent, or style.
- For ElevenLabs, call list_tts_voices or clone_tts_voice first and use an exact returned voice_id. Omit voice/default_voice to use the configured default.

Voice Cloning:
- Use reference_audio in each segment to clone voices from reference audio files
- Supports both workspace file IDs and direct file paths (absolute or relative)
- ElevenLabs creates an Instant Voice Clone for each distinct reference audio file, reuses it within the task, and deletes it during task cleanup
- Voice cloning quality depends on the reference audio quality
- Not all models support voice cloning

Batch Processing:
- All segments are processed in parallel for efficiency
- Use batch_size to control parallelism (1-20)
- Progress is shown during synthesis
- Failed segments don't stop the batch

Output:
- success (bool): Whether all syntheses succeeded
- results (list): List of synthesis results, one per segment
- total (int): Total number of segments processed
- successful (int): Number of successful syntheses
- failed (int): Number of failed syntheses
- errors (list): List of error messages for failed segments
- saved_to_workspace (bool): Whether audio files were saved to workspace

Using file_id parameter is recommended for workflows with file chaining.
file_id supports: File ID, file path, or URL.
""".strip()

# Description for list_tts_voices tool
LIST_TTS_VOICES_DESCRIPTION = """
List available voices for a configured Text-to-Speech (TTS) model.

This tool calls the selected provider only when that provider supports dynamic voice listing.
Currently supported providers: {}.

Parameters:
- provider (optional): voice provider. Currently the only supported value is "elevenlabs".
- model_id (optional): specific TTS model to inspect. Omit to use the default TTS model.

Output:
- success (bool): Whether voice listing succeeded
- supported (bool): Whether the selected provider supports dynamic voice listing
- provider (str): Selected provider name
- model_used (str): Selected model ID or "default"
- voices (list): Voice metadata filtered to model IDs configured for the selected provider. Use voice_id from an item as the voice parameter in synthesize_speech or synthesize_speech_json.
- count (int): Number of voices returned
- supported_providers (list): Providers that currently support dynamic voice listing

Notes:
- ElevenLabs category identifies the voice type: "cloned" is an Instant Voice Clone and "professional" is a Professional Voice Clone; other documented values include "premade" and "generated". Voices may also include name, description, labels, preview_url, available_for_tiers, settings, and verified_languages.
- Providers without dynamic listing may still accept provider-specific voice IDs in synthesize_speech.
""".strip()

# Description for clone_tts_voice tool
CLONE_TTS_VOICE_DESCRIPTION = """
Create a persistent provider voice clone and return its voice ID.

Currently supported providers: {}.

The returned voice ID belongs to the provider account/API key selected by provider and model_id. It is not portable to another provider or account.

Parameters:
- name (required): Human-readable name for the cloned voice.
- reference_audio_files (required): One or more workspace file IDs or local audio file paths containing the same speaker.
- provider (optional): Voice cloning provider. Currently the only supported value is "elevenlabs".
- description (optional): Description stored with the provider voice.
- labels (optional): Provider metadata such as language, accent, gender, age, or use_case.
- remove_background_noise (optional): Ask the provider to isolate background noise before cloning. Default: false. Do not enable for already-clean recordings because it can reduce quality.
- model_id (optional): Provider TTS configuration whose API key and base URL should be used. Omit to select the first configured model for provider.

Output:
- voice_id: Persistent provider voice ID. Pass this value as voice to synthesize_speech or synthesize_speech_json.
- name: Stored voice name.
- provider: Provider that owns the returned voice ID.
- persistent: Always true for successful clones created by this tool.
- requires_verification: Whether the provider requires additional voice verification.

Important:
- Only clone a voice when the user has the necessary rights and consent.
- The voice remains in the selected ElevenLabs account after the task ends. It is not deleted during tool teardown.
- For ElevenLabs Instant Voice Cloning, prefer approximately 1-2 minutes of clear, consistent, single-speaker audio without background noise or reverb.
""".strip()

# Description for delete_tts_voice tool
DELETE_TTS_VOICE_DESCRIPTION = """
Permanently delete a voice from the configured provider account.

Currently supported providers: {}.

Parameters:
- voice_id (required): Exact provider voice ID returned by list_tts_voices or clone_tts_voice.
- provider (optional): Voice provider. Currently the only supported value is "elevenlabs".
- model_id (optional): Provider TTS configuration whose API key and base URL should be used. Omit to select the first configured model for provider.

Output:
- deleted: True when the provider accepted the deletion.
- voice_id: Deleted provider voice ID.
- provider: Provider account from which the voice was deleted.
- model_used: TTS configuration used for the provider request.

Important:
- This operation is irreversible. Only call it when the user explicitly asks to delete the voice.
- The voice ID belongs to a specific provider account. Select the same provider configuration that created or listed it.
""".strip()
