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
- voice (optional): voice ID or name (e.g., 'zh-android', 'zh-female', 'en-male'). Omit for default voice.
- language (optional): language code (e.g., 'zh', 'en', 'yue'). Auto-detected from text if not specified.
- audio_format (optional): audio output format (e.g., 'mp3', 'wav', 'pcm'). Default: 'mp3'
- sample_rate (optional): sample rate in Hz when the provider/model supports it.
- reference_audio (optional): reference audio file path for voice cloning (if supported by model)
- voice_settings (optional): provider-specific voice shaping object. Use only with models that advertise supports_voice_settings in list_audio_models.
- provider_options (optional): provider-specific synthesis options. Use list_audio_models to inspect supported_provider_options.
- model_id (optional): specific TTS model to use. Omit to use the default model marked with ⭐[DEFAULT].

Voice options depend on the model:
- Most models support standard voices: male, female, neutral
- Some models support voice cloning using reference_audio
- Multilingual models can auto-detect language from text
- For providers that support dynamic voice lookup, call list_tts_voices first and pass the returned voice_id as voice.

Provider-specific options:
- ElevenLabs supports voice_settings keys: stability, similarity_boost, style, speed, use_speaker_boost.
- ElevenLabs supports provider_options keys such as seed, previous_text, next_text, optimize_streaming_latency, apply_text_normalization, apply_language_text_normalization, and pronunciation_dictionary_locators.
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
        {{"text": "你好世界", "voice": "zh-female", "reference_audio": "ref_voice_1"}},
        {{
            "text": "这是一个测试",
            "voice": "zh-male",
            "reference_audio": "ref_voice_2",
            "voice_settings": {{"stability": 0.45, "style": 0.2}}
        }}
    ],
    "default_voice": "zh-female",
    "default_provider_options": {{"seed": 1234}},
    "output_format": "mp3",
    "sample_rate": 24000
}}
```

Voice Cloning:
- Use reference_audio in each segment to clone voices from reference audio files
- Supports both workspace file IDs and direct file paths (absolute or relative)
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
- model_id (optional): specific TTS model to inspect. Omit to use the default TTS model.

Output:
- success (bool): Whether voice listing succeeded
- supported (bool): Whether the selected provider supports dynamic voice listing
- provider (str): Selected provider name
- model_used (str): Selected model ID or "default"
- voices (list): Voice metadata. Use voice_id from an item as the voice parameter in synthesize_speech or synthesize_speech_json.
- count (int): Number of voices returned
- supported_providers (list): Providers that currently support dynamic voice listing

Notes:
- ElevenLabs voices may include name, category, description, labels, preview_url, available_for_tiers, settings, and verified_languages.
- Providers without dynamic listing may still accept provider-specific voice IDs in synthesize_speech.
""".strip()
