# AGENTS.md: Pipeline Stages as Modular Agents

This file documents each stage (agent) in the dubbing pipeline. Each agent is a self-contained function/module that processes inputs (prior JSONs via metadata) and produces JSON outputs. Agents are config-driven for scalability.

## General Agent Structure
- **Inputs**: List of prior stages to load JSON from (via `load_previous_result`).
- **Outputs**: JSON saved to `tmp/{stage}_results/{stage}.json`, tracked in `metadata.json`.
- **Dependencies**: Reuses existing functions; new agents implement similar pattern.
- **Adding New Agents**: 1. Add to `config/stages.yaml`. 2. Implement `{module}.{function}(tmp_path, metadata_path, **kwargs)`. 3. Define JSON schema here. 4. Update README.

## Agents

### 1. convert_mp4_to_wav
- **Purpose**: Extract full audio from MP4 to WAV.
- **Module/Function**: `convert.mp4_wav.convert_mp4_to_wav`
- **Inputs**: None.
- **Outputs**: JSON with audio paths, metadata.
- **JSON Schema**:
  ```json
  {
    "stage": "convert_mp4_to_wav",
    "full_wav_path": "full.wav",
    "duration": 120.5,
    "sample_rate": 22050,
    "channels": 1
  }
  ```
- **Notes**: Uses FFmpeg internally.

### 2. separate_audio
- **Purpose**: Separate vocals from instrumental using Demucs.
- **Module/Function**: `convert.separate_audio.separate`
- **Inputs**: [convert_mp4_to_wav] (load full_wav_path).
- **Outputs**: JSON with separated paths.
- **JSON Schema**:
  ```json
  {
    "stage": "separate_audio",
    "vocals_path": "vocals.wav",
    "instrumental_path": "accompaniment.wav",
    "separation_method": "demucs",
    "metadata": {"total_duration": 120.5}
  }
  ```
- **Notes**: Preserves music for later mixing.

### 3. transcribe
- **Purpose**: Transcribe vocals + diarization (speakers/embeddings) for RVC.
- **Module/Function**: `transcription.whisper.transcript`
- **Inputs**: [separate_audio] (load vocals_path).
- **Outputs**: JSON with segments, speakers, embeddings.
- **JSON Schema**:
  ```json
  {
    "stage": "transcribe",
    "segments": [
      {
        "start": 0.0, "end": 5.0, "text": "Dialogue", "speaker": "SPEAKER_00",
        "is_singing": false, "no_speech_prob": 0.1, "audioFilePath": "whisper_audio/0_5.wav",
        "speaker_embedding": [0.1, 0.2, ...]
      }
    ],
    "language": "ja", "text": "Full text",
    "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, ...]}
  }
  ```
- **Notes**: Reuses Whisper + Pyannote; embeddings for RVC voice cloning (download model per unique speaker).

### 4. emotion (Optional)
- **Purpose**: Detect emotions in segments for expressive TTS.
- **Module/Function**: `transcription.emotion.detect_emotions`
- **Inputs**: [transcribe].
- **Outputs**: JSON with emotions added to segments.
- **JSON Schema**:
  ```json
  {
    "stage": "emotion",
    "segments": [
      {"start": 0.0, "end": 5.0, "text": "Dialogue", "speaker": "SPEAKER_00", "emotion": "angry", "confidence": 0.8}
    ],
    "overall_emotions": {"SPEAKER_00": {"angry": 0.6, "neutral": 0.4}}
  }
  ```
- **Notes**: Toggle in config; integrate with existing emotion.py.

### 5. translate
- **Purpose**: Translate segments to target lang, preserve timing/speakers.
- **Module/Function**: `translate.openAi.translate_with_openai`
- **Inputs**: [transcribe] (or [emotion] if active).
- **Outputs**: JSON with translated_text.
- **JSON Schema**:
  ```json
  {
    "stage": "translate",
    "segments": [
      {"start": 0.0, "end": 5.0, "original_text": "Dialogue", "translated_text": "Dubbed line",
       "speaker": "SPEAKER_00", "is_singing": false}
    ],
    "target_lang": "en", "full_text": "Translated full"
  }
  ```
- **Notes**: Uses OpenAI; context from prior segments. Can optionally export SRT subtitles during translation (see export_srt agent).

### 6. export_srt (Optional)
- **Purpose**: Export both translated and original transcription data to SRT subtitle format for external use.
- **Module/Function**: `utils.srt_export.export_translation_to_srt` (integrated into translate stage)
- **Inputs**: [translate] (automatically triggered during translate stage when --export-srt flag is used).
- **Outputs**: SRT file(s) saved to specified directory, JSON with export metadata.
- **JSON Schema**:
  ```json
  {
    "stage": "export_srt",
    "srt_files": {
      "translated": "translated_subtitles_en.srt",
      "original": "original_transcription_en.srt"
    },
    "export_paths": {
      "translated": "./srt/translated_subtitles_en.srt",
      "original": "./srt/original_transcription_en.srt"
    },
    "export_params": {
      "text_field": "translated_text",
      "include_speaker": true,
      "include_original": false,
      "title": "Anime Dubbed Subtitles"
    },
    "file_sizes": {
      "translated": 2048,
      "original": 1892
    },
    "export_timestamp": "2025-01-21T12:00:00Z"
  }
  ```
- **Command Line Integration**:
  - `--export-srt`: Enable SRT export functionality (exports both translated and original subtitles by default)
  - `--export-srt-directory`: Directory path for SRT files (default: ./srt)
  - `--srt-text-field`: Text field to export ("translated_text" or "original_text")
  - `--srt-include-speaker`: Include speaker information in subtitles
  - `--srt-include-original`: Include original text alongside translation
  - `--srt-title`: Optional title for SRT file
- **Notes**: Integrated into translate stage; creates standard SRT files compatible with most video players. Files are automatically copied to results directory. By default, exports both translated subtitles and original transcription subtitles. Use `--srt-text-field` to control which text field is used for the primary export if needed.

### 7. build_refs
- **Purpose**: Extract reference audios per speaker from original vocals with re-transcription for improved voice cloning accuracy.
- **Module/Function**: Custom in main.py or tts (extract first non-singing 4s).
- **Inputs**: [translate], [separate_audio] (for waveform).
- **Outputs**: JSON with ref paths and speaker-specific transcribed text.
- **JSON Schema**:
  ```json
  {
    "stage": "build_refs",
    "refs_by_speaker": {
      "SPEAKER_00": {
        "audio_path": "refs/SPEAKER_00.wav",
        "ref_text": "Transcribed text from this specific reference audio"
      }
    },
    "default_ref": {
      "audio_path": "refs/default.wav",
      "ref_text": "Transcribed text from default reference audio"
    },
    "extraction_criteria": "first 4s non-singing segments per speaker, with re-transcription"
  }
  ```
- **Notes**: Uses torchaudio for slicing. Re-transcription of reference audio for better voice cloning accuracy. Speaker-specific transcribed text that matches the reference audio content. Improved alignment between ref_audio and ref_text for F5-TTS.

### 8. generate_tts
- **Purpose**: Generate TTS per segment, clone voice via XTTS (primary) or RVC using diarization.
- **Module/Function**: `tts.orchestrator.generate_dubbed_segments`
- **Inputs**: [build_refs], [translate], [transcribe] (embeddings for RVC fallback).
- **Outputs**: JSON with TTS file paths/timings.
- **JSON Schema**:
  ```json
  {
    "stage": "generate_tts",
    "tts_segments": [
      {"path": "tts/seg1.wav", "start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "duration": 4.8}
    ],
    "xtts_models_used": {"SPEAKER_00": "xtts_v2"},
    "tts_method": "xtts",
    "total_duration": 120.5,
    "tts_params": {"speed": 1.0, "language": "en", "temperature": 0.7}
  }
  ```
- **Notes**: Primary engine now Coqui XTTS-v2 for multilingual cloning; falls back to F5/Edge/RVC via config. Uses ref_audio (3-10s sliced from build_refs) and ref_text for conditioning. Download XTTS-v2 (~1GB) on first run. Skip singing; emotion integration via temperature if active. Ensure ref clips >=3s for cloning quality.
- **XTTS-specific**: Voice cloning from speaker refs; multilingual output matches target_lang. Duration adjustment post-synthesis for mix sync.

### 9. mix_audio
- **Purpose**: Mix TTS with original instrumental + music preservation.
- **Module/Function**: `dub.mixer.mix_audio`
- **Inputs**: [generate_tts], [separate_audio].
- **Outputs**: JSON with dubbed WAV path.
- **JSON Schema**:
  ```json
  {
    "stage": "mix_audio",
    "dubbed_wav_path": "dubbed.wav",
    "mixing_params": {"crossfade_duration": 0.1, "volume_adjust": 1.0}
  }
  ```
- **Notes**: Copy original for music segments.

### 10. mux_video
- **Purpose**: Mux dubbed audio with original video.
- **Module/Function**: FFmpeg in main.py.
- **Inputs**: [mix_audio].
- **Outputs**: JSON with final paths.
- **JSON Schema**:
  ```json
  {
    "stage": "mux_video",
    "final_video_path": "output.mp4",
    "mux_params": {"video_codec": "copy", "audio_codec": "aac"}
  }
  ```
- **Notes**: No re-encode video.

## Extending the Pipeline
To add a new stage (e.g., "post_process"):
1. Implement `new_module.post_process(tmp_path, metadata_path) -> data`.
2. Append to `config/stages.yaml` with inputs/outputs.
3. Define schema in this file.
4. Update downstream inputs if needed.
5. Run: Pipeline auto-includes via config.