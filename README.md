# Anime Dubbing Service

This project provides an automated pipeline for dubbing anime videos into target languages while preserving original music segments for copyright compliance. It uses Whisper for transcription and speaker diarization, OpenAI for translation, F5-TTS for voice synthesis with speaker cloning, and custom mixing to replace speech while keeping music intact.

## Features
- **Transcription**: Uses Whisper to transcribe audio and detect music/silence segments.
- **Speaker Diarization**: Identifies speakers using Pyannote; saves embeddings for RVC voice cloning in TTS (ensures correct model per speaker, avoids unwanted downloads).
- **Translation**: Translates dialogue to target languages (e.g., English) with context-aware anime-style naturalness.
- **TTS Generation**: Generates dubbed speech using F5-TTS, cloning speaker voices from reference audio; integrates diarization for per-speaker RVC.
- **Music Preservation**: Detects music (based on no-speech probability > 0.6 or empty/short text), skips TTS for those segments, and mixes original audio to retain music.
- **Audio Mixing**: Overlays dubbed speech on original timeline, preserving music and applying crossfades for smoothness.
- **Video Muxing**: Combines dubbed audio with original video using FFmpeg.
- **Scalability**: Config-driven stages; add new ones (e.g., emotion detection) via `config/stages.yaml` without code changes.
- **Inter-Stage Data**: JSON results per stage, loaded via metadata for seamless access (e.g., translate accesses transcribe diarization).
- **Results Directory**: Final outputs saved to `./results/{timestamp}/` including video, metadata, and key JSONs.

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or `requirements-gpu.txt` for CUDA). Add `pyyaml` for config.
3. Set up environment variables in `.env`:
   - `OPENAI_API_KEY`: For translation (uses HackClub proxy).
   - `HUGGINGFACE_TOKEN`: For Pyannote models.
4. Ensure FFmpeg is installed and in PATH.

## Usage
Run the full pipeline:

```bash
python main.py input.mp4 output.mp4 --music_threshold 0.6 --target_lang en
```

- `input_mp4`: Path to source anime video.
- `output_mp4`: Path for dubbed output video.
- `--music_threshold`: Threshold for music detection (default 0.6).
- `--target_lang`: Target language (default "en").

Temporary files in `./tmp/`; final results in `./results/{timestamp}/`. Use `--keep-tmp` to retain intermediates.

## Scalable Pipeline Configuration
Stages defined in `config/stages.yaml`. Edit to reorder, add/remove (e.g., enable emotion). Pipeline dynamically loads and executes.

Example addition: Emotion stage after transcribe for expressive TTS.

## Music Preservation for Copyright
The pipeline detects potential music segments during transcription:
- **Detection Logic**: A segment is marked as music if:
  - `no_speech_prob > 0.6` (high confidence of no speech).
  - Text is empty.
  - Text is very short (< 2 words) and `no_speech_prob > 0.4`.
- **TTS Skipping**: Music segments are skipped during TTS generation; no dubbed audio is created for them.
- **Mixing Original Music**: During audio mixing, original audio from music segments is directly copied to the output, preserving copyright-protected content. Speech segments are replaced with generated TTS, and gaps are filled with original audio.
- This ensures compliance while dubbing.

Threshold adjustable via arg.

## Workflow
See Mermaid diagram in architecture plan for detailed flow with JSON interop.

High-level:
```
Input MP4 → Convert → Separate → Transcribe (Diarization) → [Emotion?] → Translate → Build Refs → TTS (RVC via Diarization) → Mix → Mux → results/
```

Inter-Stage: Each step saves/loads JSON via metadata (e.g., TTS uses transcribe embeddings for voice models).

## JSON Schemas
Detailed per-stage schemas in AGENTS.md. Examples:

### Transcript JSON (tmp/transcribe_results/transcribe.json)
```json
{
  "stage": "transcribe",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Original Japanese dialogue",
      "speaker": "SPEAKER_00",
      "is_singing": false,
      "no_speech_prob": 0.1,
      "audioFilePath": "tmp/whisper_audio/0.0_5.0.wav",
      "speaker_embedding": [0.1, 0.2, ...]
    },
    {
      "start": 5.0,
      "end": 10.0,
      "text": "",
      "speaker": null,
      "is_singing": true,
      "no_speech_prob": 0.7,
      "audioFilePath": "tmp/whisper_audio/5.0_10.0.wav"
    }
  ],
  "language": "ja",
  "text": "Full transcribed text",
  "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2, ...]}
}
```

### Translated JSON (tmp/translate_results/translate.json)
```json
{
  "stage": "translate",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "original_text": "Original Japanese dialogue",
      "translated_text": "English dubbed line",
      "speaker": "SPEAKER_00",
      "is_singing": false
    }
  ],
  "target_lang": "en",
  "full_text": "Full translated text"
}
```

All stages follow similar pattern; see AGENTS.md for full list.

## Results Directory
After completion, outputs saved to `./results/{workflow_start_timestamp}/`:
- `dubbed_{output_filename}.mp4`: Final video.
- `metadata.json`: Full pipeline log.
- `transcribe.json`: Transcript with diarization (for RVC reference).
- `diarization_embeddings.json`: Extracted speaker embeddings.
- `translated.json`: Final translations.
- Other key JSONs as needed.

## New Files Added/Updated
- `tts/orchestrator.py`: Coordinates TTS generation, skipping music, and duration matching for timeline sync. Now loads JSONs for diarization/RVC.
- `dub/mixer.py`: Mixes original audio with TTS, preserves music segments, applies crossfades.
- `config/stages.yaml`: Defines stage order/dependencies for scalability.
- `AGENTS.md`: Documents each stage/agent.

## Testing
Run unit and integration tests:

```bash
pytest tests/test_music_dubbing.py
```

- **Unit Tests**: Music detection logic, TTS skipping, mixing preservation.
- **Integration**: Placeholder for end-to-end with sample audio (manual verification recommended). Add tests for JSON loading/saving.

## Dependencies
See `requirements.txt`. Key libraries: `openai`, `whisper`, `pyannote.audio`, `torchaudio`, `f5-tts` (via F5.py), `pyyaml` (for config).

## Limitations
- Music detection is heuristic; may misclassify some segments.
- TTS quality depends on reference audio (3-5s clean speech recommended).
- Long videos may hit token limits in translation; context is limited to recent segments.
- GPU recommended for Whisper/Pyannote/F5-TTS.
- RVC model downloads: Automated but requires HF_TOKEN; cache models per embedding.

## Future Improvements
- Expose music threshold dynamically in config.
- Add emotion/style transfer in TTS using optional stage.
- Support batch processing.
- Improve music detection with additional models.
- Parallel stages (e.g., emotion + translate) for speed.