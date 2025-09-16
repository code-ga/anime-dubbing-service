# Anime Dubbing Service

This project provides an automated pipeline for dubbing anime videos into target languages while preserving original music segments for copyright compliance. It uses Whisper for transcription and speaker diarization, OpenAI for translation, F5-TTS for voice synthesis with speaker cloning, and custom mixing to replace speech while keeping music intact.

## Features
- **Transcription**: Uses Whisper to transcribe audio and detect music/silence segments.
- **Speaker Diarization**: Identifies speakers using Pyannote.
- **Translation**: Translates dialogue to target languages (e.g., English) with context-aware anime-style naturalness.
- **TTS Generation**: Generates dubbed speech using F5-TTS, cloning speaker voices from reference audio.
- **Music Preservation**: Detects music (based on no-speech probability > 0.6 or empty/short text), skips TTS for those segments, and mixes original audio to retain music.
- **Audio Mixing**: Overlays dubbed speech on original timeline, preserving music and applying crossfades for smoothness.
- **Video Muxing**: Combines dubbed audio with original video using FFmpeg.

## Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (or `requirements-gpu.txt` for CUDA).
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
- `--music_threshold`: Threshold for music detection (default 0.6, hardcoded in whisper.py).
- `--target_lang`: Target language (default "en").

Temporary files are stored in `./tmp/` (not cleaned automatically).

## Music Preservation for Copyright
The pipeline detects potential music segments during transcription:
- **Detection Logic**: A segment is marked as music if:
  - `no_speech_prob > 0.6` (high confidence of no speech).
  - Text is empty.
  - Text is very short (< 2 words) and `no_speech_prob > 0.4`.
- **TTS Skipping**: Music segments are skipped during TTS generation; no dubbed audio is created for them.
- **Mixing Original Music**: During audio mixing, original audio from music segments is directly copied to the output, preserving copyright-protected content. Speech segments are replaced with generated TTS, and gaps are filled with original audio.
- This ensures compliance while dubbing dialogue.

Threshold can be adjusted via arg (though currently hardcoded; future updates may expose it).

## Workflow
```
Input MP4
  ↓ (convert_mp4_to_wav)
Full WAV Audio
  ↓ (transcript)
Transcript JSON (with speakers, music flags)
  ↓ (translate_with_openai)
Translated JSON
  ↓ (build refs: extract speaker ref audios from original)
Ref Audios by Speaker
  ↓ (generate_dubbed_segments: F5-TTS, skip music)
TTS Segments (paths, timings)
  ↓ (mix_audio: overlay TTS on original, preserve music)
Dubbed WAV
  ↓ (ffmpeg mux)
Output MP4
```

## JSON Schemas

### Transcript JSON (`tmp/transcript/transcript.json`)
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Original Japanese dialogue",
      "speaker": "Speaker1",
      "is_music": false,
      "no_speech_prob": 0.1,
      "audioFilePath": "tmp/whisper_audio/0.0_5.0.wav"
    },
    {
      "start": 5.0,
      "end": 10.0,
      "text": "",
      "speaker": null,
      "is_music": true,
      "no_speech_prob": 0.7,
      "audioFilePath": "tmp/whisper_audio/5.0_10.0.wav"
    }
  ],
  "language": "ja",
  "text": "Full transcribed text"
}
```

### Translated JSON (`tmp/translated/translated.json`)
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Original Japanese dialogue",
      "translated_text": "English dubbed line",
      "speaker": "Speaker1",
      "is_music": false
    },
    {
      "start": 5.0,
      "end": 10.0,
      "text": "",
      "translated_text": "",
      "speaker": null,
      "is_music": true
    }
  ],
  "language": "ja",
  "text": "Full translated text",
  "target_language": "en"
}
```

## New Files Added
- `tts/orchestrator.py`: Coordinates TTS generation, skipping music, and duration matching for timeline sync.
- `dub/mixer.py`: Mixes original audio with TTS, preserves music segments, applies crossfades.

## Testing
Run unit and integration tests:

```bash
pytest tests/test_music_dubbing.py
```

- **Unit Tests**: Music detection logic, TTS skipping, mixing preservation.
- **Integration**: Placeholder for end-to-end with sample audio (manual verification recommended).

## Dependencies
See `requirements.txt`. Key libraries: `openai`, `whisper`, `pyannote.audio`, `torchaudio`, `f5-tts` (via F5.py).

## Limitations
- Music detection is heuristic; may misclassify some segments.
- TTS quality depends on reference audio (3-5s clean speech recommended).
- Long videos may hit token limits in translation; context is limited to recent segments.
- GPU recommended for Whisper/Pyannote/F5-TTS.

## Future Improvements
- Expose music threshold dynamically.
- Add emotion/style transfer in TTS.
- Support batch processing.
- Improve music detection with additional models.