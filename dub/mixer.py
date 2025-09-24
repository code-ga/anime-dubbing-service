import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import speedup


def remove_trailing_silence(
    audio_segment: AudioSegment,
    silence_thresh: float = -40.0,
    min_silence_len: int = 100,
) -> AudioSegment:
    """
    Remove silence from the end of an audio segment while preserving the original content.

    Args:
        audio_segment: The audio segment to process
        silence_thresh: Silence threshold in dBFS (default: -40.0, good for anime dubbing)
        min_silence_len: Minimum silence length in milliseconds to consider for removal (default: 100ms)

    Returns:
        AudioSegment with trailing silence removed
    """
    if len(audio_segment) == 0:
        return audio_segment

    # Detect non-silent chunks from the end
    non_silent_chunks = detect_nonsilent(
        audio_segment,
        min_silence_len=min_silence_len,
        silence_thresh=int(silence_thresh),
    )

    if not non_silent_chunks:
        # If no non-silent chunks found, return empty audio
        return AudioSegment.silent(duration=0)

    # Find the last non-silent chunk
    last_non_silent_end = int(non_silent_chunks[-1][1])

    # If the last non-silent chunk doesn't go to the end, there's trailing silence
    if last_non_silent_end < len(audio_segment):
        # Remove silence from the end - slice up to the last non-silent chunk end
        # Use explicit type casting to resolve type checker issues
        from typing import cast

        result = cast(AudioSegment, audio_segment[:last_non_silent_end])
        return result

    # No trailing silence found, return original
    return audio_segment


def mix_audio(
    tmp_path: str,
    metadata_path: str,
    inputs_data: dict,
    output_wav_path: Optional[str] = None,
    crossfade_duration: float = 0.1,
) -> dict:
    """
    Mixes the instrumental audio with dubbed vocals (TTS for speech + original vocals for singing).
    Starts with instrumental as base, overlays vocal segments, applies crossfades.
    """

    # Load full instrumental audio
    instrumental_path = os.path.join(tmp_path, "accompaniment.wav")
    mixed = AudioSegment.from_wav(instrumental_path)
    sr = mixed.frame_rate

    # Load transcribe data to extract singing segments (original audio paths)
    transcribe_data = inputs_data["transcribe"]
    data = transcribe_data["segments"]

    # Collect singing segments (preserve original vocal audio)
    singing_segments = [
        {
            "start": s["start"],
            "end": s["end"],
            "path": s["audioFilePath"],
            "is_singing": True,
        }
        for s in data
        if s.get("is_singing", False)
    ]

    # Load TTS segments from generate_tts structured data
    generate_tts_data = inputs_data["generate_tts"]
    speech_segments = []
    for segment in generate_tts_data["tts_segments"]:
        speech_segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "path": segment["path"],
                "is_singing": False,
            }
        )

    # All vocal segments = speech TTS + singing originals
    all_vocal_segments = speech_segments + singing_segments
    all_vocal_segments.sort(key=lambda x: x["start"])

    # Placement of vocal segments on instrumental
    for seg in all_vocal_segments:
        start_ms = int(seg["start"] * 1000)
        duration_ms = int((seg["end"] - seg["start"]) * 1000)
        if not seg.get("is_singing", False):  # Speech TTS
            print("Mixing speech segment:", seg["path"], start_ms, duration_ms)
            # Load TTS audio, resample to original SR, handle channels, trim/pad to exact duration
            seg_audio = AudioSegment.from_wav(os.path.join(tmp_path, seg["path"]))
            # Remove trailing silence to prevent oversized audio segments
            seg_audio = remove_trailing_silence(seg_audio)

            # Resample if TTS SR differs from original (F5-TTS is 22050 Hz, original may be 44100 Hz)
            if seg_audio.frame_rate != sr:
                seg_audio = seg_audio.set_frame_rate(sr)

            # Convert mono to stereo if needed
            if seg_audio.channels == 1:
                seg_audio = seg_audio.set_channels(2)

            # Speed up or pad to match exact segment duration
            if len(seg_audio) > duration_ms:
                # Calculate speed factor to fit the target duration
                if duration_ms <= 0:
                    seg_audio = AudioSegment.silent(duration=duration_ms)
                elif len(seg_audio) == 0:
                    seg_audio = AudioSegment.silent(duration=duration_ms)
                else:
                    speed_factor = len(seg_audio) / duration_ms
                    if (
                        speed_factor <= 0 or speed_factor > 10
                    ):  # Cap extreme speeds to avoid artifacts
                        seg_audio = AudioSegment.silent(duration=duration_ms)
                    else:
                        # Apply speed change (pydub speedup method) - skip if no speedup needed or segment too short
                        if speed_factor != 1.0 and len(seg_audio) >= 150:
                            seg_audio = speedup(seg_audio, playback_speed=speed_factor)
            else:
                pad_ms = duration_ms - len(seg_audio)
                silence_segment = AudioSegment.silent(duration=pad_ms)
                seg_audio = seg_audio + silence_segment

            fade_ms = int(crossfade_duration * 1000)
            seg_audio = seg_audio.fade_in(fade_ms).fade_out(fade_ms)

            # Overlay TTS audio on mixed
            mixed = mixed.overlay(seg_audio, position=start_ms)
        else:  # Singing segment: overlay original vocal
            # Load original vocal segment audio
            seg_audio = AudioSegment.from_wav(os.path.join(tmp_path, seg["path"]))

            # Resample if original SR differs
            if seg_audio.frame_rate != sr:
                seg_audio = seg_audio.set_frame_rate(sr)

            # Convert mono to stereo if needed
            if seg_audio.channels == 1:
                seg_audio = seg_audio.set_channels(2)

            # Trim or pad to match exact segment duration
            if len(seg_audio) > duration_ms:
                seg_audio = seg_audio[:duration_ms]
            else:
                pad_ms = duration_ms - len(seg_audio)
                silence_segment = AudioSegment.silent(duration=pad_ms)
                seg_audio = seg_audio + silence_segment

            fade_ms = int(crossfade_duration * 1000)
            seg_audio = seg_audio.fade_in(fade_ms).fade_out(fade_ms)

            # Overlay original singing audio on mixed
            mixed = mixed.overlay(seg_audio, position=start_ms)

    # Save the mixed audio to a standard location
    dubbed_wav_path = os.path.join(tmp_path, "dubbed.wav")
    mixed.export(dubbed_wav_path, format="wav")

    # Return stage data
    stage_data = {
        "stage": "mix_audio",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "dubbed_wav_path": "dubbed.wav",
        "mixing_params": {
            "crossfade_duration": crossfade_duration,
            "volume_adjust": 1.0,
        },
    }

    return stage_data
