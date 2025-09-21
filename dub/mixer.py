import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pydub import AudioSegment


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
            # Load TTS audio, resample to original SR, handle channels, trim/pad to exact duration
            seg_audio = AudioSegment.from_wav(os.path.join(tmp_path, seg["path"]))

            # Resample if TTS SR differs from original (F5-TTS is 22050 Hz, original may be 44100 Hz)
            if seg_audio.frame_rate != sr:
                seg_audio = seg_audio.set_frame_rate(sr)

            # Convert mono to stereo if needed
            if seg_audio.channels == 1:
                seg_audio = seg_audio.set_channels(2)

            # Speed up or pad to match exact segment duration
            if len(seg_audio) > duration_ms:
                # Calculate speed factor to fit the target duration
                speed_factor = duration_ms / len(seg_audio)
                # Apply speed change (pydub speedup method)
                seg_audio = seg_audio.speedup(playback_speed=speed_factor)
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
