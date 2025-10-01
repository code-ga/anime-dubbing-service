import os
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import speedup
from utils.metadata import load_previous_result
import logging


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
    inputs_data=None,
    max_speed_factor: float = 2.0,
    **kwargs,
) -> dict:
    """
    Mixes the instrumental audio with dubbed vocals (TTS for speech + original vocals for singing).
    Creates base track from vocals + instrumental, then mutes original vocals in spoken segments
    before overlaying TTS to reduce audio loss.
    """

    # Extract crossfade_duration from kwargs with default value and ensure it's a float
    crossfade_duration = float(kwargs.get("crossfade_duration", 0.1))

    # Load audio based on whether separation was performed
    separate_data = inputs_data.get("separate_audio") if inputs_data else None
    if (
        separate_data
        and "vocals_path" in separate_data
        and "instrumental_path" in separate_data
    ):
        # Audio separation was performed
        vocals_path = os.path.join(tmp_path, separate_data["vocals_path"])
        instrumental_path = os.path.join(tmp_path, separate_data["instrumental_path"])
        vocals = AudioSegment.from_wav(vocals_path)
        instrumental = AudioSegment.from_wav(instrumental_path)
        sr = vocals.frame_rate
        logging.info(
            f"Using separated audio: vocals={vocals_path}, instrumental={instrumental_path}"
        )
    else:
        # Audio separation was skipped - use full audio as base and overlay TTS
        convert_data = inputs_data.get("convert_mp4_to_wav", {}) if inputs_data else {}
        full_audio_path = os.path.join(
            tmp_path, convert_data.get("full_wav_path", "full.wav")
        )
        vocals = AudioSegment.from_wav(full_audio_path)
        instrumental = AudioSegment.silent(
            duration=len(vocals)
        )  # Empty instrumental track
        sr = vocals.frame_rate
        logging.info(
            f"Audio separation skipped - using full audio as base: {full_audio_path}"
        )
        logging.warning(
            "⚠️  Audio separation was skipped. TTS will be overlaid on original audio, which may cause echo/overlap effects."
        )

    # Load translate data to extract segments with is_singing information
    translate_data = load_previous_result(metadata_path, "translate")
    segments = translate_data["segments"]

    # Load TTS segments from generate_tts structured data
    generate_tts_data = load_previous_result(metadata_path, "generate_tts")
    tts_segments = generate_tts_data["tts_segments"]

    # Create base track: vocals overlaid on instrumental
    base_track = vocals.overlay(instrumental)

    # Find all non-singing segments (where we need to replace vocals with TTS)
    non_singing_segments = [
        {"start": seg["start"], "end": seg["end"]}
        for seg in segments
        if not seg.get("is_singing", False)
    ]

    # Create muted vocals by replacing all spoken segments with silence at once
    muted_vocals = vocals
    for seg in non_singing_segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            continue  # Skip invalid segments
        print(f"Muting vocals from {start_ms}ms to {end_ms}ms")
        # Create silence segment of exact duration
        silence = AudioSegment.silent(duration=duration_ms)

        # Replace the vocal segment with silence
        muted_vocals = muted_vocals[:start_ms] + silence + muted_vocals[end_ms:]

    # Create new base track with muted vocals + instrumental
    base_track = muted_vocals.overlay(instrumental)

    total_segments = len(tts_segments)
    # last_end_time = 0

    logging.info(f"🎚️  Starting audio mixing with {total_segments} TTS segments")

    # Overlay TTS segments on the base track
    for idx, tts_seg in enumerate(tts_segments, 1):
        start_ms = int(tts_seg["start"] * 1000)
        duration_ms = int((tts_seg["end"] - tts_seg["start"]) * 1000)
        # if duration_ms <= 0 or start_ms < last_end_time:
        #     logging.warning(
        #         f"⚠️  Skipping invalid or overlapping TTS segment {idx}/{total_segments}: {tts_seg['path']} for duration {duration_ms}ms"
        #     )
        #     continue  # Skip invalid or overlapping segments
        # last_end_time = start_ms + duration_ms

        # Load TTS audio
        speaker = tts_seg.get("speaker", "Unknown")
        tts_method = tts_seg.get("tts_method", "unknown")
        logging.info(f"🎵 Mixing segment {idx}/{total_segments}: {speaker} ({tts_method}) - {duration_ms}ms at {start_ms}ms")

        tts_audio = AudioSegment.from_wav(os.path.join(tmp_path, tts_seg["path"]))
        # Remove trailing silence to prevent oversized audio segments
        tts_audio = remove_trailing_silence(tts_audio)

        # Resample if TTS SR differs from original
        if tts_audio.frame_rate != sr:
            tts_audio = tts_audio.set_frame_rate(sr)

        # Convert mono to stereo if needed
        if tts_audio.channels == 1:
            tts_audio = tts_audio.set_channels(2)

        # Speed up or pad to match exact segment duration (only for Edge-TTS)
        tts_method = tts_seg.get("tts_method", "unknown")
        if len(tts_audio) > duration_ms and tts_method == "edge":
            logging.info(f"⚡ Speeding up Edge-TTS from {len(tts_audio)}ms to {duration_ms}ms")
            # Calculate speed factor to fit the target duration
            if duration_ms <= 0:
                tts_audio = AudioSegment.silent(duration=duration_ms)
            elif len(tts_audio) == 0:
                tts_audio = AudioSegment.silent(duration=duration_ms)
            else:
                speed_factor = len(tts_audio) / duration_ms
                if (
                    speed_factor <= 0
                ):  # Cap extreme speeds to avoid artifacts (configurable max speed)
                    tts_audio = AudioSegment.silent(duration=duration_ms)
                else:
                    # Apply speed change - skip if no speedup needed or segment too short
                    if speed_factor != 1.0 and len(tts_audio) >= 150:
                        # Limit speed factor to max_speed_factor
                        actual_speed = (
                            speed_factor
                            if speed_factor < max_speed_factor
                            else max_speed_factor
                        )
                        logging.info(f"🚀 Applying speed factor: {actual_speed:.2f}x (limited to max {max_speed_factor}x)")
                        tts_audio = speedup(tts_audio, playback_speed=actual_speed)
        else:
            print(f"Padding TTS from {len(tts_audio)}ms to {duration_ms}ms")
            # Pad with silence to reach target duration
            pad_ms = duration_ms - len(tts_audio)
            silence_segment = AudioSegment.silent(duration=pad_ms)
            tts_audio = tts_audio + silence_segment

        # Apply crossfades
        fade_ms = int(crossfade_duration * 1000)
        tts_audio = tts_audio.fade_in(fade_ms).fade_out(fade_ms)

        # Overlay TTS audio on base track
        base_track = base_track.overlay(tts_audio, position=start_ms)

    # Save the mixed audio to a standard location
    dubbed_wav_path = os.path.join(tmp_path, "dubbed.wav")
    base_track.export(dubbed_wav_path, format="wav")

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
