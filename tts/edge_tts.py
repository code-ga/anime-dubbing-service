import asyncio
import os
import logging
import torchaudio
import torch
import gc
from typing import List, Dict, Optional
import edge_tts
from edge_tts import VoicesManager
from tts.utils import adjust_audio_duration

# TTS Configuration Constants
TTS_SEGMENT_BATCH_SIZE = (
    1  # Number of segments to process at once for memory management
)

# Edge-tts voice mappings for different speakers
# These can be customized based on the target language and desired voice characteristics
EDGE_TTS_VOICES = {
    "SPEAKER_00": "en-US-GuyNeural",  # Male voice
    "SPEAKER_01": "en-US-AriaNeural",  # Female voice
    "SPEAKER_02": "en-US-DavisNeural",  # Male voice (alternative)
    "SPEAKER_03": "en-US-JaneNeural",  # Female voice (alternative)
    "SPEAKER_04": "en-US-JasonNeural",  # Male voice (alternative)
    "SPEAKER_05": "en-US-SaraNeural",  # Female voice (alternative)
    "default": "en-US-GuyNeural",  # Default fallback voice
}

# Language-specific voice mappings
LANGUAGE_VOICE_MAP = {
    "en": {
        "male": ["en-US-GuyNeural", "en-US-DavisNeural", "en-US-JasonNeural"],
        "female": ["en-US-AriaNeural", "en-US-JaneNeural", "en-US-SaraNeural"],
    },
    "ja": {
        "male": ["ja-JP-KeitaNeural", "ja-JP-DaichiNeural"],
        "female": ["ja-JP-NanamiNeural", "ja-JP-AoiNeural"],
    },
    "zh": {
        "male": ["zh-CN-YunxiNeural", "zh-CN-YunhaoNeural"],
        "female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaohanNeural"],
    },
}


def get_voice_for_speaker(
    speaker: str, language: str = "en", gender_preference: str = "neutral"
) -> str:
    """
    Get the appropriate edge-tts voice for a speaker based on language and gender preference.

    Args:
        speaker: Speaker identifier (e.g., "SPEAKER_00")
        language: Target language code (e.g., "en", "ja", "zh")
        gender_preference: Preferred gender ("male", "female", "neutral")

    Returns:
        Voice name string for edge-tts
    """
    # First try speaker-specific mapping
    if speaker in EDGE_TTS_VOICES:
        return EDGE_TTS_VOICES[speaker]

    # Fall back to language and gender-based selection
    if language in LANGUAGE_VOICE_MAP:
        lang_voices = LANGUAGE_VOICE_MAP[language]

        if gender_preference == "male" and lang_voices["male"]:
            return lang_voices["male"][0]
        elif gender_preference == "female" and lang_voices["female"]:
            return lang_voices["female"][0]
        else:
            # Return first available voice for the language
            all_voices = lang_voices["male"] + lang_voices["female"]
            return all_voices[0] if all_voices else EDGE_TTS_VOICES["default"]

    # Final fallback
    return EDGE_TTS_VOICES["default"]


async def generate_tts_audio_async(
    text: str,
    output_path: str,
    voice: str,
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
) -> bool:
    """
    Generate TTS audio asynchronously using edge-tts.

    Args:
        text: Text to synthesize
        output_path: Path to save the generated audio file
        voice: Voice name to use
        rate: Speaking rate adjustment (e.g., "+50%", "-20%")
        volume: Volume adjustment (e.g., "+10%", "-5%")
        pitch: Pitch adjustment (e.g., "+2Hz", "-1Hz")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create TTS communication object
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, volume=volume, pitch=pitch
        )

        # Generate and save audio
        await communicate.save(output_path)

        # Verify file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            logging.error(f"Generated audio file is empty or missing: {output_path}")
            return False

    except Exception as e:
        logging.error(f"Error generating TTS audio: {e}")
        return False


def generate_tts_for_speaker(
    segments: List[Dict],
    speaker: str,
    ref_audios_by_speaker: Dict,
    default_ref: str,
    tmp_path: str,
    target_sr: int,
    language: str = "en",
    gender_preference: str = "neutral",
    **kwargs,
) -> List[Dict]:
    """
    Generate TTS for all segments of a single speaker using edge-tts.

    Args:
        segments: List of segments for this speaker
        speaker: Speaker identifier
        ref_audios_by_speaker: Dictionary mapping speakers to reference audio paths
        default_ref: Default reference audio path
        tmp_path: Temporary directory path
        target_sr: Target sample rate
        language: Target language for TTS
        gender_preference: Preferred gender for voice selection
        **kwargs: Additional arguments

    Returns:
        List of TTS segment dictionaries
    """
    # Get voice for this speaker
    voice = get_voice_for_speaker(speaker, language, gender_preference)

    # Create output directory if it doesn't exist
    tts_dir = os.path.join(tmp_path, "tts")
    os.makedirs(tts_dir, exist_ok=True)

    speaker_tts_segments = []

    # Process segments in batches to manage memory usage
    for i in range(0, len(segments), TTS_SEGMENT_BATCH_SIZE):
        batch_segments = segments[i : i + TTS_SEGMENT_BATCH_SIZE]

        for seg in batch_segments:
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            output_wav = os.path.join(tts_dir, f"{start:.1f}_{end:.1f}.wav")

            # Get text to synthesize
            text = seg.get("translated_text", "").strip()
            if not text:
                logging.warning(
                    f"Empty translated text for segment {start}-{end}, skipping"
                )
                continue

            # Generate TTS audio using edge-tts
            success = asyncio.run(
                generate_tts_audio_async(
                    text=text,
                    output_path=output_wav,
                    voice=voice,
                    rate="+0%",  # Normal speed
                    volume="+0%",  # Normal volume
                    pitch="+0Hz",  # Normal pitch
                )
            )

            if not success:
                logging.error(f"Failed to generate TTS for segment {start}-{end}")
                continue

            # Load the generated audio
            try:
                waveform, sample_rate = torchaudio.load(output_wav)

                # Resample if needed
                if sample_rate != target_sr:
                    resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                    waveform = resampler(waveform)
                    sample_rate = target_sr
                    # Clean up resampler
                    del resampler

                # Adjust audio duration to match segment duration
                waveform = adjust_audio_duration(waveform, sample_rate, duration)

                # Save the adjusted audio
                torchaudio.save(output_wav, waveform, sample_rate)

                # Clean up waveform tensor
                del waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                speaker_tts_segments.append(
                    {
                        "path": output_wav,
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "duration": duration,
                    }
                )

            except Exception as e:
                logging.error(
                    f"Error processing generated audio for segment {start}-{end}: {e}"
                )
                continue

            gc.collect()

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return speaker_tts_segments


def list_available_voices(language: Optional[str] = None) -> List[str]:
    """
    List available edge-tts voices, optionally filtered by language.

    Args:
        language: Language code to filter voices (e.g., "en", "ja", "zh")

    Returns:
        List of available voice names
    """
    try:
        voices_manager = asyncio.run(VoicesManager.create())
        if language:
            voices = voices_manager.find(Language=language)
        else:
            voices = voices_manager

        # Handle different possible return types from VoicesManager
        try:
            return [voice["Name"] for voice in voices]  # type: ignore
        except (TypeError, KeyError):
            # If iteration fails, return empty list
            return []
    except Exception as e:
        logging.error(f"Error listing available voices: {e}")
        return []


def test_voice_quality(
    voice: str, test_text: str = "This is a test of the text-to-speech system."
) -> bool:
    """
    Test a voice to ensure it's working properly.

    Args:
        voice: Voice name to test
        test_text: Text to synthesize for testing

    Returns:
        True if voice works, False otherwise
    """
    try:
        # Create a temporary test file
        test_output = f"test_{voice.replace('-', '_')}.wav"

        success = asyncio.run(
            generate_tts_audio_async(
                text=test_text, output_path=test_output, voice=voice
            )
        )

        # Clean up test file
        if os.path.exists(test_output):
            os.remove(test_output)

        return success

    except Exception as e:
        logging.error(f"Voice test failed for {voice}: {e}")
        return False
