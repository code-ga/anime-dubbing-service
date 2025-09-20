import torch
import torchaudio
from f5_tts.api import F5TTS
import numpy as np
from tts.utils import adjust_audio_duration
from vocos import Vocos
from typing import List, Dict
import os
import logging
import gc

# TTS Configuration Constants
# These constants control batch processing for memory management during TTS generation.
# Adjust these values based on your system's memory capacity:
# - Lower values use less memory but may be slower
# - Higher values are faster but require more memory
TTS_SEGMENT_BATCH_SIZE = (
    10  # Number of segments to process at once for memory management
)


def generate_tts_custom(
    text: str,
    output_path: str,
    ref_audio_path: str,
    ref_text: str,
    model: F5TTS,
    vocos: Vocos,
    checkpoint_path: str = "checkpoints/f5_tts_multilingual.pth",
):
    """
    Generate TTS using a custom fine-tuned F5-TTS model.

    :param checkpoint_path: Path to the custom model checkpoint (e.g., ckpt.pth).
    :param text: Input text to synthesize.
    :param output_path: Path to save the generated WAV audio file.
    :param ref_audio_path: Path to reference audio for voice cloning.
    :param ref_text: Reference text corresponding to the reference audio.
    :param model: Pre-loaded F5TTS model instance.
    :param vocos: Pre-loaded Vocos model instance.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference to get mel spectrogram
    print("Performing inference...")
    print(f"{ref_audio_path} - {ref_text} - {text}")
    _, _, spec = model.infer(ref_audio_path, gen_text=text)

    if spec is None:
        raise ValueError(
            "TTS inference returned None spectrogram; check model or inputs."
        )

    # Decode mel spectrogram with Vocos
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(device)  # [1, n_mels, time]
    with torch.no_grad():
        audio = vocos.decode(spec_tensor)

    # Clean up spec tensor immediately after decoding
    del spec_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to CPU and numpy for saving
    audio = audio.squeeze(0).cpu().numpy()

    # Save output at 24kHz
    sr = 24000
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    torchaudio.save(output_path, audio_tensor, sr)

    # Clean up tensors after saving
    del audio_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def generate_tts_for_speaker(
    segments: List[Dict],
    speaker: str,
    ref_audios_by_speaker: Dict,
    default_ref: str,
    tmp_path: str,
    target_sr: int,
    model: F5TTS,
    vocos: Vocos,
    ref_text: str = "",
) -> List[Dict]:
    """
    Generate TTS for all segments of a single speaker, reusing model instances for efficiency.

    Args:
        segments: List of segments for this speaker
        speaker: Speaker identifier
        ref_audios_by_speaker: Dictionary mapping speakers to reference audio paths
        default_ref: Default reference audio path
        tmp_path: Temporary directory path
        target_sr: Target sample rate
        model: Pre-loaded F5TTS model instance
        vocos: Pre-loaded Vocos model instance

    Returns:
        List of TTS segment dictionaries
    """
    ref_audio_rel = ref_audios_by_speaker.get(speaker, default_ref)
    ref_audio = os.path.join(tmp_path, ref_audio_rel) if ref_audio_rel else None

    if ref_audio is None:
        logging.warning(
            f"No reference audio for speaker {speaker}, skipping all segments"
        )
        return []

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

            # Use provided ref_text if available, otherwise fall back to original text
            ref_text_to_use = (
                ref_text
                if ref_text
                else seg.get("original_text", seg["translated_text"])
            )

            # Generate TTS for translated text using speaker reference
            generate_tts_custom(
                text=seg["translated_text"],
                ref_audio_path=ref_audio,
                ref_text=ref_text_to_use,
                output_path=output_wav,
                model=model,
                vocos=vocos,
            )

            # Load the generated audio and match duration for timeline sync
            waveform, sample_rate = torchaudio.load(output_wav)

            # Resample if the output sample rate differs (Vocos outputs 24000Hz)
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
                sample_rate = target_sr
                # Overwrite with resampled audio
                torchaudio.save(output_wav, waveform, sample_rate)

                # Clean up resampler
                del resampler

            # Adjust audio duration to match segment duration
            # This will speed up if too long, pad with silence if too short, or do nothing if it fits
            waveform = adjust_audio_duration(waveform, sample_rate, duration)

            # Save the adjusted audio
            torchaudio.save(output_wav, waveform, sample_rate)

            # Clean up waveform tensor after processing
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

            gc.collect()

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return speaker_tts_segments
