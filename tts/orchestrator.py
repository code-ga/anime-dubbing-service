import json
import os
import torch
import torchaudio
from typing import List, Dict, Optional
from tts.F5 import generate_tts_custom


def generate_dubbed_segments(tmp_path: str, ref_audios_by_speaker: Dict[str, str], default_ref: Optional[str] = None) -> List[Dict]:
    """
    Generate dubbed audio segments from the translated JSON file, skipping singing segments and those with empty translated text.
    
    For each valid segment:
    - Uses speaker-specific reference audio if available, otherwise falls back to default_ref.
    - Generates TTS audio using F5-TTS at 22kHz.
    - Adjusts the generated audio duration to match the segment's (end - start) seconds by trimming or padding with silence.
    - This ensures timeline synchronization during later mixing.
    
    Args:
        tmp_path: Path to temporary directory.
        ref_audios_by_speaker: Dict mapping speaker names to reference audio paths.
        default_ref: Optional default reference audio path for speakers without specific refs.
    
    Returns:
        List of dicts with 'start', 'end', 'path' to generated WAV, and 'speaker'.
    """
    # Create output directory if it doesn't exist
    tts_dir = os.path.join(tmp_path, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    
    translated_path = os.path.join(tmp_path, "translated", "translated.json")
    with open(translated_path, 'r') as f:
        data = json.load(f)
    
    tts_segments = []
    SR = 22050  # F5-TTS output sample rate
    
    for seg in data.get('segments', []):
        # Skip singing segments and those with empty translated text
        if seg.get('is_singing', False) or not seg.get('translated_text', '').strip():
            continue  # Skipping logic for singing and silent/empty text
        
        speaker = seg.get('speaker')
        ref_audio = ref_audios_by_speaker.get(speaker, default_ref)
        
        if ref_audio is None:
            import logging
            logging.warning(f"No reference audio for speaker {speaker}, skipping segment {seg['start']}-{seg['end']}")
            continue
        
        start = seg['start']
        end = seg['end']
        duration = end - start
        output_wav = os.path.join(tts_dir, f"{start:.1f}_{end:.1f}.wav")
        
        # Generate TTS for translated text using speaker reference
        generate_tts_custom(
            text=seg['translated_text'],
            ref_audio_path=ref_audio,
            ref_text=seg.get('text', seg['translated_text']),  # Use original text if available for ref, else translated
            output_path=output_wav
        )
        
        # Load the generated audio and match duration for timeline sync
        waveform, sample_rate = torchaudio.load(output_wav)
        
        # Resample if the output sample rate differs (F5-TTS should be 22050, but handle variations)
        if sample_rate != SR:
            resampler = torchaudio.transforms.Resample(sample_rate, SR)
            waveform = resampler(waveform)
            sample_rate = SR
            # Overwrite with resampled audio
            torchaudio.save(output_wav, waveform, sample_rate)
        
        target_samples = int(duration * SR)
        current_samples = waveform.shape[1]
        
        if current_samples > target_samples:
            # Trim to target duration if too long
            waveform = waveform[:, :target_samples]
        elif current_samples < target_samples:
            # Pad with silence if too short
            pad_samples = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_samples))
        
        # Save the adjusted audio
        torchaudio.save(output_wav, waveform, sample_rate)
        
        tts_segments.append({
            'start': start,
            'end': end,
            'path': output_wav,
            'speaker': speaker
        })
    
    # Return empty list if no valid segments (e.g., all music or no segments)
    return tts_segments