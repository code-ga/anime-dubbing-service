import json
import os
import shutil
import logging
import torch
import torchaudio
from datetime import datetime
from typing import List, Dict, Optional
from tts.F5 import generate_tts_custom
from utils.metadata import load_previous_result


def generate_dubbed_segments(tmp_path, metadata_path, inputs_data, **kwargs) -> dict:
    """
    Generate dubbed audio segments from the translated JSON file, skipping singing segments and those with empty translated text.
    
    For each valid segment:
    - Uses speaker-specific reference audio if available, otherwise falls back to default_ref.
    - Generates TTS audio using F5-TTS at 22kHz.
    - Adjusts the generated audio duration to match the segment's (end - start) seconds by trimming or padding with silence.
    - This ensures timeline synchronization during later mixing.
    
    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data.
        **kwargs: Additional arguments.
    
    Returns:
        Stage data with tts_segments, rvc_models_used, total_duration.
    """
    # Load previous results
    translate_data = inputs_data["translate"]
    transcribe_data = inputs_data["transcribe"]
    build_refs_data = inputs_data["build_refs"]
    
    ref_audios_by_speaker = build_refs_data["refs_by_speaker"]
    default_ref = build_refs_data["default_ref"]
    speaker_embeddings = transcribe_data["speaker_embeddings"]
    
    # Create output directory if it doesn't exist
    tts_dir = os.path.join(tmp_path, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    
    data = translate_data
    tts_segments = []
    SR = 22050  # F5-TTS output sample rate
    
    rvc_models_used = {}
    for speaker in ref_audios_by_speaker:
        # Placeholder for RVC model selection based on embedding
        # In real implementation, compute similarity to known models or download if unknown
        rvc_model_id = "default_rvc_model"  # Replace with actual logic using speaker_embeddings[speaker]
        rvc_models_used[speaker] = rvc_model_id
    
    for seg in data.get('segments', []):
        # Skip singing segments and those with empty translated text
        if seg.get('is_singing', False) or not seg.get('translated_text', '').strip():
            continue  # Skipping logic for singing and silent/empty text
        
        speaker = seg.get('speaker')
        ref_audio_rel = ref_audios_by_speaker.get(speaker, default_ref)
        ref_audio = os.path.join(tmp_path, ref_audio_rel) if ref_audio_rel else None
        
        if ref_audio is None:
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
            ref_text=seg.get('original_text', seg['translated_text']),  # Use original text for ref
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
            'path': output_wav,
            'start': start,
            'end': end,
            'speaker': speaker,
            'duration': duration
        })
    
    total_duration = sum(seg['end'] - seg['start'] for seg in tts_segments)
    
    stage_data = {
        "stage": "generate_tts",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "tts_segments": tts_segments,
        "rvc_models_used": rvc_models_used,
        "total_duration": total_duration
    }
    
    return stage_data

def build_speaker_refs(tmp_path, metadata_path, inputs_data, **kwargs) -> dict:
    """
    Extract reference audios per speaker from original vocals.
 
    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data.
        **kwargs: Additional arguments.
 
    Returns:
        Stage data with refs_by_speaker, default_ref, extraction_criteria.
    """
    transcribe_data = load_previous_result(metadata_path, "transcribe")
    separate_data = inputs_data["separate_audio"]
    
    vocals_path = os.path.join(tmp_path, separate_data["vocals_path"])
    waveform, sr = torchaudio.load(vocals_path)
    
    refs_dir = os.path.join(tmp_path, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    
    refs_by_speaker = {}
    speakers = set(seg.get("speaker") for seg in transcribe_data["segments"] if seg.get("speaker"))
    
    for speaker in speakers:
        non_singing_segs = [
            seg for seg in transcribe_data["segments"]
            if seg.get("speaker") == speaker
            and not seg.get("is_singing", False)
            and seg.get("no_speech_prob", 1.0) < 0.5
        ]
        if non_singing_segs:
            non_singing_segs.sort(key=lambda x: x["start"])
            slices = []
            for seg in non_singing_segs:
                start = seg["start"]
                end = seg["end"]
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                slice_w = waveform[:, start_sample:end_sample]
                slices.append(slice_w)
            if slices:
                concatenated = torch.cat(slices, dim=1)
                ref_path = os.path.join(refs_dir, f"{speaker}_long.wav")
                torchaudio.save(ref_path, concatenated, sr)
                refs_by_speaker[speaker] = f"refs/{speaker}_long.wav"
    
    default_ref = None
    if refs_by_speaker:
        first_speaker = list(refs_by_speaker.keys())[0]
        default_ref_path = os.path.join(refs_dir, "default.wav")
        source_path = os.path.join(tmp_path, refs_by_speaker[first_speaker])
        shutil.copy(source_path, default_ref_path)
        default_ref = "refs/default.wav"
    else:
        all_non_singing_segs = [
            seg for seg in transcribe_data["segments"]
            if not seg.get("is_singing", False)
            and seg.get("no_speech_prob", 1.0) < 0.5
        ]
        if all_non_singing_segs:
            all_non_singing_segs.sort(key=lambda x: x["start"])
            slices = []
            for seg in all_non_singing_segs:
                start = seg["start"]
                end = seg["end"]
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                slice_w = waveform[:, start_sample:end_sample]
                slices.append(slice_w)
            if slices:
                concatenated = torch.cat(slices, dim=1)
                default_ref_path = os.path.join(refs_dir, "default_long.wav")
                torchaudio.save(default_ref_path, concatenated, sr)
                default_ref = "refs/default_long.wav"
    
    stage_data = {
        "stage": "build_refs",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "refs_by_speaker": refs_by_speaker,
        "default_ref": default_ref,
        "extraction_criteria": "all non-singing segments concatenated"
    }
    
    return stage_data