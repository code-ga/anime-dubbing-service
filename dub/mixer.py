import json
import os
import glob
import torch
import torchaudio
from torchaudio.transforms import Resample
from typing import List, Dict


def mix_audio(tmp_path: str, output_wav_path: str, crossfade_duration: float = 0.1) -> None:
    """
    Mixes the instrumental audio with dubbed vocals (TTS for speech + original vocals for singing).
    Starts with instrumental as base, overlays vocal segments, applies crossfades.
    """
    import glob
    import re

    # Load full instrumental audio
    instrumental_path = os.path.join(tmp_path, 'accompaniment.wav')
    full_audio, sr = torchaudio.load(instrumental_path)
    device = full_audio.device
    dtype = full_audio.dtype

    # Load translated data to extract singing segments
    translated_path = os.path.join(tmp_path, 'translated', 'translated.json')
    with open(translated_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['segments']

    # Collect singing segments (preserve original vocal audio)
    singing_segments = [{'start': s['start'], 'end': s['end'], 'path': s['audioFilePath'], 'is_singing': True} for s in data if s.get('is_singing', False)]

    # Load TTS segments by scanning tts directory
    tts_dir = os.path.join(tmp_path, 'tts')
    speech_segments = []
    tts_files = glob.glob(os.path.join(tts_dir, '*.wav'))
    for file_path in tts_files:
        basename = os.path.basename(file_path).replace('.wav', '')
        match = re.match(r'([\d.]+)_([\d.]+)', basename)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            speech_segments.append({'start': start, 'end': end, 'path': file_path, 'is_singing': False})

    # Collect singing segments (preserve original vocal audio)
    singing_segments = [{'start': s['start'], 'end': s['end'], 'path': s['audioFilePath'], 'is_singing': True} for s in data if s.get('is_singing', False)]

    # All vocal segments = speech TTS + singing originals
    all_vocal_segments = speech_segments + singing_segments
    all_vocal_segments.sort(key=lambda x: x['start'])

    # Initialize mixed audio as instrumental base
    mixed = full_audio.clone()

    # Placement of vocal segments on instrumental
    for seg in all_vocal_segments:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)

        if not seg.get('is_singing', False):  # Speech TTS
            # Load TTS audio, resample to original SR, handle channels, trim/pad to exact duration
            seg_audio, tts_sr = torchaudio.load(seg['path'])
            # Resample if TTS SR differs from original (F5-TTS is 22050 Hz, original may be 44100 Hz)
            if tts_sr != sr:
                resampler = Resample(orig_freq=tts_sr, new_freq=sr)
                seg_audio = resampler(seg_audio)
            # Convert mono to stereo if needed
            if seg_audio.shape[0] == 1:
                seg_audio = seg_audio.repeat(2, 1)
            # Trim or pad to match exact segment duration
            target_len = end_sample - start_sample
            if seg_audio.shape[1] > target_len:
                seg_audio = seg_audio[:, :target_len]
            else:
                pad_len = target_len - seg_audio.shape[1]
                seg_audio = torch.nn.functional.pad(seg_audio, (0, pad_len), value=0)
            # Overlay TTS audio on mixed
            mixed[:, start_sample:end_sample] += seg_audio.to(device=device)
        else:  # Singing segment: overlay original vocal
            # Load original vocal segment audio
            seg_audio, orig_sr = torchaudio.load(seg['path'])
            # Resample if original SR differs
            if orig_sr != sr:
                resampler = Resample(orig_freq=orig_sr, new_freq=sr)
                seg_audio = resampler(seg_audio)
            # Convert mono to stereo if needed
            if seg_audio.shape[0] == 1:
                seg_audio = seg_audio.repeat(2, 1)
            # Trim or pad to match exact segment duration
            target_len = end_sample - start_sample
            if seg_audio.shape[1] > target_len:
                seg_audio = seg_audio[:, :target_len]
            else:
                pad_len = target_len - seg_audio.shape[1]
                seg_audio = torch.nn.functional.pad(seg_audio, (0, pad_len), value=0)
            # Overlay original singing audio on mixed
            mixed[:, start_sample:end_sample] += seg_audio.to(device=device)

    # Apply crossfades at boundaries between segments for smoothness (avoids clicks)
    fade_len_samples = int(crossfade_duration * sr)
    for i in range(1, len(all_vocal_segments)):
        boundary_sample = int(all_vocal_segments[i]['start'] * sr)
        # Fade out before boundary (on previous segment end)
        prev_end_sample = int(all_vocal_segments[i - 1]['end'] * sr)
        out_start = max(prev_end_sample - fade_len_samples, 0)
        out_end = prev_end_sample
        if out_start < out_end:
            fade_out = torch.linspace(1, 0, out_end - out_start).unsqueeze(0).repeat(2, 1).to(device)
            mixed[:, out_start:out_end] *= fade_out
        # Fade in after boundary (on next segment start)
        next_end_sample = int(all_vocal_segments[i]['end'] * sr)
        in_start = boundary_sample
        in_end = min(next_end_sample, boundary_sample + fade_len_samples)
        if in_start < in_end:
            fade_in = torch.linspace(0, 1, in_end - in_start).unsqueeze(0).repeat(2, 1).to(device)
            mixed[:, in_start:in_end] *= fade_in

    # Clamp to prevent clipping
    mixed = torch.clamp(mixed, -1.0, 1.0)

    # Save the mixed audio
    torchaudio.save(output_wav_path, mixed, sr)