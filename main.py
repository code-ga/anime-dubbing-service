#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
import torchaudio
import sys
import shutil
from dotenv import load_dotenv
from convert.mp4_wav import convert_mp4_to_wav
from transcription.whisper import transcript
from translate.openAi import translate_with_openai
from tts.orchestrator import generate_dubbed_segments
from dub.mixer import mix_audio
from convert.separate_audio import separate

load_dotenv("./.env")

def main():
    parser = argparse.ArgumentParser(description="Anime Dubbing Service with Music Preservation")
    parser.add_argument("input_mp4", help="Input MP4 file path")
    parser.add_argument("output_mp4", help="Output MP4 file path")
    parser.add_argument("--music_threshold", type=float, default=0.6, help="Music detection threshold (hardcoded in whisper.py)")
    parser.add_argument("--target_lang", default="en", help="Target language")
    parser.add_argument("--singing_model", default="gpt-3.5-turbo", help="LLM model for singing detection")
    parser.add_argument("--tmp-dir", default="./tmp", help="Temporary directory path")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary files after processing")
    args = parser.parse_args()

    tmp_dir = os.environ.get('TMP_DIR', args.tmp_dir)
    tmp_path = os.path.abspath(tmp_dir)
    parent_dir = os.path.dirname(tmp_path) or '.'
    if not os.access(parent_dir, os.W_OK):
        raise ValueError(f"Cannot write to parent directory of tmp: {parent_dir}")
    os.makedirs(tmp_path, exist_ok=True)
    transcript_dir = os.path.join(tmp_path, "transcript")
    translated_dir = os.path.join(tmp_path, "translated")
    refs_dir = os.path.join(tmp_path, "refs")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    try:
        # Convert MP4 to WAV
        original_wav = os.path.join(tmp_path, "full.wav")
        convert_mp4_to_wav(args.input_mp4, original_wav)

        vocals_path, instrumental_path = separate(original_wav, tmp_path)

        # Transcribe
        audio_transcript = transcript(vocals_path, tmp_path=tmp_path)
        transcript_path = os.path.join(transcript_dir, "transcript.json")

        # Translate
        translated = translate_with_openai(audio_transcript, tmp_path, args.target_lang, singing_model=args.singing_model)
        translated_path = os.path.join(translated_dir, "translated.json")

        # Build reference audios
        ref_audios_by_speaker = {}
        with open(translated_path, "r") as f:
            translated_data = json.load(f)
        waveform, sample_rate = torchaudio.load(vocals_path)
        speakers = set(seg.get("speaker", "default") for seg in translated_data.get("segments", []))
        for speaker in speakers:
            non_singing_segs = [
                seg for seg in translated_data["segments"]
                if seg.get("speaker") == speaker and not seg.get("is_singing", False)
            ]
            if non_singing_segs:
                first_seg = non_singing_segs[0]
                start = first_seg["start"]
                start_sample = int((start + 0.5) * sample_rate)
                duration = 4.0
                end_sample = int((start + 0.5 + duration) * sample_rate)
                if end_sample > waveform.shape[1]:
                    end_sample = waveform.shape[1]
                ref_waveform = waveform[:, start_sample:end_sample]
                ref_path = os.path.join(refs_dir, f"{speaker}.wav")
                torchaudio.save(ref_path, ref_waveform, sample_rate)
                ref_audios_by_speaker[speaker] = ref_path

        # Default ref
        default_ref = None
        if ref_audios_by_speaker:
            first_speaker = list(ref_audios_by_speaker.keys())[0]
            default_ref_path = os.path.join(refs_dir, "default.wav")
            shutil.copy(ref_audios_by_speaker[first_speaker], default_ref_path)
            default_ref = default_ref_path

        # Generate dubbed segments
        tts_segments = generate_dubbed_segments(tmp_path, ref_audios_by_speaker, default_ref)

        # Mix audio
        dubbed_wav = os.path.join(tmp_path, "dubbed.wav")
        mix_audio(tmp_path, dubbed_wav)

        # Mux with ffmpeg
        cmd = [
            "ffmpeg",
            "-i", args.input_mp4,
            "-i", dubbed_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-y",
            args.output_mp4
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully created {args.output_mp4}")

        # Optional cleanup
        if not args.keep_tmp:
            shutil.rmtree(tmp_path)

    except Exception as e:
        print(f"Error in processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
