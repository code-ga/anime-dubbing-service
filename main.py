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

from datetime import datetime
from utils.metadata import (
    load_metadata,
    create_metadata,
    update_success,
    update_failure,
    set_overall_error,
    is_complete
)

load_dotenv("./.env")


def main():
    parser = argparse.ArgumentParser(
        description="Anime Dubbing Service with Music Preservation"
    )
    parser.add_argument("input_mp4", help="Input MP4 file path")
    parser.add_argument("output_mp4", help="Output MP4 file path")
    parser.add_argument(
        "--music_threshold",
        type=float,
        default=0.6,
        help="Music detection threshold (hardcoded in whisper.py)",
    )
    parser.add_argument("--target_lang", default="en", help="Target language")
    parser.add_argument(
        "--singing_model",
        default="openai/gpt-oss-120b",
        help="LLM model for singing detection",
    )
    parser.add_argument("--tmp-dir", default="./tmp", help="Temporary directory path")
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Keep temporary files after processing"
    )
    args = parser.parse_args()

    tmp_dir = os.environ.get("TMP_DIR", args.tmp_dir)
    tmp_path = os.path.abspath(tmp_dir)
    parent_dir = os.path.dirname(tmp_path) or "."
    if not os.access(parent_dir, os.W_OK):
        raise ValueError(f"Cannot write to parent directory of tmp: {parent_dir}")
    os.makedirs(tmp_path, exist_ok=True)
    transcript_dir = os.path.join(tmp_path, "transcript")
    translated_dir = os.path.join(tmp_path, "translated")
    refs_dir = os.path.join(tmp_path, "refs")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    metadata_path = os.path.join(tmp_path, "metadata.json")
    input_abs = os.path.abspath(args.input_mp4)
    metadata = load_metadata(metadata_path)
    if metadata:
        if metadata["input_file"] != input_abs:
            os.remove(metadata_path)
            metadata = None
        elif is_complete(metadata):
            print(f"Workflow already complete: {args.output_mp4}")
            if not args.keep_tmp:
                for item in os.listdir(tmp_path):
                    if item == "metadata.json":
                        continue
                    item_path = os.path.join(tmp_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            sys.exit(0)
    if metadata is None:
        metadata = create_metadata(
            metadata_path, input_abs, os.path.abspath(args.output_mp4), tmp_path, args.target_lang
        )
    # Validate completed stages outputs
    for i, entry in enumerate(metadata["completed_stages"]):
        stage = entry["stage"]
        outputs = entry.get("output_files", {})
        invalid = False
        for rel_path in outputs.values():
            full_path = os.path.join(tmp_path, rel_path)
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                invalid = True
                break
        if invalid:
            metadata["completed_stages"] = metadata["completed_stages"][:i]
            metadata["current_stage"] = stage
            temp_path = metadata_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            os.replace(temp_path, metadata_path)
            break

    try:
        current_stage = metadata["current_stage"]
        while current_stage != "complete":
            if current_stage == "convert_mp4_to_wav":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    original_wav = os.path.join(tmp_path, "full.wav")
                    convert_mp4_to_wav(args.input_mp4, original_wav)
                    outputs = {"full_wav": "full.wav"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "separate_audio":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    original_wav = os.path.join(tmp_path, "full.wav")
                    vocals_path, instrumental_path = separate(original_wav, tmp_path)
                    outputs = {"vocals": "vocals.wav", "instrumental": "accompaniment.wav"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "transcribe":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    vocals_path = os.path.join(tmp_path, "vocals.wav")
                    audio_transcript = transcript(vocals_path, tmp_path=tmp_path)
                    transcript_path = os.path.join(transcript_dir, "transcript.json")
                    with open(transcript_path, "w") as f:
                        json.dump(audio_transcript, f, indent=4)
                    outputs = {"transcript_json": "transcript/transcript.json"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                    raise Exception("Simulated failure after transcription for testing resumption")
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "translate":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    transcript_path = os.path.join(transcript_dir, "transcript.json")
                    with open(transcript_path, "r") as f:
                        audio_transcript = json.load(f)
                    translated = translate_with_openai(
                        audio_transcript,
                        tmp_path,
                        args.target_lang,
                        singing_model=args.singing_model,
                    )
                    translated_path = os.path.join(translated_dir, "translated.json")
                    with open(translated_path, "w") as f:
                        json.dump(translated, f, indent=4)
                    outputs = {"translated_json": "translated/translated.json"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "build_refs":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    ref_audios_by_speaker = {}
                    translated_path = os.path.join(translated_dir, "translated.json")
                    with open(translated_path, "r") as f:
                        translated_data = json.load(f)
                    vocals_path = os.path.join(tmp_path, "vocals.wav")
                    waveform, sample_rate = torchaudio.load(vocals_path)
                    speakers = set(
                        seg.get("speaker", "default") for seg in translated_data.get("segments", [])
                    )
                    for speaker in speakers:
                        non_singing_segs = [
                            seg
                            for seg in translated_data["segments"]
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

                    default_ref = None
                    if ref_audios_by_speaker:
                        first_speaker = list(ref_audios_by_speaker.keys())[0]
                        default_ref_path = os.path.join(refs_dir, "default.wav")
                        shutil.copy(ref_audios_by_speaker[first_speaker], default_ref_path)
                        default_ref = default_ref_path

                    outputs = {"refs_dir": "refs"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "generate_tts":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    refs_dir = os.path.join(tmp_path, "refs")
                    ref_files = [f for f in os.listdir(refs_dir) if f.endswith('.wav') and f != 'default.wav']
                    ref_audios_by_speaker = {os.path.splitext(f)[0]: os.path.join(refs_dir, f) for f in ref_files}
                    default_ref = os.path.join(refs_dir, "default.wav") if os.path.exists(os.path.join(refs_dir, "default.wav")) else None
                    tts_segments = generate_dubbed_segments(
                        tmp_path, ref_audios_by_speaker, default_ref
                    )
                    outputs = {"tts_dir": "tts"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "mix_audio":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    dubbed_wav = os.path.join(tmp_path, "dubbed.wav")
                    mix_audio(tmp_path, dubbed_wav)
                    outputs = {"dubbed_wav": "dubbed.wav"}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            elif current_stage == "mux_video":
                start_time = datetime.utcnow().isoformat() + "Z"
                try:
                    dubbed_wav = os.path.join(tmp_path, "dubbed.wav")
                    cmd = [
                        "ffmpeg",
                        "-i",
                        args.input_mp4,
                        "-i",
                        dubbed_wav,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-y",
                        args.output_mp4,
                    ]
                    subprocess.run(cmd, check=True)
                    outputs = {}
                    update_success(metadata_path, current_stage, start_time, outputs)
                except Exception as stage_e:
                    update_failure(metadata_path, current_stage, start_time, str(stage_e))
                    raise
            else:
                raise ValueError(f"Unknown current_stage: {current_stage}")

            metadata = load_metadata(metadata_path)
            current_stage = metadata["current_stage"]

        print(f"Successfully created {args.output_mp4}")

        if not args.keep_tmp:
            for item in os.listdir(tmp_path):
                if item == "metadata.json":
                    continue
                item_path = os.path.join(tmp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
    except Exception as e:
        if 'metadata_path' in locals():
            set_overall_error(metadata_path, str(e))
        print(f"Error in processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
