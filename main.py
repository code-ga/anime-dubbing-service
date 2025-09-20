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

import yaml
import importlib

from datetime import datetime
from utils.metadata import (
    load_metadata,
    create_metadata,
    update_success,
    update_failure,
    set_overall_error,
    is_complete,
    load_previous_result,
)
from utils.logger import get_logger

load_dotenv("./.env")


def main():
    # Initialize logger
    logger = get_logger("anime-dubbing-pipeline")

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
    parser.add_argument(
        "--tts-method",
        default="edge-tts",
        choices=["edge-tts", "f5-tts"],
        help="TTS method to use for audio generation (default: edge-tts)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    # Update logger level based on argument
    logger.logger.setLevel(getattr(__import__('logging'), args.log_level.upper()))

    tmp_dir = os.environ.get("TMP_DIR", args.tmp_dir)
    tmp_path = os.path.abspath(tmp_dir)
    parent_dir = os.path.dirname(tmp_path) or "."
    if not os.access(parent_dir, os.W_OK):
        logger.log_error("setup", ValueError(f"Cannot write to parent directory of tmp: {parent_dir}"))
        raise ValueError(f"Cannot write to parent directory of tmp: {parent_dir}")
    os.makedirs(tmp_path, exist_ok=True)
    transcript_dir = os.path.join(tmp_path, "transcript")
    translated_dir = os.path.join(tmp_path, "translated")
    refs_dir = os.path.join(tmp_path, "refs")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    logger.log_file_operation("create", tmp_path, True)

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
            metadata_path,
            input_abs,
            os.path.abspath(args.output_mp4),
            tmp_path,
            args.target_lang,
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
            temp_path = metadata_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(metadata, f, indent=4)
            os.replace(temp_path, metadata_path)
            break

    # Load stages config
    config_path = os.path.join(".", "config", "stages.yaml")
    if not os.path.exists(config_path):
        logger.log_error("config", FileNotFoundError(f"Stages config not found: {config_path}"))
        raise FileNotFoundError(f"Stages config not found: {config_path}")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    stages = config_data["stages"]

    # Log pipeline start (now that we have stages)
    logger.log_pipeline_start(input_abs, os.path.abspath(args.output_mp4), len(stages))

    def validate_prerequisites(metadata_path, stages, metadata):
        current_stage = metadata["current_stage"]
        current_idx = next(
            (
                i
                for i, stage_info in enumerate(stages)
                if stage_info["name"] == current_stage
            ),
            len(stages),
        )
        for i in range(current_idx):
            stage_name = stages[i]["name"]
            if stage_name not in metadata["stage_results"]:
                logger.log_warning("prerequisites", f"Missing prerequisite stage '{stage_name}'. Resetting current_stage to '{stage_name}'.", "stage validation")
                metadata["current_stage"] = stage_name
                temp_path = metadata_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                os.replace(temp_path, metadata_path)
                return i
            else:
                json_path = metadata["stage_results"][stage_name]["json_path"]
                if not os.path.exists(json_path):
                    logger.log_warning("prerequisites", f"Missing result file for stage '{stage_name}'. Resetting current_stage to '{stage_name}'.", "file validation")
                    metadata["current_stage"] = stage_name
                    temp_path = metadata_path + ".tmp"
                    with open(temp_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                    os.replace(temp_path, metadata_path)
                    return i
        return current_idx

    try:
        # Validate prerequisites and potentially reset current_stage
        current_idx = validate_prerequisites(metadata_path, stages, metadata)
        current_stage = metadata["current_stage"]

        if current_idx == len(stages):
            logger.logger.info("✅ All stages completed.")
        else:
            for stage_idx in range(current_idx, len(stages)):
                stage_info = stages[stage_idx]
                stage = stage_info["name"]
                module_path = stage_info["module"]
                func_name = stage_info["function"]
                inputs = stage_info["inputs"]
                start_time = datetime.utcnow().isoformat() + "Z"

                # Log stage start
                logger.log_stage_start(stage, stage_idx, len(stages))

                logger.logger.info(f"🔧 Executing stage: {stage}")
                logger.logger.info(f"📦 Module: {module_path}.{func_name}")
                logger.logger.info(f"📥 Inputs: {inputs}")

                stage_data = None
                try:
                    # Load input data from previous stages
                    inputs_data = {}
                    for inp in inputs:
                        inputs_data[inp] = load_previous_result(metadata_path, inp)

                    # Dynamic import and call
                    module_obj = importlib.import_module(module_path)
                    func = getattr(module_obj, func_name)

                    # Call function with stage-specific args
                    if stage == "convert_mp4_to_wav":
                        stage_data = func(
                            tmp_path, metadata_path, inputs_data, input_file=input_abs
                        )
                    elif stage == "translate":
                        stage_data = func(
                            tmp_path,
                            metadata_path,
                            inputs_data,
                            target_lang=args.target_lang,
                            singing_model=args.singing_model,
                        )
                    elif stage == "mux_video":
                        stage_data = func(
                            tmp_path,
                            metadata_path,
                            inputs_data,
                            input_file=input_abs,
                            output_file=os.path.abspath(args.output_mp4),
                        )
                    else:
                        stage_data = func(tmp_path, metadata_path, inputs_data, tts_method=args.tts_method)

                    print(stage_data)

                    # Update success with stage_data
                    update_success(
                        metadata_path, stage, start_time, None, stage_data=stage_data
                    )

                    # Log stage completion
                    logger.log_stage_completion(stage, stage_idx, len(stages))

                    # Log stage-specific information
                    if stage_data:
                        if "tts_method" in stage_data:
                            logger.log_tts_method(stage_data["tts_method"])
                        if "total_duration" in stage_data:
                            logger.logger.info(f"  ⏱️  Stage output duration: {stage_data['total_duration']:.2f} seconds")

                except Exception as stage_e:
                    # Log error with context
                    logger.log_error(stage, stage_e, "stage execution")

                    # Partial outputs if available
                    partial_outputs = (
                        stage_data.get("output_files", {}) if stage_data else {}
                    )
                    update_failure(
                        metadata_path, stage, start_time, str(stage_e), partial_outputs
                    )
                    raise

        # Save final results
        logger.logger.info("💾 Saving final results...")
        save_final_results(tmp_path, metadata_path, args.output_mp4)

        output_file = args.output_mp4
        logger.logger.info(f"✅ Successfully created {output_file}")
        logger.log_file_operation("create", output_file, True)

        # Log pipeline completion
        total_duration = logger.get_elapsed_time()
        logger.log_pipeline_completion(total_duration, success=True)

        if not args.keep_tmp:
            logger.logger.info("🧹 Cleaning up temporary files...")
            for item in os.listdir(tmp_path):
                if item == "metadata.json":
                    continue
                item_path = os.path.join(tmp_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.logger.info("✅ Temporary files cleaned up")
    except Exception as e:
        # Log overall error
        logger.log_error("pipeline", e, "overall execution")

        if "metadata_path" in locals():
            set_overall_error(metadata_path, str(e))

        # Log pipeline completion with failure
        total_duration = logger.get_elapsed_time()
        logger.log_pipeline_completion(total_duration, success=False)

        logger.logger.error(f"💥 Pipeline failed: {e}")
        sys.exit(1)


def save_final_results(tmp_path, metadata_path, output_file):
    metadata = load_metadata(metadata_path)
    if not metadata:
        raise ValueError("Metadata not found")

    timestamp = (
        metadata["timestamps"]["workflow_start"]
        .replace(":", "")
        .replace(".", "")
        .replace("Z", "")
    )
    results_dir = os.path.join("./results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Copy output video
    output_basename = os.path.basename(output_file)
    shutil.copy(output_file, os.path.join(results_dir, f"dubbed_{output_basename}"))

    # Copy metadata
    shutil.copy(metadata_path, os.path.join(results_dir, "metadata.json"))

    # Copy key JSONs
    stage_results = metadata.get("stage_results", {})
    key_stages = {
        "transcribe.json": "transcribe",
        "translated.json": "translate",
        "tts.json": "generate_tts",
    }
    for dest_name, stage_name in key_stages.items():
        json_path = stage_results.get(stage_name, {}).get("json_path")
        if json_path and os.path.exists(json_path):
            shutil.copy(json_path, os.path.join(results_dir, dest_name))

    # Extract and save diarization embeddings
    transcribe_path = stage_results.get("transcribe", {}).get("json_path")
    if transcribe_path and os.path.exists(transcribe_path):
        with open(transcribe_path, "r") as f:
            transcribe_data = json.load(f)
        embeddings = transcribe_data.get("speaker_embeddings", {})
        emb_path = os.path.join(results_dir, "diarization_embeddings.json")
        with open(emb_path, "w") as f:
            json.dump({"speakers": embeddings}, f, indent=4)


if __name__ == "__main__":
    main()


def mux_video(
    tmp_path, metadata_path, inputs_data, input_file, output_file, **kwargs
) -> dict:
    """
    Mux dubbed audio with original video using FFmpeg.

    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data.
        input_file: Original MP4.
        output_file: Output MP4.
        **kwargs: Additional arguments.

    Returns:
        Stage data with final_video_path, mux_params.
    """
    mix_data = inputs_data["mix_audio"]
    dubbed_wav = os.path.join(tmp_path, mix_data["dubbed_wav_path"])

    cmd = [
        "ffmpeg",
        "-i",
        input_file,
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
        output_file,
    ]
    subprocess.run(cmd, check=True)

    stage_data = {
        "stage": "mux_video",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "final_video_path": os.path.basename(output_file),
        "mux_params": {"video_codec": "copy", "audio_codec": "aac"},
    }

    return stage_data
