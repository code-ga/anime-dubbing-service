#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import yaml
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.metadata import (create_metadata, is_complete, load_metadata,
                              load_previous_result, set_overall_error,
                              update_failure, update_success)
from utils.srt_export import export_segments_to_srt, export_translation_to_srt
from utils.burn_subtitles import burn_subtitles

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
        "--export-srt-directory",
        default="./srt",
        help="Directory path for exporting SRT subtitle files (default: ./srt)"
    )
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Keep temporary files after processing"
    )
    parser.add_argument(
        "--tts-method",
        default="xtts",
        choices=["xtts", "f5", "edge", "rvc"],
        help="Select the TTS method for voice generation (default: xtts)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Export subtitles in SRT format"
    )
    parser.add_argument(
        "--srt-text-field",
        default="translated_text",
        choices=["translated_text", "original_text"],
        help="Text field to use for SRT export (default: translated_text)"
    )
    parser.add_argument(
        "--srt-include-speaker",
        action="store_true",
        default=False,
        help="Include speaker information in SRT subtitles (default: False)"
    )
    parser.add_argument(
        "--srt-include-original",
        action="store_true",
        help="Include original text alongside translation in SRT (default: False)"
    )
    parser.add_argument(
        "--srt-title",
        type=str,
        help="Optional title for SRT file"
    )
    parser.add_argument(
        "--transcription-only",
        action="store_true",
        help="Generate video with burned-in subtitles only (no dubbing), preserving original audio"
    )
    parser.add_argument(
        "--skip-audio-separation",
        action="store_true",
        help="Skip audio separation stage for faster processing, using full audio track for transcription and mixing"
    )
    parser.add_argument(
        "--subtitle-font-size",
        type=int,
        default=24,
        help="Font size for burned-in subtitles (default: 24)"
    )
    parser.add_argument(
        "--subtitle-color",
        type=str,
        default="white",
        help="Color for burned-in subtitles (default: white)"
    )
    parser.add_argument(
        "--subtitle-position",
        type=str,
        default="bottom",
        choices=["bottom", "top", "middle"],
        help="Position for burned-in subtitles (default: bottom)"
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        default=False,
        help="Burn subtitles into the final video (default: False)"
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
    
    # Filter stages based on transcription_only flag, skip_audio_separation flag and conditions
    all_stages = config_data["stages"]
    stages = []
    for stage in all_stages:
        condition = stage.get("condition")
        if condition:
            # Evaluate condition based on flags
            if condition == "not transcription_only" and args.transcription_only:
                continue  # Skip this stage
            elif condition == "transcription_only" and not args.transcription_only:
                continue  # Skip this stage
            elif condition == "not skip_audio_separation" and args.skip_audio_separation:
                continue  # Skip this stage
        stages.append(stage)

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

                # Set current_stage before processing to ensure error handling matches
                metadata["current_stage"] = stage
                temp_path = metadata_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                os.replace(temp_path, metadata_path)

                # Log stage start
                logger.log_stage_start(stage, stage_idx, len(stages))

                logger.logger.info(f"🔧 Executing stage: {stage}")
                logger.logger.info(f"📦 Module: {module_path}.{func_name}")
                logger.logger.info(f"📥 Inputs: {inputs}")

                stage_data = None
                try:
                    # Load input data from previous stages, handling optional/missing inputs gracefully
                    inputs_data = {}
                    for inp in inputs:
                        try:
                            inputs_data[inp] = load_previous_result(metadata_path, inp)
                        except ValueError as ve:
                            if "result not found" in str(ve):
                                logger.logger.warning(f"Optional input '{inp}' for stage '{stage}' not available (skipped stage); setting to None for fallback handling.")
                                inputs_data[inp] = None
                            else:
                                raise

                    # Dynamic import and call
                    module_obj = importlib.import_module(module_path)
                    func = getattr(module_obj, func_name)

                    # Call function with stage-specific args
                    if stage == "convert_mp4_to_wav":
                        stage_data = func(
                            tmp_path, metadata_path, inputs_data, input_file=input_abs
                        )
                    elif stage == "translate":
                        # Determine appropriate text field for SRT export
                        srt_text_field = args.srt_text_field
                        # Auto-enable export_srt if burn_subtitles is True
                        export_srt = args.export_srt or args.burn_subtitles

                        if args.transcription_only or args.burn_subtitles:
                            # In transcription-only mode or when burning subtitles, use translated_text if target_lang is set, otherwise original_text
                            if args.target_lang and args.target_lang != "en":  # Assuming "en" might be default but no translation
                                srt_text_field = "translated_text"
                            else:
                                srt_text_field = "original_text"

                        stage_data = func(
                            tmp_path,
                            metadata_path,
                            inputs_data,
                            target_lang=args.target_lang,
                            singing_model=args.singing_model,
                            export_srt=export_srt,
                            srt_text_field=srt_text_field,
                            srt_include_speaker=args.srt_include_speaker,
                            srt_include_original=args.srt_include_original,
                            srt_title=args.srt_title,
                            export_srt_directory=args.export_srt_directory,
                        )
                    elif stage == "mux_video":
                        stage_data = func(
                            tmp_path,
                            metadata_path,
                            inputs_data,
                            input_file=input_abs,
                            output_file=os.path.abspath(args.output_mp4),
                        )
                    elif stage == "burn_subtitles":
                        stage_data = func(
                            tmp_path,
                            metadata_path,
                            inputs_data,
                            original_video_path=input_abs,
                            output_file=os.path.abspath(args.output_mp4),
                            font_size=args.subtitle_font_size,
                            color=args.subtitle_color,
                            position=args.subtitle_position,
                        )
                    elif stage == "mix_audio":
                        stage_data = func(tmp_path, metadata_path, inputs_data)
                    elif stage == "generate_tts":
                        stage_data = func(tmp_path, metadata_path, inputs_data, tts_method=args.tts_method)
                    else:
                        stage_data = func(tmp_path, metadata_path, inputs_data)

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

                    # Handle burn_subtitles after mux_video stage
                    if stage == "mux_video" and args.burn_subtitles:
                        logger.logger.info("🔥 Burning subtitles into video as requested...")

                        # Determine appropriate text field for subtitles
                        subtitle_text_field = "translated_text" if args.target_lang and args.target_lang != "en" else "original_text"

                        # Find the SRT file from translate stage
                        translate_data = load_previous_result(metadata_path, "translate")
                        srt_file = None
                        if translate_data and "srt_file" in translate_data:
                            srt_file = translate_data["srt_file"]
                        elif translate_data and "stage_data" in translate_data and "srt_file" in translate_data["stage_data"]:
                            srt_file = translate_data["stage_data"]["srt_file"]

                        if not srt_file:
                            logger.logger.error("No SRT file found for subtitle burning")
                            raise ValueError("No SRT file available for subtitle burning")

                        srt_path = os.path.join(tmp_path, srt_file)
                        if not os.path.exists(srt_path):
                            logger.logger.error(f"SRT file not found: {srt_path}")
                            raise FileNotFoundError(f"SRT file not found: {srt_path}")

                        # Generate output path for subtitled video
                        output_basename = os.path.basename(args.output_mp4)
                        name, ext = os.path.splitext(output_basename)
                        subtitled_output = os.path.join(os.path.dirname(args.output_mp4), f"{name}_subtitled{ext}")

                        # Call burn_subtitles function
                        burn_stage_data = None
                        try:
                            burn_stage_data = burn_subtitles(
                                tmp_path=tmp_path,
                                metadata_path=metadata_path,
                                inputs_data={"translate": translate_data},
                                original_video_path=args.input_mp4,
                                output_file=subtitled_output,
                                font_size=args.subtitle_font_size,
                                color=args.subtitle_color,
                                position=args.subtitle_position
                            )

                            # Update success with burn_subtitles data
                            update_success(
                                metadata_path, "burn_subtitles", start_time, None, stage_data=burn_stage_data
                            )

                            # Update the final output path to the subtitled version
                            args.output_mp4 = subtitled_output
                            logger.logger.info(f"✅ Subtitles burned successfully. Final output: {subtitled_output}")

                        except Exception as burn_e:
                            logger.log_error("burn_subtitles", burn_e, "subtitle burning")
                            raise

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

    # Generate SRT files directly from translate stage data
    translate_data = stage_results.get("translate", {})
    if translate_data:
        actual_stage_data = {}
        if 'json_path' in translate_data:
            json_path = os.path.join(tmp_path, translate_data['json_path'])
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    actual_stage_data = json.load(f)
        
        # Fallback to old way if no json_path
        stage_data = actual_stage_data or translate_data.get("stage_data", {})
        
        # Prioritize actual_stage_data, then root, then stage_data
        segments = actual_stage_data.get("segments", translate_data.get("segments", stage_data.get("segments", [])))
        target_lang = actual_stage_data.get("target_lang", translate_data.get("target_lang", stage_data.get("target_lang", "en")))
        
        if not segments:
            print("⚠️  No segments found in translate data, skipping SRT export")
        else:
            # Infer source language from actual_stage_data or segments or default to 'ja'
            source_lang = "ja"  # default
            if actual_stage_data.get("language"):
                source_lang = actual_stage_data["language"]
            elif segments and "language" in segments[0]:
                source_lang = segments[0].get("language", "ja")
            elif translate_data.get("language") or stage_data.get("language"):
                source_lang = translate_data.get("language", stage_data.get("language", "ja"))
            
            # Check if speaker information should be included, prioritize actual_stage_data
            include_speaker = actual_stage_data.get("srt_include_speaker", translate_data.get("srt_include_speaker", stage_data.get("srt_include_speaker", False)))
            
            try:
                # Generate translated subtitles SRT file
                translated_srt_filename = f"translated_subtitles_{target_lang}.srt"
                translated_srt_path = os.path.join(results_dir, translated_srt_filename)

                export_translation_to_srt(
                    translation_data=actual_stage_data or stage_data,
                    output_path=translated_srt_path,
                    text_field="translated_text",
                    include_speaker=include_speaker,
                    title=f"Translated Subtitles ({target_lang.upper()})"
                )
                print(f"✅ Exported translated SRT file: {translated_srt_filename}")

                # Generate original transcription SRT file
                original_srt_filename = f"original_transcription_{source_lang}.srt"
                original_srt_path = os.path.join(results_dir, original_srt_filename)

                export_segments_to_srt(
                    segments=segments,
                    output_path=original_srt_path,
                    text_field="original_text",
                    include_speaker=include_speaker,
                    title=f"Original Transcription ({source_lang.upper()})"
                )
                print(f"✅ Exported original SRT file: {original_srt_filename}")

            except Exception as e:
                print(f"⚠️  Error generating SRT files: {e}")
                # For backward compatibility, try to copy legacy files if they exist
                legacy_srt_filename = (actual_stage_data or stage_data).get("srt_file")
                if legacy_srt_filename:
                    legacy_srt_path = os.path.join(tmp_path, legacy_srt_filename)
                    if os.path.exists(legacy_srt_path):
                        shutil.copy(legacy_srt_path, os.path.join(results_dir, legacy_srt_filename))
                        print(f"✅ Copied legacy SRT file: {legacy_srt_filename}")

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
