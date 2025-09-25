"""
Subtitle burning utilities for anime dubbing service.
Provides functionality to burn SRT subtitles directly into video using FFmpeg.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Dict, Any
from utils.logger import get_logger


def burn_subtitles(
    tmp_path: str,
    metadata_path: str,
    inputs_data: Dict[str, Any],
    original_video_path: str,
    output_file: str,
    font_size: int = 24,
    color: str = "white",
    position: str = "bottom"
) -> Dict[str, Any]:
    """
    Burn subtitles into video using FFmpeg.

    Args:
        tmp_path: Path to temporary directory
        metadata_path: Path to metadata file
        inputs_data: Dict containing previous stage data (translate stage)
        original_video_path: Path to original video file
        output_file: Path to output video file
        font_size: Font size for subtitles
        color: Color for subtitles (white, yellow, etc.)
        position: Position for subtitles (bottom, top, middle)

    Returns:
        Dict containing stage data with output paths and parameters

    Raises:
        FileNotFoundError: If SRT file is missing
        subprocess.CalledProcessError: If FFmpeg fails
        ValueError: If invalid parameters provided
    """
    logger = get_logger("burn_subtitles")

    # Get translate data to find SRT file
    translate_data = inputs_data.get("translate", {})
    if not translate_data:
        raise ValueError("No translate data found in inputs")

    # Look for SRT file in translate stage data
    srt_file = None
    if "srt_file" in translate_data:
        srt_file = translate_data["srt_file"]
    elif "stage_data" in translate_data and "srt_file" in translate_data["stage_data"]:
        srt_file = translate_data["stage_data"]["srt_file"]

    if not srt_file:
        raise FileNotFoundError("No SRT file found in translate stage data")

    srt_path = os.path.join(tmp_path, srt_file)
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    logger.logger.info(f"Burning subtitles from: {srt_path}")
    logger.logger.info(f"Original video: {original_video_path}")
    logger.logger.info(f"Output video: {output_file}")
    logger.logger.info(f"Subtitle settings: font_size={font_size}, color={color}, position={position}")

    # Validate parameters
    if font_size <= 0 or font_size > 200:
        raise ValueError(f"Invalid font size: {font_size} (must be between 1-200)")
    if position not in ["bottom", "top", "middle"]:
        raise ValueError(f"Invalid position: {position} (must be bottom, top, or middle)")

    # Validate input files exist
    if not os.path.exists(original_video_path):
        raise FileNotFoundError(f"Original video file not found: {original_video_path}")

    # Check if output directory is writable
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to output directory: {output_dir}")

    # Map position to FFmpeg subtitle position
    position_map = {
        "bottom": "10",
        "top": "10",
        "middle": "H/2"
    }
    y_position = position_map[position]

    # Map color to FFmpeg format (basic colors)
    color_map = {
        "white": "&Hffffff",
        "yellow": "&Hffff00",
        "black": "&H000000",
        "red": "&Hff0000",
        "green": "&H00ff00",
        "blue": "&H0000ff"
    }
    ffmpeg_color = color_map.get(color.lower(), "&Hffffff")

    # Validate color
    if color.lower() not in color_map and not color.startswith("&H"):
        logger.logger.warning(f"Unknown color '{color}', defaulting to white")
        ffmpeg_color = "&Hffffff"

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", original_video_path,
        "-vf", f"subtitles={srt_path}:force_style='FontSize={font_size},PrimaryColour={ffmpeg_color},Alignment=2,MarginV={y_position}'",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-preset", "medium",
        "-crf", "23",
        "-y", output_file
    ]

    # Check if FFmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg is not installed or not available in PATH")

    logger.logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

    try:
        # Run FFmpeg command
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        logger.logger.info("FFmpeg completed successfully")
        logger.logger.debug(f"FFmpeg stdout: {result.stdout}")

        if result.stderr:
            logger.logger.debug(f"FFmpeg stderr: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.logger.error(f"FFmpeg failed with return code {e.returncode}")
        logger.logger.error(f"FFmpeg stderr: {e.stderr}")
        if e.returncode == 1 and "No such file or directory" in str(e.stderr):
            raise FileNotFoundError(f"FFmpeg could not find input file: {original_video_path}")
        elif e.returncode == 1 and "subtitle" in str(e.stderr).lower():
            raise ValueError(f"FFmpeg subtitle filter error. Check SRT file format: {srt_path}")
        else:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.logger.error("FFmpeg timed out after 1 hour")
        raise RuntimeError("FFmpeg operation timed out")

    # Verify output file was created
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output video file was not created: {output_file}")

    output_size = os.path.getsize(output_file)
    logger.logger.info(f"Output video created successfully: {output_size} bytes")

    # Prepare stage data
    stage_data = {
        "stage": "burn_subtitles",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "subtitled_video_path": os.path.basename(output_file),
        "burn_params": {
            "font_size": font_size,
            "color": color,
            "position": position,
            "srt_source": srt_file
        },
        "output_file_size": output_size
    }

    # Save stage results to JSON file
    results_dir = os.path.join(tmp_path, "burn_subtitles_results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "burn_subtitles.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stage_data, f, indent=2, ensure_ascii=False)

    logger.logger.info(f"Stage results saved to: {json_path}")

    return stage_data


def get_subtitle_position_code(position: str) -> str:
    """
    Get FFmpeg subtitle positioning code.

    Args:
        position: Position string (bottom, top, middle)

    Returns:
        FFmpeg positioning code
    """
    position_codes = {
        "bottom": "Alignment=2,MarginV=10",  # Bottom center
        "top": "Alignment=8,MarginV=10",     # Top center
        "middle": "Alignment=5,MarginV=10"   # Middle center
    }
    return position_codes.get(position, "Alignment=2,MarginV=10")