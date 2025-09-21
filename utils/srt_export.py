"""
SRT export utilities for anime dubbing service.
Provides functionality to export transcription and translation data to SRT subtitle format.
"""

import os
from typing import List, Dict, Any, Optional


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (float)

    Returns:
        Time string in SRT format (e.g., "00:01:23,456")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def export_segments_to_srt(
    segments: List[Dict[str, Any]],
    output_path: str,
    text_field: str = "text",
    include_speaker: bool = False,
    title: Optional[str] = None,
    export_srt_directory: Optional[str] = None
) -> str:
    """
    Export segments to SRT subtitle format.

    Args:
        segments: List of segment dictionaries with start, end, and text fields
        output_path: Path where to save the SRT file
        text_field: Field name containing the text to export (e.g., "text", "translated_text")
        include_speaker: Whether to include speaker information in the subtitle text
        title: Optional title for the SRT file
        export_srt_directory: Optional custom directory for SRT export (overrides output_path directory)

    Returns:
        Path to the created SRT file

    Raises:
        ValueError: If required fields are missing from segments
        IOError: If unable to write to output_path
    """
    # Validate segments have required fields
    required_fields = ["start", "end", text_field]
    for i, segment in enumerate(segments):
        missing_fields = [field for field in required_fields if field not in segment]
        if missing_fields:
            raise ValueError(f"Segment {i} missing required fields: {missing_fields}")

    # Create SRT content
    srt_lines = []

    # Add title if provided
    if title:
        srt_lines.append(title)
        srt_lines.append("")

    # Process each segment
    for i, segment in enumerate(segments, 1):
        start_time = seconds_to_srt_time(segment["start"])
        end_time = seconds_to_srt_time(segment["end"])
        text = segment.get(text_field, "")

        # Include speaker information if requested
        if include_speaker and "speaker" in segment:
            speaker = segment["speaker"]
            text = f"[{speaker}] {text}"

        # Skip empty segments
        if not text.strip():
            continue

        # Format SRT entry
        srt_lines.extend([
            str(i),
            f"{start_time} --> {end_time}",
            text,
            ""
        ])

    # Handle custom export directory
    if export_srt_directory:
        os.makedirs(export_srt_directory, exist_ok=True)
        # Extract filename from output_path and combine with custom directory
        filename = os.path.basename(output_path)
        final_output_path = os.path.join(export_srt_directory, filename)
    else:
        final_output_path = output_path

    # Write to file
    try:
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        return final_output_path
    except IOError as e:
        raise IOError(f"Failed to write SRT file to {final_output_path}: {e}")


def export_transcription_to_srt(
    transcription_data: Dict[str, Any],
    output_path: str,
    include_speaker: bool = True,
    title: Optional[str] = None,
    export_srt_directory: Optional[str] = None
) -> str:
    """
    Export transcription data to SRT format.

    Args:
        transcription_data: Transcription stage output data
        output_path: Path where to save the SRT file
        include_speaker: Whether to include speaker information
        title: Optional title for the SRT file
        export_srt_directory: Optional custom directory for SRT export

    Returns:
        Path to the created SRT file
    """
    segments = transcription_data.get("segments", [])
    return export_segments_to_srt(
        segments=segments,
        output_path=output_path,
        text_field="text",
        include_speaker=include_speaker,
        title=title,
        export_srt_directory=export_srt_directory
    )


def export_translation_to_srt(
    translation_data: Dict[str, Any],
    output_path: str,
    text_field: str = "translated_text",
    include_speaker: bool = True,
    include_original: bool = False,
    title: Optional[str] = None,
    export_srt_directory: Optional[str] = None
) -> str:
    """
    Export translation data to SRT format.

    Args:
        translation_data: Translation stage output data
        output_path: Path where to save the SRT file
        text_field: Field to use for text ("translated_text" or "original_text")
        include_speaker: Whether to include speaker information
        include_original: Whether to include original text alongside translation
        title: Optional title for the SRT file
        export_srt_directory: Optional custom directory for SRT export

    Returns:
        Path to the created SRT file
    """
    segments = translation_data.get("segments", [])

    # Create enhanced segments with combined text if requested
    if include_original and text_field == "translated_text":
        enhanced_segments = []
        for segment in segments:
            enhanced_segment = segment.copy()
            original_text = segment.get("original_text", "")
            translated_text = segment.get("translated_text", "")

            if original_text and translated_text:
                enhanced_segment["translated_text"] = f"{translated_text}\n({original_text})"
            elif original_text:
                enhanced_segment["translated_text"] = f"({original_text})"
            elif translated_text:
                enhanced_segment["translated_text"] = translated_text

            enhanced_segments.append(enhanced_segment)
        segments = enhanced_segments

    return export_segments_to_srt(
        segments=segments,
        output_path=output_path,
        text_field=text_field,
        include_speaker=include_speaker,
        title=title,
        export_srt_directory=export_srt_directory
    )


def create_srt_filename(base_name: str, suffix: str = "subtitles", extension: str = "srt") -> str:
    """
    Create a standardized SRT filename.

    Args:
        base_name: Base name for the file
        suffix: Suffix to add before extension
        extension: File extension (default: "srt")

    Returns:
        Formatted filename
    """
    return f"{base_name}_{suffix}.{extension}"