"""
Utils package for anime dubbing service.
Contains utility modules for logging, metadata management, and other common functionality.
"""

from .logger import get_logger, PipelineLogger, setup_console_logging
from .metadata import (
    load_metadata,
    create_metadata,
    update_success,
    update_failure,
    set_overall_error,
    is_complete,
    load_previous_result,
    save_stage_result,
    STAGES_ORDER
)
from .srt_export import (
    export_segments_to_srt,
    export_transcription_to_srt,
    export_translation_to_srt,
    create_srt_filename,
    seconds_to_srt_time
)

__all__ = [
    'get_logger',
    'PipelineLogger',
    'setup_console_logging',
    'load_metadata',
    'create_metadata',
    'update_success',
    'update_failure',
    'set_overall_error',
    'is_complete',
    'load_previous_result',
    'save_stage_result',
    'STAGES_ORDER',
    'export_segments_to_srt',
    'export_transcription_to_srt',
    'export_translation_to_srt',
    'create_srt_filename',
    'seconds_to_srt_time'
]