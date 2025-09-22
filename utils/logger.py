import logging
import sys
import time
from datetime import datetime
import traceback
from typing import Optional, Dict, Any


class PipelineLogger:
    """
    Comprehensive logging utility for the anime dubbing pipeline.
    Provides structured logging with progress tracking, timing, and error handling.
    """

    def __init__(self, name: str = "anime-dubbing", log_level: str = "INFO"):
        """
        Initialize the pipeline logger.

        Args:
            name: Logger name for identification
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.start_time = time.time()
        self.stage_start_times: Dict[str, float] = {}
        self.segment_counts: Dict[str, int] = {}
        self.operation_times: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create console handler with custom formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # Custom formatter with timestamps and colors
        formatter = PipelineFormatter()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def log_pipeline_start(self, input_file: str, output_file: str, total_stages: int):
        """Log the start of the entire pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("🎬 ANIME DUBBING PIPELINE STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"📁 Input file: {input_file}")
        self.logger.info(f"📁 Output file: {output_file}")
        self.logger.info(f"📊 Total stages: {total_stages}")
        self.logger.info(
            f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        self.logger.info("=" * 60)

    def log_stage_start(self, stage_name: str, stage_index: int, total_stages: int):
        """Log the start of a specific stage."""
        self.stage_start_times[stage_name] = time.time()
        progress = (stage_index / total_stages) * 100

        self.logger.info(
            f"🔄 STAGE {stage_index + 1}/{total_stages} - {stage_name.upper()}"
        )
        self.logger.info(f"📈 Overall Progress: {progress:.1f}%")
        self.logger.info("-" * 60)

    def log_stage_progress(
        self, stage_name: str, current: int, total: int, item_type: str = "items"
    ):
        """Log progress within a stage."""
        if total > 0:
            progress = (current / total) * 100
            self.logger.info(
                f"  ⏳ Processing {item_type}: {current}/{total} ({progress:.1f}%)"
            )

    def log_stage_completion(
        self,
        stage_name: str,
        stage_index: int,
        total_stages: int,
        duration: Optional[float] = None,
    ):
        """Log the completion of a specific stage."""
        if duration is None and stage_name in self.stage_start_times:
            duration = time.time() - self.stage_start_times[stage_name]

        progress = ((stage_index + 1) / total_stages) * 100

        self.logger.info(
            f"✅ STAGE {stage_index + 1}/{total_stages} COMPLETED - {stage_name.upper()}"
        )
        if duration:
            self.logger.info(f"  ⏱️  Duration: {duration:.2f} seconds")
        self.logger.info(f"📈 Overall Progress: {progress:.1f}%")
        self.logger.info("-" * 60)

    def log_segment_processing(
        self,
        stage_name: str,
        segment_index: int,
        total_segments: int,
        speaker: Optional[str] = None,
    ):
        """Log individual segment processing."""
        if stage_name not in self.segment_counts:
            self.segment_counts[stage_name] = 0

        self.segment_counts[stage_name] += 1

        if speaker:
            self.logger.debug(
                f"  🎯 Processing segment {segment_index + 1}/{total_segments} (Speaker: {speaker})"
            )
        else:
            self.logger.debug(
                f"  🎯 Processing segment {segment_index + 1}/{total_segments}"
            )

    def log_speaker_batch(self, speaker: str, segment_count: int):
        """Log speaker batch processing."""
        self.logger.info(
            f"  👤 Processing {segment_count} segments for speaker: {speaker}"
        )

    def log_tts_method(self, method: str):
        """Log the selected TTS method."""
        self.logger.info(f"  🎤 Using TTS method: {method}")

    def log_memory_usage(
        self, stage_name: str, allocated_gb: float, reserved_gb: float
    ):
        """Log GPU memory usage."""
        self.logger.debug(
            f"  💾 [{stage_name}] GPU Memory - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB"
        )

    def log_error(
        self, stage_name: str, error: Exception, context: Optional[str] = None
    ):
        """Log errors with context and recovery suggestions."""
        error_msg = f"❌ ERROR in {stage_name.upper()}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(error)}"

        self.logger.error(error_msg)
        self.logger.error("-" * 60)
        # log traceback
        self.logger.error(f"  🔍 Traceback:")
        self.logger.error(traceback.format_exc())
        self.logger.debug(f"  🔍 Error details: {type(error).__name__}: {str(error)}")

        # Add recovery suggestions based on error type and stage
        self._log_recovery_suggestions(stage_name, error, context)

    def _log_recovery_suggestions(
        self, stage_name: str, error: Exception, context: Optional[str] = None
    ):
        """Log recovery suggestions based on error type and context."""
        suggestions = []

        # Common error patterns and suggestions
        error_str = str(error).lower()

        if "permission" in error_str or "access" in error_str:
            suggestions.append("Check file/directory permissions")
            suggestions.append(
                "Ensure you have write access to the temporary directory"
            )

        if "disk" in error_str or "space" in error_str:
            suggestions.append("Check available disk space")
            suggestions.append("Clean up temporary files if disk is full")

        if "memory" in error_str or "cuda" in error_str:
            suggestions.append("Reduce batch size for memory-intensive operations")
            suggestions.append("Close other applications to free up memory")
            suggestions.append("Consider using CPU instead of GPU for this operation")

        if "network" in error_str or "connection" in error_str:
            suggestions.append("Check your internet connection")
            suggestions.append("Verify API keys and credentials")
            suggestions.append("Try again in a few minutes")

        if "file not found" in error_str:
            suggestions.append("Verify input file paths")
            suggestions.append(
                "Check if required files exist in the temporary directory"
            )

        if stage_name == "transcribe" and (
            "whisper" in error_str or "model" in error_str
        ):
            suggestions.append("Ensure Whisper model is properly installed")
            suggestions.append("Check available disk space for model download")

        if stage_name == "translate" and ("openai" in error_str or "api" in error_str):
            suggestions.append("Verify OpenAI API key is set correctly")
            suggestions.append("Check API quota and billing status")

        if stage_name == "generate_tts" and (
            "tts" in error_str or "voice" in error_str
        ):
            suggestions.append("Check TTS method configuration")
            suggestions.append("Verify reference audio files are valid")

        # Stage-specific suggestions
        if stage_name == "convert_mp4_to_wav":
            suggestions.append("Ensure FFmpeg is installed and accessible")
            suggestions.append("Check input video file format and integrity")

        if stage_name == "separate_audio":
            suggestions.append("Verify Demucs is properly installed")
            suggestions.append("Check if vocals separation model is available")

        if suggestions:
            self.logger.info("  💡 RECOVERY SUGGESTIONS:")
            for suggestion in suggestions[:3]:  # Limit to top 3 suggestions
                self.logger.info(f"    • {suggestion}")

    def log_warning(self, stage_name: str, warning: str, context: Optional[str] = None):
        """Log warnings."""
        warning_msg = f"⚠️  WARNING in {stage_name.upper()}"
        if context:
            warning_msg += f" ({context})"
        warning_msg += f": {warning}"

        self.logger.warning(warning_msg)

    def log_pipeline_completion(self, total_duration: float, success: bool = True):
        """Log the completion of the entire pipeline."""
        self.logger.info("=" * 60)
        if success:
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            self.logger.error("💥 PIPELINE FAILED!")
        self.logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        self.logger.info(
            f"🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        self.logger.info("=" * 60)

    def log_file_operation(self, operation: str, file_path: str, success: bool = True):
        """Log file operations."""
        if success:
            self.logger.debug(f"  📄 {operation}: {file_path}")
        else:
            self.logger.warning(f"  ⚠️  Failed {operation}: {file_path}")

    def log_api_call(
        self, api_name: str, success: bool = True, duration: Optional[float] = None
    ):
        """Log API calls."""
        if success:
            if duration:
                self.logger.debug(
                    f"  🌐 API call to {api_name} completed in {duration:.2f}s"
                )
            else:
                self.logger.debug(f"  🌐 API call to {api_name} completed")
        else:
            self.logger.warning(f"  ⚠️  API call to {api_name} failed")

    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger initialization."""
        return time.time() - self.start_time

    def start_timing(self, operation_name: str):
        """Start timing an operation."""
        self.operation_times[operation_name] = time.time()
        self.logger.debug(f"⏱️  Started timing: {operation_name}")

    def end_timing(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time."""
        if operation_name not in self.operation_times:
            self.logger.warning(
                f"⚠️  No start time found for operation: {operation_name}"
            )
            return 0.0

        elapsed = time.time() - self.operation_times[operation_name]
        del self.operation_times[operation_name]

        self.logger.debug(f"⏱️  Completed {operation_name}: {elapsed:.2f}s")
        return elapsed

    def log_performance_metric(
        self, category: str, metric_name: str, value: float, unit: str = ""
    ):
        """Log a performance metric."""
        if category not in self.performance_metrics:
            self.performance_metrics[category] = {}

        self.performance_metrics[category][metric_name] = value

        unit_str = f" {unit}" if unit else ""
        self.logger.debug(
            f"📊 Performance [{category}]: {metric_name} = {value:.2f}{unit_str}"
        )

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all performance metrics."""
        return self.performance_metrics.copy()

    def log_timing_summary(self):
        """Log a summary of timing information."""
        if not self.performance_metrics:
            return

        self.logger.info("📊 PERFORMANCE SUMMARY:")
        for category, metrics in self.performance_metrics.items():
            self.logger.info(f"  {category.upper()}:")
            for metric_name, value in metrics.items():
                self.logger.info(f"    {metric_name}: {value:.2f}")


class PipelineFormatter(logging.Formatter):
    """Custom formatter for pipeline logs with colors and emojis."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Color the level name
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        colored_level = f"{level_color}{record.levelname}{self.COLORS['RESET']}"

        # Format the message
        formatted_message = f"[{timestamp}] {colored_level}: {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted_message += f"\n{self.COLORS['ERROR']}{self.formatException(record.exc_info)}{self.COLORS['RESET']}"

        return formatted_message


# Global logger instance
_pipeline_logger = None


def get_logger(name: str = "anime-dubbing", log_level: str = "INFO") -> PipelineLogger:
    """
    Get or create a global pipeline logger instance.

    Args:
        name: Logger name
        log_level: Logging level

    Returns:
        PipelineLogger instance
    """
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger(name, log_level)
    return _pipeline_logger


def setup_console_logging(level: str = "INFO"):
    """
    Setup basic console logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
