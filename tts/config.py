import os
import logging
import yaml

# TTS Configuration Constants
# These constants control batch processing for memory management during TTS generation.
# Adjust these values based on your system's memory capacity:
# - Lower values use less memory but may be slower
# - Higher values are faster but require more memory
TTS_SPEAKER_BATCH_SIZE = (
    1  # Number of speakers to process at once for memory management
)


def load_tts_config():
    """
    Load TTS configuration from config file.

    Returns:
        dict: TTS configuration dictionary
    """
    config_path = os.path.join(".", "config", "tts_config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Ensure we always return a dict
            return config if isinstance(config, dict) else {}
    except Exception as e:
        logging.warning(f"Failed to load TTS config from {config_path}: {e}. Using defaults.")
        return {
            "reference_audio": {
                "min_duration_minutes": 1
            }
        }