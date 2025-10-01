import os
import torch
import torchaudio
import logging
import gc
import yaml
from typing import List, Dict, Optional
import numpy as np
from TTS.api import TTS
from utils.logger import get_logger

# Supported languages for XTTS
XTTS_SUPPORTED_LANGUAGES = [
    "en", "ja", "zh", "ko", "es", "fr", "de", "it", "pt", "ru", "ar", "hi"
]


def validate_language(target_lang: str) -> bool:
    """
    Validate if the target language is supported by XTTS.

    Args:
        target_lang: Target language code (e.g., "en", "ja")

    Returns:
        bool: True if language is supported

    Raises:
        ValueError: If language is not supported
    """
    logger = get_logger("xtts-validator")

    if target_lang not in XTTS_SUPPORTED_LANGUAGES:
        logger.logger.error(f"Unsupported language {target_lang} for XTTS")
        raise ValueError(f"Target language {target_lang} not supported by XTTS")

    return True

def load_tts_config():
    """
    Load TTS configuration from config file.

    Returns:
        dict: TTS configuration dictionary
    """
    config_path = os.path.join(".", "config", "tts_config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load TTS config from {config_path}: {e}. Using defaults.")
        return {
            "reference_audio": {
                "min_duration_minutes": 1
            }
        }


def cleanup_memory():
    """
    Clean up memory by collecting garbage and clearing GPU cache.
    Call this periodically during long-running TTS operations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class XTTSGenerator:
    """
    XTTS (Coqui XTTS-v2) TTS generator for voice cloning and multilingual synthesis.
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "auto"):
        """
        Initialize XTTS model.

        Args:
            model_name: Name of the XTTS model to load
            device: Device to load model on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.logger = get_logger("xtts-generator")

        # Load model on first use
        self._load_model()

    def _load_model(self):
        """
        Load the XTTS model. Downloads ~1GB model on first run.
        """
        if self.model is not None:
            return

        try:
            # Always determine GPU availability for TTS model initialization
            gpu_available = torch.cuda.is_available()

            if self.device == "auto":
                device = "cuda" if gpu_available else "cpu"
            else:
                device = self.device

            self.logger.logger.info(f"Loading XTTS model: {self.model_name} on {device}")
            self.model = TTS(model_name=self.model_name, progress_bar=False, gpu=gpu_available)
            self.logger.logger.info(f"XTTS model loaded successfully on {device}")

        except Exception as e:
            self.logger.logger.error(f"Failed to load XTTS model: {e}")
            raise

    def generate_tts_segment(
        self,
        text: str,
        speaker_wav: str,
        language: str = "en",
        temperature: float = 0.7,
        speed: float = 1.0,
        target_duration: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate TTS audio for a single text segment.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning
            language: Target language for synthesis
            temperature: Temperature for voice variation (0.1-1.0)
            speed: Speaking speed multiplier
            target_duration: Target duration in seconds (optional)
            output_path: Path to save audio file (optional)

        Returns:
            torch.Tensor: Generated audio tensor
        """
        if self.model is None:
            raise RuntimeError("XTTS model not loaded")

        try:
            # Generate TTS with XTTS
            self.logger.logger.info(f"Generating TTS: '{text[:50]}...'")

            wav = self.model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                temperature=temperature,
                speed=speed,
            )

            # Convert to torch tensor
            wav_tensor = torch.from_numpy(np.array(wav)).unsqueeze(0).float()

            # Resample if needed (XTTS outputs at 24000Hz)
            target_sr = 24000
            current_sr = 24000  # XTTS default sample rate

            # Note: Resampling logic is currently disabled since both rates are identical
            # If different sample rates are needed in the future, uncomment below:
            # if current_sr != target_sr:
            #     resampler = torchaudio.transforms.Resample(current_sr, target_sr)
            #     wav_tensor = resampler(wav_tensor)

            # Adjust duration if specified
            if target_duration is not None:
                current_duration = wav_tensor.shape[1] / target_sr
                if current_duration > target_duration:
                    # Trim
                    trim_samples = int(target_duration * target_sr)
                    wav_tensor = wav_tensor[:, :trim_samples]
                elif current_duration < target_duration:
                    # Pad with silence
                    pad_samples = int((target_duration - current_duration) * target_sr)
                    silence = torch.zeros(1, pad_samples)
                    wav_tensor = torch.cat([wav_tensor, silence], dim=1)

            # Save if output path specified
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, wav_tensor, target_sr)

            return wav_tensor

        except Exception as e:
            self.logger.logger.error(f"Failed to generate TTS: {e}")
            raise

    def process_reference_audio(self, ref_audio_path: str, max_duration: float = 10.0) -> str:
        """
        Process reference audio by slicing to optimal duration.

        Args:
            ref_audio_path: Path to reference audio file
            max_duration: Maximum duration in seconds

        Returns:
            Path to processed reference audio
        """
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        # Load audio
        waveform, sr = torchaudio.load(ref_audio_path)
        duration = waveform.shape[1] / sr

        # Slice if longer than max_duration
        if duration > max_duration:
            slice_samples = int(max_duration * sr)
            waveform = waveform[:, :slice_samples]

            # Save sliced version
            sliced_path = ref_audio_path.replace(".wav", "_sliced.wav")
            torchaudio.save(sliced_path, waveform, sr)
            self.logger.logger.info(f"Sliced reference audio to {max_duration}s")
            return sliced_path
        else:
            return ref_audio_path

    def get_emotion_parameters(self, speaker: str, emotion_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get emotion-based parameters for TTS generation.

        Args:
            speaker: Speaker identifier
            emotion_data: Optional emotion data from emotion stage

        Returns:
            Dict with temperature and speed parameters
        """
        tts_config = load_tts_config()

        # Ensure tts_config is a dictionary
        if not isinstance(tts_config, dict):
            tts_config = {}

        # Try to get temperature and speed from XTTS config, fallback to root level, then defaults
        tts_methods = tts_config.get("tts_methods", {})
        if not isinstance(tts_methods, dict):
            tts_methods = {}

        xtts_config = tts_methods.get("xtts", {})
        if not isinstance(xtts_config, dict):
            xtts_config = {}

        config_section = xtts_config.get("config", {})
        if not isinstance(config_section, dict):
            config_section = {}

        temperature = config_section.get("temperature", tts_config.get("temperature", 0.7))
        speed = config_section.get("speed", tts_config.get("speed", 1.0))
        emotion_support = config_section.get("emotion_support", tts_config.get("emotion_support", True))

        # Ensure temperature and speed are valid floats
        try:
            temperature = float(temperature) if temperature is not None else 0.7
        except (TypeError, ValueError):
            temperature = 0.7

        try:
            speed = float(speed) if speed is not None else 1.0
        except (TypeError, ValueError):
            speed = 1.0

        # Clamp speed to maximum of 2.0x
        if speed > 2.0:
            self.logger.logger.warning(f"Speed factor {speed} exceeds maximum limit of 2.0, clamping to 2.0")
            speed = 2.0

        # Apply emotion-based adjustments if available
        if emotion_data and emotion_support:
            speaker_emotions = emotion_data.get("overall_emotions", {}).get(speaker, {})
            if speaker_emotions:
                dominant_emotion = max(speaker_emotions, key=speaker_emotions.get)
                if dominant_emotion == "angry":
                    temperature = 0.9
                    speed = 1.1
                elif dominant_emotion == "happy":
                    temperature = 0.8
                    speed = 1.05
                elif dominant_emotion == "sad":
                    temperature = 0.6
                    speed = 0.95

        return {"temperature": temperature, "speed": speed}

    def generate_tts_for_speaker(
        self,
        segments: List[Dict],
        speaker: str,
        refs_by_speaker: Dict,
        default_ref: str,
        tmp_path: str,
        target_sr: int = 24000,
        language: str = "en",
        ref_text: str = "",
        emotion_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate TTS audio segments for a specific speaker using XTTS.

        Args:
            segments: List of segments for this speaker
            speaker: Speaker identifier
            refs_by_speaker: Dictionary of speaker reference audio paths
            default_ref: Default reference audio path
            tmp_path: Path to temporary directory
            target_sr: Target sample rate for output audio
            language: Target language for TTS
            ref_text: Reference text (for logging/fallback)
            emotion_data: Optional emotion data for expressive TTS

        Returns:
            List of TTS segment dictionaries with paths and metadata
        """
        tts_segments = []

        # Get reference audio path
        ref_audio_path = refs_by_speaker.get(speaker, default_ref)
        if isinstance(ref_audio_path, dict):
            ref_audio_path = ref_audio_path["audio_path"]

        if not ref_audio_path:
            self.logger.logger.warning(f"No reference audio found for speaker {speaker}")
            return tts_segments

        # Process reference audio
        ref_audio_full_path = os.path.join(tmp_path, ref_audio_path)
        processed_ref_path = self.process_reference_audio(ref_audio_full_path)

        # Get emotion parameters
        emotion_params = self.get_emotion_parameters(speaker, emotion_data)

        for segment in segments:
            text = segment.get("translated_text", "")
            if not text.strip():
                continue

            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time

            try:
                # Generate TTS with emotion parameters
                output_filename = f"tts/{speaker}_{start_time:.1f}_{end_time:.1f}.wav"
                output_path = os.path.join(tmp_path, output_filename)

                wav_tensor = self.generate_tts_segment(
                    text=text,
                    speaker_wav=processed_ref_path,
                    language=language,
                    temperature=emotion_params["temperature"],
                    speed=emotion_params["speed"],
                    target_duration=duration,
                    output_path=output_path
                )

                # Create segment entry
                tts_segment = {
                    "path": output_filename,
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker,
                    "duration": duration,
                    "text": text,
                }
                tts_segments.append(tts_segment)

                self.logger.logger.info(f"Generated: {output_filename} ({duration:.2f}s)")

            except Exception as e:
                self.logger.logger.error(f"Failed to generate TTS for segment {start_time:.1f}-{end_time:.1f}: {e}")
                continue

        return tts_segments

    def cleanup(self):
        """
        Clean up model and free memory.
        """
        if self.model is not None:
            del self.model
            self.model = None

        cleanup_memory()
        self.logger.logger.info("XTTS model cleaned up")


# Global instance for backward compatibility
_xtts_instance = None

def get_xtts_generator() -> XTTSGenerator:
    """
    Get or create global XTTS generator instance.

    Returns:
        XTTSGenerator: Global XTTS generator instance
    """
    global _xtts_instance
    if _xtts_instance is None:
        _xtts_instance = XTTSGenerator()
    return _xtts_instance


def generate_tts_for_speaker_xtts(
    segments: List[Dict],
    speaker: str,
    refs_by_speaker: Dict,
    default_ref: str,
    tmp_path: str,
    target_sr: int = 24000,
    language: str = "en",
    ref_text: str = "",
    emotion_data: Optional[Dict] = None,
) -> List[Dict]:
    """
    Generate TTS audio segments for a specific speaker using XTTS.
    This function maintains backward compatibility with the original implementation.

    Args:
        segments: List of segments for this speaker
        speaker: Speaker identifier
        refs_by_speaker: Dictionary of speaker reference audio paths
        default_ref: Default reference audio path
        tmp_path: Path to temporary directory
        target_sr: Target sample rate for output audio
        language: Target language for TTS
        ref_text: Reference text (for logging/fallback)
        emotion_data: Optional emotion data for expressive TTS

    Returns:
        List of TTS segment dictionaries with paths and metadata
    """
    generator = get_xtts_generator()
    return generator.generate_tts_for_speaker(
        segments=segments,
        speaker=speaker,
        refs_by_speaker=refs_by_speaker,
        default_ref=default_ref,
        tmp_path=tmp_path,
        target_sr=target_sr,
        language=language,
        ref_text=ref_text,
        emotion_data=emotion_data,
    )