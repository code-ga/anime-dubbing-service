import os
import shutil
import logging
import torch
import torchaudio
import whisper
import gc
from datetime import datetime
from typing import List, Dict, Optional
from vocos import Vocos
from tts.F5 import generate_tts_custom, generate_tts_for_speaker, F5TTS, validate_language as validate_f5_language
from tts.edge_tts import generate_tts_for_speaker as generate_tts_for_speaker_edge, validate_language as validate_edge_language
from tts.xtts import generate_tts_for_speaker_xtts, validate_language as validate_xtts_language
from tts.config import TTS_SPEAKER_BATCH_SIZE, load_tts_config
from utils.metadata import load_previous_result
from utils.logger import get_logger
from abc import ABC, abstractmethod

# Custom exceptions for TTS engines
class TTSEngineError(Exception):
    """Base exception for TTS engine errors"""
    pass

class TTSEngineLoadError(TTSEngineError):
    """Exception raised when TTS engine fails to load"""
    pass

class TTSEngineGenerationError(TTSEngineError):
    """Exception raised when TTS generation fails"""
    pass

class TTSEngineValidationError(TTSEngineError):
    """Exception raised when TTS engine validation fails"""
    pass

# Abstract base class for TTS engines
class TTSEngine(ABC):
    """
    Abstract base class for TTS engines.
    All TTS engines should inherit from this class and implement the required methods.
    """

    def __init__(self, engine_name: str, target_sr: int = 24000):
        """
        Initialize the TTS engine.

        Args:
            engine_name: Name of the TTS engine
            target_sr: Target sample rate for generated audio
        """
        self.engine_name = engine_name
        self.target_sr = target_sr
        self.logger = get_logger("tts-orchestrator")

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load the TTS model. This method should be implemented by subclasses.

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        pass

    @abstractmethod
    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional engine-specific parameters

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up resources and free memory.
        This method should be called when the engine is no longer needed.
        """
        pass

    def validate_language(self, language: str):
        """
        Validate if the engine supports the given language.
        Default implementation does nothing - subclasses should override if needed.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        pass

class XTTSGenerator(TTSEngine):
    """
    XTTS TTS engine implementation.
    """

    def __init__(self):
        super().__init__("xtts", target_sr=24000)
        self.xtts_model = None

    def load_model(self, **kwargs):
        """
        Load the XTTS model.

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        try:
            # XTTS model is loaded by the XTTS module, so we don't need to do anything here
            # The actual model loading happens in the generate_tts_for_speaker_xtts function
            logging.info("XTTS model loaded successfully")
        except Exception as e:
            raise TTSEngineLoadError(f"Failed to load XTTS model: {e}")

    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment using XTTS.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional parameters (language, emotion_data, etc.)

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        try:
            text = segment.get("translated_text", "")
            if not text.strip():
                raise TTSEngineGenerationError("Empty text for TTS generation")

            # Use the existing XTTS function
            speaker_tts_segments = generate_tts_for_speaker_xtts(
                [segment],  # Pass as list since the function expects segments
                segment.get("speaker", "default"),
                {segment.get("speaker", "default"): ref_audio_path},
                ref_audio_path,  # default_ref
                kwargs.get("tmp_path", ""),
                self.target_sr,
                language=kwargs.get("language", "en"),
                ref_text=ref_text or "",
                emotion_data=kwargs.get("emotion_data"),
            )

            if not speaker_tts_segments:
                raise TTSEngineGenerationError("No TTS segments generated")

            return speaker_tts_segments[0]  # Return the first (and only) segment

        except Exception as e:
            raise TTSEngineGenerationError(f"XTTS generation failed: {e}")

    def validate_language(self, language: str):
        """
        Validate if XTTS supports the given language.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        try:
            validate_xtts_language(language)
        except Exception as e:
            raise TTSEngineValidationError(f"XTTS does not support language {language}: {e}")

    def cleanup(self):
        """
        Clean up XTTS resources and free memory.
        """
        if self.xtts_model is not None:
            del self.xtts_model
        cleanup_memory()

class OrpheusGenerator(TTSEngine):
    """
    Orpheus TTS engine implementation.
    """

    def __init__(self):
        super().__init__("orpheus", target_sr=24000)
        self.orpheus_model = None

    def load_model(self, **kwargs):
        """
        Load the Orpheus model.

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        try:
            if not torch.cuda.is_available():
                raise TTSEngineLoadError("Orpheus-TTS requires GPU but CUDA is not available")

            if not ORPHEUS_AVAILABLE:
                raise TTSEngineLoadError("Orpheus-TTS not available. Install with: pip install orpheus-speech")

            device = torch.device("cuda")
            tts_config = load_tts_config()
            if isinstance(tts_config, dict):
                orpheus_config = tts_config.get("tts_methods", {}).get("orpheus", {}).get("config", {})
            else:
                orpheus_config = {}

            model_type = orpheus_config.get("model_type", "finetuned")
            if model_type == "finetuned":
                model_path = "canopyai/Orpheus-3B-0.1-ft"
            elif model_type == "pretrained":
                model_path = "canopyai/Orpheus-3B-0.1"
            else:  # multilingual
                model_path = "canopyai/Orpheus-3B-0.1-multilingual"

            self.orpheus_model = OrpheusModel.from_pretrained(model_path, device=device)
            logging.info(f"Orpheus model loaded successfully: {model_path}")

        except Exception as e:
            raise TTSEngineLoadError(f"Failed to load Orpheus model: {e}")

    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment using Orpheus.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional parameters (emotion_data, etc.)

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        try:
            text = segment.get("translated_text", "")
            if not text.strip():
                raise TTSEngineGenerationError("Empty text for TTS generation")

            if self.orpheus_model is None:
                raise TTSEngineGenerationError("Orpheus model not loaded")

            # Load reference audio
            ref_wav, ref_sr = torchaudio.load(ref_audio_path)

            # Get TTS configuration for Orpheus settings
            tts_config = load_tts_config()
            if isinstance(tts_config, dict):
                orpheus_config = tts_config.get("tts_methods", {}).get("orpheus", {}).get("config", {})
            else:
                orpheus_config = {}

            # Get voice name from config or use default
            voice = orpheus_config.get("voice", "tara")

            # Build prompt with voice and emotion tags if enabled
            prompt = f"{voice}: {text}"

            # Add emotion tags if emotion stage is active and enabled
            emotion_data = kwargs.get("emotion_data")
            if emotion_data and orpheus_config.get("emotion_tags", True):
                emotion = getattr(segment, 'emotion', None)
                if emotion and emotion != "neutral":
                    prompt = f"<{emotion}>{prompt}</emotion>"

            # Generate audio using Orpheus
            generated_audio = self.orpheus_model.generate(
                prompt,
                reference_audio=ref_wav,
                reference_text=ref_text or text,
                stream=True,
                temperature=orpheus_config.get("temperature", 0.7),
                repetition_penalty=orpheus_config.get("repetition_penalty", 1.1),
            )

            # Convert to tensor and save
            audio_tensor = torch.tensor(generated_audio, dtype=torch.float32).unsqueeze(0)

            # Adjust duration to match segment timing
            target_duration = segment["end"] - segment["start"]
            current_duration = audio_tensor.shape[1] / self.target_sr

            if current_duration != target_duration:
                if current_duration < target_duration:
                    # Pad with silence
                    silence_duration = target_duration - current_duration
                    silence_samples = int(silence_duration * self.target_sr)
                    silence = torch.zeros(1, silence_samples)
                    audio_tensor = torch.cat([audio_tensor, silence], dim=1)
                else:
                    # Trim audio
                    max_samples = int(target_duration * self.target_sr)
                    audio_tensor = audio_tensor[:, :max_samples]

            # Save audio file
            tmp_path = kwargs.get("tmp_path", "")
            seg_filename = f"tts/seg_{segment.get('speaker', 'default')}_{id(segment)}.wav"
            seg_path = os.path.join(tmp_path, seg_filename)
            torchaudio.save(seg_path, audio_tensor, self.target_sr)

            # Create TTS segment entry
            tts_segment = {
                "path": seg_filename,
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment.get("speaker", "default"),
                "duration": target_duration,
                "text": text,
                "tts_method": "orpheus"
            }

            # Clean up memory
            del audio_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return tts_segment

        except Exception as e:
            raise TTSEngineGenerationError(f"Orpheus generation failed: {e}")

    def validate_language(self, language: str):
        """
        Validate if Orpheus supports the given language.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        supported_languages = ["en", "ja", "zh", "ko", "es", "fr", "de", "it", "pt", "ru"]
        if language not in supported_languages:
            logging.warning(f"Orpheus may not fully support language: {language}")

    def cleanup(self):
        """
        Clean up Orpheus resources and free memory.
        """
        if self.orpheus_model is not None:
            del self.orpheus_model
        cleanup_memory()

class EdgeTTSGenerator(TTSEngine):
    """
    Edge-TTS engine implementation.
    """

    def __init__(self):
        super().__init__("edge", target_sr=24000)

    def load_model(self, **kwargs):
        """
        Load the Edge-TTS model.
        Edge-TTS doesn't require explicit model loading, so this is a no-op.

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        try:
            # Edge-TTS doesn't require explicit model loading
            logging.info("Edge-TTS model loaded successfully")
        except Exception as e:
            raise TTSEngineLoadError(f"Failed to load Edge-TTS model: {e}")

    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment using Edge-TTS.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional parameters (language, etc.)

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        try:
            text = segment.get("translated_text", "")
            if not text.strip():
                raise TTSEngineGenerationError("Empty text for TTS generation")

            # Use the existing Edge-TTS function
            speaker_tts_segments = generate_tts_for_speaker_edge(
                [segment],  # Pass as list since the function expects segments
                segment.get("speaker", "default"),
                {segment.get("speaker", "default"): ref_audio_path},
                ref_audio_path,  # default_ref
                kwargs.get("tmp_path", ""),
                self.target_sr,
                language=kwargs.get("language", "en"),
                ref_text=ref_text or "",
                speed=1.0,  # Default speed, can be made configurable later
            )

            if not speaker_tts_segments:
                raise TTSEngineGenerationError("No TTS segments generated")

            return speaker_tts_segments[0]  # Return the first (and only) segment

        except Exception as e:
            raise TTSEngineGenerationError(f"Edge-TTS generation failed: {e}")

    def validate_language(self, language: str):
        """
        Validate if Edge-TTS supports the given language.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        try:
            validate_edge_language(language)
        except Exception as e:
            raise TTSEngineValidationError(f"Edge-TTS does not support language {language}: {e}")

    def cleanup(self):
        """
        Clean up Edge-TTS resources and free memory.
        Edge-TTS doesn't require explicit cleanup, so this is a no-op.
        """
        cleanup_memory()

class RVCGenerator(TTSEngine):
    """
    RVC TTS engine implementation using F5-TTS as backend.
    """

    def __init__(self):
        super().__init__("rvc", target_sr=24000)
        self.model = None
        self.vocos = None

    def load_model(self, **kwargs):
        """
        Load the RVC model (F5-TTS backend).

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = F5TTS(model="F5TTS_Base")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
            logging.info("RVC model (F5-TTS backend) loaded successfully")
        except Exception as e:
            raise TTSEngineLoadError(f"Failed to load RVC model: {e}")

    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment using RVC.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional parameters

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        try:
            text = segment.get("translated_text", "")
            if not text.strip():
                raise TTSEngineGenerationError("Empty text for TTS generation")

            if self.model is None or self.vocos is None:
                raise TTSEngineGenerationError("RVC models not loaded")

            # Use the existing F5-TTS function for RVC
            speaker_tts_segments = generate_tts_for_speaker(
                [segment],  # Pass as list since the function expects segments
                segment.get("speaker", "default"),
                {segment.get("speaker", "default"): ref_audio_path},
                ref_audio_path,  # default_ref
                kwargs.get("tmp_path", ""),
                self.target_sr,
                self.model,
                self.vocos,
                ref_text=ref_text or "",
            )

            if not speaker_tts_segments:
                raise TTSEngineGenerationError("No TTS segments generated")

            return speaker_tts_segments[0]  # Return the first (and only) segment

        except Exception as e:
            raise TTSEngineGenerationError(f"RVC generation failed: {e}")

    def validate_language(self, language: str):
        """
        Validate if RVC (F5-TTS backend) supports the given language.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        try:
            validate_f5_language(language)
        except Exception as e:
            raise TTSEngineValidationError(f"RVC (F5-TTS backend) does not support language {language}: {e}")

    def cleanup(self):
        """
        Clean up RVC resources and free memory.
        """
        if self.model is not None:
            del self.model
        if self.vocos is not None:
            del self.vocos
        cleanup_memory()

class F5Generator(TTSEngine):
    """
    F5-TTS engine implementation.
    """

    def __init__(self):
        super().__init__("f5", target_sr=24000)
        self.model = None
        self.vocos = None

    def load_model(self, **kwargs):
        """
        Load the F5-TTS model.

        Raises:
            TTSEngineLoadError: If model loading fails
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = F5TTS(model="F5TTS_Base")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
            logging.info("F5-TTS model loaded successfully")
        except Exception as e:
            raise TTSEngineLoadError(f"Failed to load F5-TTS model: {e}")

    def generate_segment(self, segment: dict, ref_audio_path: str, ref_text: Optional[str] = None, **kwargs) -> dict:
        """
        Generate TTS audio for a single segment using F5-TTS.

        Args:
            segment: Segment dictionary containing text, timing, etc.
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Reference text for voice cloning
            **kwargs: Additional parameters

        Returns:
            Dictionary containing audio path, timing, and metadata

        Raises:
            TTSEngineGenerationError: If generation fails
        """
        try:
            text = segment.get("translated_text", "")
            if not text.strip():
                raise TTSEngineGenerationError("Empty text for TTS generation")

            if self.model is None or self.vocos is None:
                raise TTSEngineGenerationError("F5-TTS models not loaded")

            # Use the existing F5-TTS function
            speaker_tts_segments = generate_tts_for_speaker(
                [segment],  # Pass as list since the function expects segments
                segment.get("speaker", "default"),
                {segment.get("speaker", "default"): ref_audio_path},
                ref_audio_path,  # default_ref
                kwargs.get("tmp_path", ""),
                self.target_sr,
                self.model,
                self.vocos,
                ref_text=ref_text or "",
            )

            if not speaker_tts_segments:
                raise TTSEngineGenerationError("No TTS segments generated")

            return speaker_tts_segments[0]  # Return the first (and only) segment

        except Exception as e:
            raise TTSEngineGenerationError(f"F5-TTS generation failed: {e}")

    def validate_language(self, language: str):
        """
        Validate if F5-TTS supports the given language.

        Args:
            language: Language code to validate

        Raises:
            TTSEngineValidationError: If language is not supported
        """
        try:
            validate_f5_language(language)
        except Exception as e:
            raise TTSEngineValidationError(f"F5-TTS does not support language {language}: {e}")

    def cleanup(self):
        """
        Clean up F5-TTS resources and free memory.
        """
        if self.model is not None:
            del self.model
        if self.vocos is not None:
            del self.vocos
        cleanup_memory()

def create_tts_engine(tts_method: str) -> TTSEngine:
    """
    Factory function to create TTS engine instances based on method.

    Args:
        tts_method: TTS method name (edge, xtts, orpheus, rvc, f5)

    Returns:
        TTSEngine instance

    Raises:
        ValueError: If TTS method is not supported
    """
    engine_map = {
        "edge": EdgeTTSGenerator,
        "xtts": XTTSGenerator,
        "orpheus": OrpheusGenerator,
        "rvc": RVCGenerator,
        "f5": F5Generator,
    }

    if tts_method not in engine_map:
        raise ValueError(f"Unsupported TTS method: {tts_method}. Supported methods: {list(engine_map.keys())}")

    return engine_map[tts_method]()

class ReferenceBuilder:
    """
    Class for building reference audio files for speakers.
    """

    def __init__(self):
        self.logger = get_logger("tts-orchestrator")

    def build_speaker_refs(self, tmp_path, metadata_path, inputs_data, **kwargs) -> dict:
        """
        Extract reference audios per speaker from original vocals.

        Args:
            tmp_path: Path to temporary directory.
            metadata_path: Path to metadata.
            inputs_data: Dict of previous stage data.
            **kwargs: Additional arguments.

        Returns:
            Stage data with refs_by_speaker, default_ref, extraction_criteria.
        """
        # Load TTS configuration
        tts_config = load_tts_config()
        if isinstance(tts_config, dict):
            min_duration_minutes = tts_config.get("reference_audio", {}).get("min_duration_minutes", 1)
        else:
            min_duration_minutes = 1

        self.logger.logger.info("🎯 Starting reference audio extraction")
        self.logger.logger.info(f"📏 Using minimum reference duration: {min_duration_minutes} minute(s)")

        transcribe_data = load_previous_result(metadata_path, "transcribe")
        separate_data = inputs_data.get("separate_audio")

        # Use vocals_path if separate_audio was run, otherwise fallback to full_wav_path
        if separate_data and "vocals_path" in separate_data:
            vocals_path = os.path.join(tmp_path, separate_data["vocals_path"])
            self.logger.logger.info(f"📁 Loading separated vocals from: {vocals_path}")
        else:
            # Fallback to full audio when audio separation is skipped
            convert_data = inputs_data.get("convert_mp4_to_wav", {})
            vocals_path = os.path.join(tmp_path, convert_data.get("full_wav_path", "full.wav"))
            self.logger.logger.info(f"📁 Loading full audio (audio separation skipped) from: {vocals_path}")

        waveform, sr = torchaudio.load(vocals_path)
        self.logger.logger.info(f"🎵 Audio loaded: {waveform.shape[1]/sr:.2f} seconds at {sr}Hz")

        refs_dir = os.path.join(tmp_path, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        self.logger.log_file_operation("create", refs_dir, True)

        refs_by_speaker = {}
        speakers = set(
            seg.get("speaker") for seg in transcribe_data["segments"] if seg.get("speaker")
        )

        self.logger.logger.info(
            f"👥 Found {len(speakers)} speakers: {', '.join(sorted(speakers))}"
        )

        # Process speakers in batches to manage memory usage
        speaker_list = list(speakers)

        for i in range(0, len(speaker_list), TTS_SPEAKER_BATCH_SIZE):
            batch_speakers = speaker_list[i : i + TTS_SPEAKER_BATCH_SIZE]
            self.logger.logger.info(
                f"🔄 Processing batch {i//TTS_SPEAKER_BATCH_SIZE + 1}: {len(batch_speakers)} speakers"
            )

            for speaker_idx, speaker in enumerate(batch_speakers):
                self.logger.logger.info(
                    f"  👤 Processing speaker {speaker} ({speaker_idx + 1}/{len(batch_speakers)})"
                )

                non_singing_segs = [
                    seg
                    for seg in transcribe_data["segments"]
                    if seg.get("speaker") == speaker
                    and not seg.get("is_singing", False)
                    and seg.get("no_speech_prob", 1.0) < 0.5
                ]

                if non_singing_segs:
                    non_singing_segs.sort(key=lambda x: x["start"])

                    slices = []
                    total_duration = 0
                    min_duration_seconds = min_duration_minutes * 60  # Convert minutes to seconds

                    # Concatenate segments until we reach minimum duration
                    for seg in non_singing_segs:
                        start = seg["start"]
                        end = seg["end"]
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        slice_w = waveform[:, start_sample:end_sample]
                        slices.append(slice_w)
                        total_duration += end - start

                        # Stop if we've reached the minimum duration
                        if total_duration >= min_duration_seconds:
                            break

                    if slices:
                        concatenated = torch.cat(slices, dim=1)
                        ref_path = os.path.join(refs_dir, f"{speaker}_long.wav")
                        torchaudio.save(ref_path, concatenated, sr)

                        # Clean up slices and concatenated tensor
                        del slices
                        del concatenated
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Transcribe the reference audio to get ref_text
                        ref_text = transcribe_reference_audio(
                            ref_path, transcribe_data.get("language", "ja")
                        )

                        # Update refs_by_speaker with both audio path and transcribed text
                        refs_by_speaker[speaker] = {
                            "audio_path": f"refs/{speaker}_long.wav",
                            "ref_text": ref_text,
                        }

                        self.logger.logger.info(
                            f"    ✅ Reference created for {speaker}: {total_duration:.2f}s (min: {min_duration_seconds:.0f}s)"
                        )
                        if ref_text:
                            pass
                    else:
                        self.logger.logger.warning(
                            f"    ⚠️  No valid segments found for speaker {speaker}"
                        )
                else:
                    self.logger.logger.warning(
                        f"    ⚠️  No non-singing segments found for speaker {speaker}"
                    )

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        default_ref = None
        if refs_by_speaker:
            self.logger.logger.info("🎯 Creating default reference from first speaker")
            first_speaker = list(refs_by_speaker.keys())[0]
            default_ref_path = os.path.join(refs_dir, "default.wav")
            source_path = os.path.join(
                tmp_path, refs_by_speaker[first_speaker]["audio_path"]
            )
            shutil.copy(source_path, default_ref_path)

            # Transcribe the default reference audio
            default_ref_text = transcribe_reference_audio(
                default_ref_path, transcribe_data.get("language", "ja")
            )
            default_ref = {"audio_path": "refs/default.wav", "ref_text": default_ref_text}

            self.logger.logger.info(
                f"  ✅ Default reference created from speaker {first_speaker}"
            )
        else:
            self.logger.logger.info(
                "⚠️  No speaker-specific references found, creating fallback default reference"
            )
            all_non_singing_segs = [
                seg
                for seg in transcribe_data["segments"]
                if not seg.get("is_singing", False) and seg.get("no_speech_prob", 1.0) < 0.5
            ]
            if all_non_singing_segs:
                all_non_singing_segs.sort(key=lambda x: x["start"])
                slices = []
                total_duration = 0
                min_duration_seconds = min_duration_minutes * 60  # Convert minutes to seconds

                # Concatenate segments until we reach minimum duration
                for seg in all_non_singing_segs:
                    start = seg["start"]
                    end = seg["end"]
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    slice_w = waveform[:, start_sample:end_sample]
                    slices.append(slice_w)
                    total_duration += end - start

                    # Stop if we've reached the minimum duration
                    if total_duration >= min_duration_seconds:
                        break

                if slices:
                    concatenated = torch.cat(slices, dim=1)
                    default_ref_path = os.path.join(refs_dir, "default_long.wav")
                    torchaudio.save(default_ref_path, concatenated, sr)

                    # Clean up slices and concatenated tensor
                    del slices
                    del concatenated
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Transcribe the default reference audio
                    default_ref_text = transcribe_reference_audio(
                        default_ref_path, transcribe_data.get("language", "ja")
                    )
                    default_ref = {
                        "audio_path": "refs/default_long.wav",
                        "ref_text": default_ref_text,
                    }

                    self.logger.logger.info(f"  ✅ Fallback default reference created: {total_duration:.2f}s (min: {min_duration_seconds:.0f}s)")
            else:
                self.logger.logger.warning("  ⚠️  No valid segments found for default reference")

        # Clean up the main waveform tensor after all processing
        del waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stage_data = {
            "stage": "build_refs",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "errors": [],
            "refs_by_speaker": refs_by_speaker,
            "default_ref": default_ref,
            "extraction_criteria": f"concatenated non-singing segments until >= {min_duration_minutes} minute(s) per speaker, with re-transcription",
        }

        return stage_data

# Optional Orpheus import - handle gracefully if not installed
try:
    from orpheus_speech import OrpheusModel
    ORPHEUS_AVAILABLE = True
except ImportError:
    ORPHEUS_AVAILABLE = False
    logging.warning("Orpheus-TTS not available. Install with: pip install orpheus-speech")



def cleanup_memory():
    """
    Clean up memory by collecting garbage and clearing GPU cache.
    Call this periodically during long-running TTS operations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_memory_usage(stage_name: str):
    """
    Log current memory usage for monitoring purposes.

    Args:
        stage_name: Name of the current processing stage
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        logging.info(
            f"[{stage_name}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def transcribe_reference_audio(audio_path: str, language: str = "ja") -> str:
    """
    Transcribe a reference audio file using Whisper.

    Args:
        audio_path: Path to the audio file to transcribe
        language: Language code for transcription

    Returns:
        Transcribed text from the audio file
    """
    try:
        # Load Whisper model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        whisper_model = whisper.load_model("turbo", device=device)

        # Transcribe the audio
        result = whisper_model.transcribe(audio_path, language=language)

        # Clean up
        del whisper_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text = result["text"]
        if isinstance(text, list):
            text = " ".join(text)
        return text.strip()
    except Exception as e:
        logging.warning(f"Failed to transcribe reference audio {audio_path}: {e}")
        return ""


def generate_tts_for_speaker_orpheus(
    segments,
    speaker,
    ref_audios_by_speaker,
    default_ref,
    tmp_path,
    target_sr,
    orpheus_model,
    ref_text=None,
    emotion_data=None,
):
    """
    Generate TTS audio segments using Orpheus-TTS for a specific speaker.

    Args:
        segments: List of segments for this speaker
        speaker: Speaker ID
        ref_audios_by_speaker: Dict mapping speakers to reference audio paths
        default_ref: Default reference audio path
        tmp_path: Temporary directory path
        target_sr: Target sample rate
        orpheus_model: Loaded Orpheus model
        ref_text: Reference text for voice cloning
        emotion_data: Optional emotion data

    Returns:
        List of TTS segments with audio paths and timing
    """
    logger = get_logger("tts-orchestrator")
    tts_segments = []

    # Load TTS configuration for Orpheus settings
    tts_config = load_tts_config()
    if not isinstance(tts_config, dict):
        tts_config = {}
    if isinstance(tts_config, dict):
        orpheus_config = tts_config.get("tts_methods", {}).get("orpheus", {}).get("config", {})
    else:
        orpheus_config = {}

    # Get reference audio for this speaker
    ref_audio = ref_audios_by_speaker.get(speaker, default_ref)
    if not ref_audio:
        logger.logger.warning(f"No reference audio found for speaker {speaker}")
        return tts_segments

    # Load reference audio
    ref_audio_path = os.path.join(tmp_path, ref_audio)
    ref_wav, ref_sr = torchaudio.load(ref_audio_path)

    # Get voice name from config or use default
    voice = orpheus_config.get("voice", "tara")

    # Process each segment
    for i, seg in enumerate(segments):
        try:
            text = seg.get("translated_text", "")
            if not text.strip():
                continue

            # Build prompt with voice and emotion tags if enabled
            prompt = f"{voice}: {text}"

            # Add emotion tags if emotion stage is active and enabled
            if emotion_data and orpheus_config.get("emotion_tags", True):
                # Get emotion for this segment (simplified - would need proper emotion data)
                emotion = getattr(seg, 'emotion', None)
                if emotion and emotion != "neutral":
                    prompt = f"<{emotion}>{prompt}</emotion>"

            # Generate audio using Orpheus
            logger.logger.info(f"Generating Orpheus TTS for segment {i+1}/{len(segments)}")

            # Generate audio with reference conditioning
            generated_audio = orpheus_model.generate(
                prompt,
                reference_audio=ref_wav,
                reference_text=ref_text or text,
                stream=True,
                temperature=orpheus_config.get("temperature", 0.7),
                repetition_penalty=orpheus_config.get("repetition_penalty", 1.1),
            )

            # Convert to tensor and save
            audio_tensor = torch.tensor(generated_audio, dtype=torch.float32).unsqueeze(0)

            # Adjust duration to match segment timing
            target_duration = seg["end"] - seg["start"]
            current_duration = audio_tensor.shape[1] / target_sr

            if current_duration != target_duration:
                if current_duration < target_duration:
                    # Pad with silence
                    silence_duration = target_duration - current_duration
                    silence_samples = int(silence_duration * target_sr)
                    silence = torch.zeros(1, silence_samples)
                    audio_tensor = torch.cat([audio_tensor, silence], dim=1)
                else:
                    # Trim audio
                    max_samples = int(target_duration * target_sr)
                    audio_tensor = audio_tensor[:, :max_samples]

            # Save audio file
            seg_filename = f"tts/seg_{speaker}_{i}.wav"
            seg_path = os.path.join(tmp_path, seg_filename)
            torchaudio.save(seg_path, audio_tensor, target_sr)

            # Create TTS segment entry
            tts_segment = {
                "path": seg_filename,
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "duration": target_duration,
                "text": text,
                "tts_method": "orpheus"
            }

            tts_segments.append(tts_segment)

            # Clean up memory
            del audio_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.logger.error(f"Failed to generate TTS for segment {i}: {e}")
            continue

    return tts_segments


def generate_dubbed_segments(
    tmp_path, metadata_path, inputs_data, tts_method="edge-tts", **kwargs
) -> dict:
    """
    Generate dubbed audio segments from the translated JSON file, skipping singing segments and those with empty translated text.

    For each valid segment:
    - Uses speaker-specific reference audio if available, otherwise falls back to default_ref.
    - Generates TTS audio using the specified TTS method.
    - Adjusts the generated audio duration to match the segment's (end - start) seconds by trimming or padding with silence.
    - This ensures timeline synchronization during later mixing.

    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data.
        tts_method: TTS method to use for audio generation (default: edge-tts).
        **kwargs: Additional arguments.

    Returns:
        Stage data with tts_segments, rvc_models_used, total_duration, tts_method.
    """
    # Initialize logger
    logger = get_logger("tts-orchestrator")

    # Log selected TTS method
    logger.log_tts_method(tts_method)
    logging.info(f"Using TTS method: {tts_method}")

    # Load previous results
    translate_data = inputs_data["translate"]
    transcribe_data = inputs_data["transcribe"]
    build_refs_data = inputs_data["build_refs"]

    # Extract audio paths from the new structure
    ref_audios_by_speaker = {}
    ref_texts_by_speaker = {}
    for speaker, ref_data in build_refs_data["refs_by_speaker"].items():
        if isinstance(ref_data, dict):
            ref_audios_by_speaker[speaker] = ref_data["audio_path"]
            ref_texts_by_speaker[speaker] = ref_data["ref_text"]
        else:
            # Fallback for old structure
            ref_audios_by_speaker[speaker] = ref_data
            ref_texts_by_speaker[speaker] = ""

    # Handle default_ref
    if isinstance(build_refs_data["default_ref"], dict):
        default_ref = build_refs_data["default_ref"]["audio_path"]
        default_ref_text = build_refs_data["default_ref"]["ref_text"]
    else:
        # Fallback for old structure
        default_ref = build_refs_data["default_ref"]
        default_ref_text = ""

    # Create TTS engine using factory pattern
    try:
        engine = create_tts_engine(tts_method)
        engine.load_model()
    except (ValueError, TTSEngineLoadError) as e:
        logging.error(f"Failed to create TTS engine: {e}")
        # Fallback to edge-tts
        logging.info("Falling back to edge-tts")
        engine = create_tts_engine("edge")
        engine.load_model()
        tts_method = "edge"

    # Create output directory if it doesn't exist
    tts_dir = os.path.join(tmp_path, "tts")
    os.makedirs(tts_dir, exist_ok=True)

    data = translate_data
    tts_segments = []

    # Group segments by speaker for batch processing
    segments_by_speaker = {}
    total_segments = len(data.get("segments", []))
    processed_segments = 0
    skipped_segments = 0

    logger.logger.info(f"📊 Processing {total_segments} total segments")

    for seg in data.get("segments", []):
        # Skip singing segments and those with empty translated text
        if seg.get("is_singing", False) or not seg.get("translated_text", "").strip():
            skipped_segments += 1
            continue  # Skipping logic for singing and silent/empty text

        speaker = seg.get("speaker")
        if speaker not in segments_by_speaker:
            segments_by_speaker[speaker] = []
        segments_by_speaker[speaker].append(seg)
        processed_segments += 1

    logger.logger.info(f"✅ {processed_segments} segments will be processed")
    if skipped_segments > 0:
        logger.logger.info(f"⏭️  {skipped_segments} segments skipped (singing/empty)")

    # Log speaker distribution
    logger.logger.info(f"👥 Processing {len(segments_by_speaker)} unique speakers")
    for speaker, segments in segments_by_speaker.items():
        logger.logger.info(f"  👤 {speaker}: {len(segments)} segments")

    # Process each speaker's segments using the TTS engine
    processed_speakers = 0
    total_speakers = len(segments_by_speaker)

    for speaker_idx, (speaker, segments) in enumerate(segments_by_speaker.items()):
        # Log speaker batch start
        logger.log_speaker_batch(speaker, len(segments))

        # Use transcribed reference text if available, otherwise fall back to original text
        ref_text = ref_texts_by_speaker.get(speaker, default_ref_text)
        if not ref_text and default_ref_text:
            ref_text = default_ref_text

        # Log reference audio information
        ref_audio = ref_audios_by_speaker.get(speaker, default_ref)
        if ref_text:
            pass

        try:
            # Validate language support before TTS generation
            target_lang = translate_data.get("target_lang", "en")
            engine.validate_language(target_lang)

            # Generate TTS for each segment using the engine
            for segment in segments:
                try:
                    tts_segment = engine.generate_segment(
                        segment,
                        ref_audio,
                        ref_text,
                        tmp_path=tmp_path,
                        language=target_lang,
                        emotion_data=None,  # TODO: Pass emotion data if available
                    )
                    tts_segments.append(tts_segment)
                except TTSEngineGenerationError as e:
                    logger.logger.warning(f"  ⚠️  Failed to generate TTS for segment: {e}")
                    continue

            # Log successful speaker processing
            logger.logger.info(
                f"  ✅ Speaker {speaker} completed: {len(segments)} segments generated"
            )

        except TTSEngineValidationError as e:
            logger.logger.warning(f"  ⚠️  Language validation failed for speaker {speaker}: {e}")
            continue
        except Exception as e:
            logger.log_error("tts_generation", e, f"speaker {speaker}")
            logger.logger.warning(f"  ⚠️  Failed to process speaker {speaker}: {e}")
            continue

        # Clean up after each speaker and periodically clear GPU cache
        processed_speakers += 1
        if processed_speakers % 3 == 0:  # Clear cache every 3 speakers
            cleanup_memory()

    # Clean up engine
    engine.cleanup()

    total_duration = sum(seg["end"] - seg["start"] for seg in tts_segments)
    tts_segments = sorted(tts_segments, key=lambda seg: seg["start"])

    stage_data = {
        "stage": "generate_tts",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "tts_segments": tts_segments,
        "rvc_models_used": {},  # TODO: Track model usage if needed
        "total_duration": total_duration,
        "tts_method": tts_method,
    }

    # Add method-specific data
    if tts_method == "xtts":
        stage_data["xtts_models_used"] = {speaker: "xtts_v2" for speaker in segments_by_speaker.keys()}
    elif tts_method == "orpheus":
        tts_config = load_tts_config()
        if isinstance(tts_config, dict):
            orpheus_config = tts_config.get("tts_methods", {}).get("orpheus", {}).get("config", {})
        else:
            orpheus_config = {}
        model_type = orpheus_config.get("model_type", "finetuned")
        if model_type == "finetuned":
            model_path = "canopyai/Orpheus-3B-0.1-ft"
        elif model_type == "pretrained":
            model_path = "canopyai/Orpheus-3B-0.1"
        else:  # multilingual
            model_path = "canopyai/Orpheus-3B-0.1-multilingual"
        stage_data["orpheus_models_used"] = {speaker: model_path for speaker in segments_by_speaker.keys()}

    return stage_data




def build_speaker_refs(tmp_path, metadata_path, inputs_data, **kwargs) -> dict:
    """
    Extract reference audios per speaker from original vocals.

    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data.
        **kwargs: Additional arguments.

    Returns:
        Stage data with refs_by_speaker, default_ref, extraction_criteria.
    """
    builder = ReferenceBuilder()
    return builder.build_speaker_refs(tmp_path, metadata_path, inputs_data, **kwargs)
