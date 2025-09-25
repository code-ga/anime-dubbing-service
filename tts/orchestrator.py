import os
import shutil
import logging
import torch
import torchaudio
import whisper
import gc
import yaml
from datetime import datetime
from typing import List, Dict, Optional
from vocos import Vocos
from tts.F5 import generate_tts_custom, generate_tts_for_speaker, F5TTS, validate_language as validate_f5_language
from tts.edge_tts import generate_tts_for_speaker as generate_tts_for_speaker_edge, validate_language as validate_edge_language
from tts.xtts import generate_tts_for_speaker_xtts, validate_language as validate_xtts_language
from utils.metadata import load_previous_result
from utils.logger import get_logger

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

    speaker_embeddings = transcribe_data["speaker_embeddings"]

    # Create output directory if it doesn't exist
    tts_dir = os.path.join(tmp_path, "tts")
    os.makedirs(tts_dir, exist_ok=True)

    data = translate_data
    tts_segments = []
    SR = 24000  # Vocos-decoded output sample rate

    # Initialize TTS method-specific resources
    rvc_models_used = {}
    xtts_models_used = {}
    model = None
    vocos = None
    if tts_method == "edge":
        # For edge, we don't need F5-TTS models or RVC models
        # Voice selection is handled by edge-tts voice mapping
        logging.info("Using Edge TTS for TTS generation")
        target_sr = 24000  # Standard sample rate for edge-tts output
    elif tts_method == "xtts":
        # XTTS model is loaded by the XTTS module
        logging.info("Using XTTS for TTS generation")
        target_sr = 24000  # XTTS output sample rate

        for speaker in ref_audios_by_speaker:
            xtts_models_used[speaker] = "xtts_v2"
    elif tts_method == "rvc":
        # RVC as fallback - use F5-TTS models for RVC processing
        logging.info("Using RVC for TTS generation")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = F5TTS(model="F5TTS_Base")
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        target_sr = 24000  # Vocos-decoded output sample rate

        for speaker in ref_audios_by_speaker:
            # Placeholder for RVC model selection based on embedding
            # In real implementation, compute similarity to known models or download if unknown
            rvc_model_id = "default_rvc_model"  # Replace with actual logic using speaker_embeddings[speaker]
            rvc_models_used[speaker] = rvc_model_id
    else:
        # Load F5-TTS models for traditional TTS method (default/f5)
        logging.info("Using F5-TTS for TTS generation")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = F5TTS(model="F5TTS_Base")
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        target_sr = 24000  # Vocos-decoded output sample rate

        for speaker in ref_audios_by_speaker:
            # Placeholder for RVC model selection based on embedding
            # In real implementation, compute similarity to known models or download if unknown
            rvc_model_id = "default_rvc_model"  # Replace with actual logic using speaker_embeddings[speaker]
            rvc_models_used[speaker] = rvc_model_id

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

    # Process each speaker's segments in batch with memory management
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
            if tts_method == "edge":
                # Validate language support for Edge-TTS
                validate_edge_language(target_lang)
                # Use edge-tts implementation
                speaker_tts_segments = generate_tts_for_speaker_edge(
                    segments,
                    speaker,
                    ref_audios_by_speaker,
                    default_ref,
                    tmp_path,
                    target_sr,
                    language=target_lang,
                    ref_text=ref_text,
                )
            elif tts_method == "xtts":
                # Validate language support for XTTS
                validate_xtts_language(target_lang)
                # Use XTTS implementation
                speaker_tts_segments = generate_tts_for_speaker_xtts(
                    segments,
                    speaker,
                    ref_audios_by_speaker,
                    default_ref,
                    tmp_path,
                    target_sr,
                    language=target_lang,
                    ref_text=ref_text,
                    emotion_data=None,  # TODO: Pass emotion data if available
                )
            elif tts_method == "rvc":
                # Validate language support for F5-TTS (used for RVC)
                validate_f5_language(target_lang)
                # Use F5-TTS implementation for RVC - model and vocos are guaranteed to be initialized
                assert model is not None, "F5-TTS model should be initialized"
                assert vocos is not None, "Vocos model should be initialized"
                speaker_tts_segments = generate_tts_for_speaker(
                    segments,
                    speaker,
                    ref_audios_by_speaker,
                    default_ref,
                    tmp_path,
                    target_sr,
                    model,
                    vocos,
                    ref_text=ref_text,
                )
            else:
                # Validate language support for F5-TTS (f5 method)
                validate_f5_language(target_lang)
                # Use F5-TTS implementation - model and vocos are guaranteed to be initialized in the else branch
                assert model is not None, "F5-TTS model should be initialized"
                assert vocos is not None, "Vocos model should be initialized"
                speaker_tts_segments = generate_tts_for_speaker(
                    segments,
                    speaker,
                    ref_audios_by_speaker,
                    default_ref,
                    tmp_path,
                    target_sr,
                    model,
                    vocos,
                    ref_text=ref_text,
                )
            tts_segments.extend(speaker_tts_segments)

            # Log successful speaker processing
            logger.logger.info(
                f"  ✅ Speaker {speaker} completed: {len(speaker_tts_segments)} segments generated"
            )

        except Exception as e:
            logger.log_error("tts_generation", e, f"speaker {speaker}")
            logger.logger.warning(f"  ⚠️  Failed to process speaker {speaker}: {e}")
            continue

        # Clean up after each speaker and periodically clear GPU cache
        processed_speakers += 1
        if processed_speakers % 3 == 0:  # Clear cache every 3 speakers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Clean up models and free memory after processing all speakers
    if tts_method not in ["edge", "xtts"] and model is not None and vocos is not None:
        # Only clean up F5-TTS models if they were used (f5 and rvc use F5-TTS models)
        del model
        del vocos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_duration = sum(seg["end"] - seg["start"] for seg in tts_segments)
    tts_segments = sorted(tts_segments, key=lambda seg: seg["start"])

    stage_data = {
        "stage": "generate_tts",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "tts_segments": tts_segments,
        "rvc_models_used": rvc_models_used,
        "total_duration": total_duration,
        "tts_method": tts_method,
    }

    # Add XTTS-specific data if using XTTS
    if tts_method == "xtts":
        stage_data["xtts_models_used"] = xtts_models_used

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
    # Initialize logger
    logger = get_logger("tts-orchestrator")

    # Load TTS configuration
    tts_config = load_tts_config()
    min_duration_minutes = tts_config.get("reference_audio", {}).get("min_duration_minutes", 1)

    logger.logger.info("🎯 Starting reference audio extraction")
    logger.logger.info(f"📏 Using minimum reference duration: {min_duration_minutes} minute(s)")

    transcribe_data = load_previous_result(metadata_path, "transcribe")
    separate_data = inputs_data.get("separate_audio")

    # Use vocals_path if separate_audio was run, otherwise fallback to full_wav_path
    if separate_data and "vocals_path" in separate_data:
        vocals_path = os.path.join(tmp_path, separate_data["vocals_path"])
        logger.logger.info(f"📁 Loading separated vocals from: {vocals_path}")
    else:
        # Fallback to full audio when audio separation is skipped
        convert_data = inputs_data.get("convert_mp4_to_wav", {})
        vocals_path = os.path.join(tmp_path, convert_data.get("full_wav_path", "full.wav"))
        logger.logger.info(f"📁 Loading full audio (audio separation skipped) from: {vocals_path}")

    waveform, sr = torchaudio.load(vocals_path)
    logger.logger.info(f"🎵 Audio loaded: {waveform.shape[1]/sr:.2f} seconds at {sr}Hz")

    refs_dir = os.path.join(tmp_path, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    logger.log_file_operation("create", refs_dir, True)

    refs_by_speaker = {}
    speakers = set(
        seg.get("speaker") for seg in transcribe_data["segments"] if seg.get("speaker")
    )

    logger.logger.info(
        f"👥 Found {len(speakers)} speakers: {', '.join(sorted(speakers))}"
    )

    # Process speakers in batches to manage memory usage
    speaker_list = list(speakers)

    for i in range(0, len(speaker_list), TTS_SPEAKER_BATCH_SIZE):
        batch_speakers = speaker_list[i : i + TTS_SPEAKER_BATCH_SIZE]
        logger.logger.info(
            f"🔄 Processing batch {i//TTS_SPEAKER_BATCH_SIZE + 1}: {len(batch_speakers)} speakers"
        )

        for speaker_idx, speaker in enumerate(batch_speakers):
            logger.logger.info(
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

                    logger.logger.info(
                        f"    ✅ Reference created for {speaker}: {total_duration:.2f}s (min: {min_duration_seconds:.0f}s)"
                    )
                    if ref_text:
                        pass
                else:
                    logger.logger.warning(
                        f"    ⚠️  No valid segments found for speaker {speaker}"
                    )
            else:
                logger.logger.warning(
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
        logger.logger.info("🎯 Creating default reference from first speaker")
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

        logger.logger.info(
            f"  ✅ Default reference created from speaker {first_speaker}"
        )
    else:
        logger.logger.info(
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

                logger.logger.info(f"  ✅ Fallback default reference created: {total_duration:.2f}s (min: {min_duration_seconds:.0f}s)")
        else:
            logger.logger.warning("  ⚠️  No valid segments found for default reference")

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
