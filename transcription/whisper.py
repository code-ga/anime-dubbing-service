from typing import List, Optional
import numpy as np
import pandas as pd
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import os
import gc
from typing import TypedDict, Optional
from tqdm import tqdm

# Import silero VAD utils
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WhisperSegment(TypedDict):
    seek: int
    start: float
    end: float
    text: str
    tokens: torch.Tensor
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    audioFilePath: str

    # def __init__(self, **segment_data):
    #     # check if all keys are present
    #     """
    #     Initialize a WhisperSegment instance.

    #     Parameters
    #     ----------
    #     **segment_data : dict
    #         A dictionary containing the following keys:

    #         - seek: int
    #         - start: float
    #         - end: float
    #         - text: str
    #         - tokens: torch.Tensor
    #         - temperature: float
    #         - avg_logprob: float
    #         - compression_ratio: float
    #         - no_speech_prob: float
    #         - audioFilePath: str

    #     Raises
    #     ------
    #     ValueError
    #         If any of the required keys are missing from the `segment_data` dictionary.
    #     """
    #     for key in [
    #         "seek",
    #         "start",
    #         "end",
    #         "text",
    #         "tokens",
    #         "temperature",
    #         "avg_logprob",
    #         "compression_ratio",
    #         "no_speech_prob",
    #         "audioFilePath",
    #     ]:
    #         if key not in segment_data:
    #             raise ValueError(f"Missing key {key} in segment data")
    #     self.seek = segment_data["seek"]
    #     self.start = segment_data["start"]
    #     self.end = segment_data["end"]
    #     self.text = segment_data["text"]
    #     self.tokens = segment_data["tokens"]
    #     self.temperature = segment_data["temperature"]
    #     self.avg_logprob = segment_data["avg_logprob"]
    #     self.compression_ratio = segment_data["compression_ratio"]
    #     self.no_speech_prob = segment_data["no_speech_prob"]
    #     self.audioFilePath = segment_data["audioFilePath"]


class WhisperResult(TypedDict):
    segments: List[WhisperSegment]
    text: str
    language: str


def apply_vad(
    audio_path: str, threshold: float = 0.5, min_speech_duration: float = 0.25
) -> List[dict[str, float]]:
    """
    Apply Silero VAD to audio file to get speech segments.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    threshold : float
        Speech detection threshold (default: 0.5)
    min_speech_duration : float
        Minimum speech duration in seconds (default: 0.25)

    Returns
    -------
    List[tuple]
        List of (start, end) timestamps in seconds
    """
    # Load Silero VAD model
    model = load_silero_vad()

    # Load audio
    audio = read_audio(audio_path, sampling_rate=16000)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        return_seconds=True,
    )

    return speech_timestamps


def transcript(tmp_path, metadata_path, inputs_data, language="ja", disable_vad=False):
    """
    Transcribe an audio file using Whisper and assign speakers to words using Pyannote.

    Parameters
    ----------
    tmp_path : str
    metadata_path : str
    inputs_data : dict
    language : str, optional

    Returns
    -------
    dict
        Stage data with segments, language, text, speaker_embeddings.

    """

    # Use vocals_path if separate_audio was run, otherwise fallback to full_wav_path
    separate_audio_data = inputs_data.get("separate_audio")
    if separate_audio_data and "vocals_path" in separate_audio_data:
        audioFilePath = os.path.join(tmp_path, separate_audio_data["vocals_path"])
        print(f"Using separated vocals: {audioFilePath}")
    else:
        # Fallback to full audio when audio separation is skipped
        convert_data = inputs_data.get("convert_mp4_to_wav", {})
        audioFilePath = os.path.join(tmp_path, convert_data.get("full_wav_path", "full.wav"))
        print(f"Using full audio (audio separation skipped): {audioFilePath}")

    # Apply VAD to get speech segments (unless disabled)
    if disable_vad:
        print("VAD disabled, using full audio for transcription...")
        speech_segments = [{"start": 0, "end": None}]  # Use entire audio
    else:
        print("Applying VAD to detect speech segments...")
        speech_segments = apply_vad(audioFilePath, threshold=0.5, min_speech_duration=0.1)

        if not speech_segments:
            print("No speech segments detected by VAD")
            return {
                "segments": [],
                "language": language,
                "text": "",
                "speaker_embeddings": {},
            }

        print(f"Found {len(speech_segments)} speech segments")

    # Load full audio for segment extraction
    audio, sr = torchaudio.load(audioFilePath)
    saving_dir = os.path.join(tmp_path, "whisper_audio")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    whisper_model = whisper.load_model("turbo", device=DEVICE)
    audio_transcript: list[WhisperSegment] = []

    # Process each speech segment with Whisper
    print("Transcribing speech segments...")
    all_texts = []
    detected_language = None

    for i, segment in enumerate(
        tqdm(speech_segments, desc="Processing speech segments")
    ):
        start_time = segment["start"]
        end_time = segment["end"] if segment["end"] is not None else None
        if end_time is not None:
            print(
                f"Processing segment {i+1}/{len(speech_segments)}: {start_time:.2f}s to {end_time:.2f}s"
            )
        else:
            print(
                f"Processing segment {i+1}/{len(speech_segments)}: {start_time:.2f}s to end of audio"
            )

        # Extract segment audio
        start_sample = int(start_time * sr)
        if end_time is not None:
            end_sample = int(end_time * sr)
        else:
            end_sample = audio.shape[1]  # Use entire audio length
        segment_audio = audio[:, start_sample:end_sample]

        # Save temporary audio file for Whisper
        temp_path = os.path.join(saving_dir, f"temp_segment_{i}.wav")
        torchaudio.save(
            temp_path,
            segment_audio,
            sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        # Transcribe segment
        segment_result = whisper_model.transcribe(temp_path)

        # Store language from first segment
        if detected_language is None:
            detected_language = segment_result["language"]

        # Process segment results and adjust timings to original audio
        for c in segment_result["segments"]:
            if isinstance(c, str):
                raise ValueError

            # Adjust timings to match original audio
            adjusted_start = start_time + c["start"]
            adjusted_end = start_time + c["end"]

            saving_path = os.path.join(
                saving_dir, f"{adjusted_start}_{adjusted_end}.wav"
            )
            torchaudio.save(
                saving_path,
                audio[:, int(adjusted_start * sr) : int(adjusted_end * sr)],
                sr,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            whisper_segment: WhisperSegment = {
                "audioFilePath": saving_path,
                "seek": c["seek"],
                "start": adjusted_start,
                "end": adjusted_end,
                "text": c["text"],
                "tokens": c["tokens"],
                "temperature": c["temperature"],
                "avg_logprob": c["avg_logprob"],
                "compression_ratio": c["compression_ratio"],
                "no_speech_prob": c["no_speech_prob"],
            }
            audio_transcript.append(whisper_segment)

        # Collect text for full transcript
        all_texts.append(segment_result["text"])

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    # Create combined result
    combined_text = " ".join(all_texts)
    whisper_transcript: WhisperResult = {
        "segments": audio_transcript,
        "language": str(detected_language or language),
        "text": combined_text,
    }
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    ).to(DEVICE)
    with ProgressHook() as hook:
        diarization, embeddings = pipeline(
            audioFilePath, return_embeddings=True, hook=hook
        )

    diarize_df = pd.DataFrame(
        diarization.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
    speaker_embeddings = {
        speaker: embeddings[s].tolist()
        for s, speaker in enumerate(diarization.labels())
    }

    result_with_speakers = assign_word_speakers(
        diarize_df, whisper_transcript, speaker_embeddings
    )

    # Initialize is_singing flag
    for seg in result_with_speakers["segments"]:
        seg["is_singing"] = False
        speaker = seg.get("speaker")
        if speaker:
            seg["speaker_embedding"] = speaker_embeddings.get(speaker, [])

    # Clean up Pyannote pipeline and free memory
    del pipeline
    del diarization
    del embeddings
    del diarize_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_with_speakers


class DiarizationResult(TypedDict, total=False):
    seek: int
    start: float
    end: float
    text: str
    tokens: torch.Tensor
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    audioFilePath: str
    speaker: Optional[str]
    is_singing: bool
    speaker_embedding: Optional[list[float]]


class AssignWordSpeakersResult(TypedDict):
    speaker_embeddings: dict[str, list[float]]
    segments: List[DiarizationResult]
    language: str
    text: str


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: WhisperResult,
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> AssignWordSpeakersResult:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    result: List[DiarizationResult] = []
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        speaker = None
        intersections = np.minimum(diarize_df["end"], seg["end"]) - np.maximum(
            diarize_df["start"], seg["start"]
        )
        inter_series = pd.Series(intersections, index=diarize_df.index)
        if not fill_nearest:
            valid = inter_series > 0
            if not valid.any():
                speaker = None
            else:
                speaker_sums = (
                    inter_series[valid]
                    .groupby(diarize_df.loc[valid, "speaker"])
                    .sum()
                    .sort_values(ascending=False)
                )
                speaker = speaker_sums.index[0]
        else:
            speaker_sums = (
                inter_series.groupby(diarize_df["speaker"])
                .sum()
                .sort_values(ascending=False)
            )
            speaker = speaker_sums.index[0]
        result.append({**seg, "speaker": speaker})

        # assign speaker to words
        # if "words" in seg:
        #     for word in seg["words"]:
        #         if "start" in word:
        #             diarize_df["intersection"] = np.minimum(
        #                 diarize_df["end"], word["end"]
        #             ) - np.maximum(diarize_df["start"], word["start"])
        #             diarize_df["union"] = np.maximum(
        #                 diarize_df["end"], word["end"]
        #             ) - np.minimum(diarize_df["start"], word["start"])
        #             # remove no hit
        #             if not fill_nearest:
        #                 dia_tmp = diarize_df[diarize_df["intersection"] > 0]
        #             else:
        #                 dia_tmp = diarize_df
        #             if len(dia_tmp) > 0:
        #                 # sum over speakers
        #                 speaker = (
        #                     dia_tmp.groupby("speaker")["intersection"]
        #                     .sum()
        #                     .sort_values(ascending=False)
        #                     .index[0]
        #                 )
        #                 word["speaker"] = speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        return {
            "speaker_embeddings": speaker_embeddings,
            "segments": result,
            "language": transcript_result["language"],
            "text": transcript_result["text"],
        }

    return {
        "speaker_embeddings": {},
        "segments": result,
        "language": transcript_result["language"],
        "text": transcript_result["text"],
    }
