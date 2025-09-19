from typing import List, Optional
import numpy as np
import pandas as pd
import torchaudio
import whisper
import json
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import os
import gc
from typing import TypedDict, Optional


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


def transcript(tmp_path, metadata_path, inputs_data, language="ja"):
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

    audioFilePath = os.path.join(tmp_path, inputs_data["separate_audio"]["vocals_path"])
    print(audioFilePath)

    whisper_model = whisper.load_model("turbo", device=DEVICE)
    result = whisper_model.transcribe(audioFilePath)
    audio_transcript: list[WhisperSegment] = []
    audio, sr = torchaudio.load(audioFilePath)
    saving_dir = os.path.join(tmp_path, "whisper_audio")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    for c in result["segments"]:
        if isinstance(c, str):
            raise ValueError
        saving_path = os.path.join(saving_dir, f"{c['start']}_{c['end']}.wav")
        torchaudio.save(
            saving_path,
            audio[:, int(c["start"] * sr) : int(c["end"] * sr)],
            sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        # audio_transcript.append(WhisperSegment(**c, audioFilePath=saving_path))
        whisper_segment: WhisperSegment = {
            "audioFilePath": saving_path,
            "seek": c["seek"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "tokens": c["tokens"],
            "temperature": c["temperature"],
            "avg_logprob": c["avg_logprob"],
            "compression_ratio": c["compression_ratio"],
            "no_speech_prob": c["no_speech_prob"],
        }
        audio_transcript.append(whisper_segment)
    whisper_transcript: WhisperResult = {
        "segments": audio_transcript,
        "language": str(result["language"]),
        "text": str(result["text"]),
    }
    del whisper_model
    del result
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
