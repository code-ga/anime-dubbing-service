from typing import List, TypedDict
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch

from transcription.whisper import AssignWordSpeakersResult, DiarizationResult

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionModel:
    def __init__(self):
        self.model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(DEVICE)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def __call__(self, audio):
        pass

class Emotion(TypedDict):
    emotion: str
    score: float


class EmotionTranscript(DiarizationResult):
    emotional: Emotion
class EmotionTranscriptResult(TypedDict):
    speaker_embeddings: dict[str, list[float]]
    segments: List[EmotionTranscript]
    language: str


def emotional_transcript(
    whisper_transcript_output: AssignWordSpeakersResult,
):
    pass
