from pydantic import BaseModel
from typing import Optional


class VoiceSeparatorSegment(BaseModel):
    start: float
    end: float
    speaker: str
    filename: str
    id: int


class VoiceSeparator(BaseModel):
    audio_files: list[str]
    speakers_time: dict[str, int]
    all_segments: list[VoiceSeparatorSegment]


class VoiceTranscript(BaseModel):
    start: float
    end: float
    text: str
    id: int
    seek: int
    emotion: Optional[str]


class VoiceTranscriptSegment(BaseModel):
    text: str
    language: str
    segments: list[VoiceTranscript]


class TranscriptExtendedSegment(VoiceSeparatorSegment):
    transcript: VoiceTranscriptSegment

    @classmethod
    def from_voice_separator_segment(
        cls,
        voice_separator_segment: VoiceSeparatorSegment,
        transcript: VoiceTranscriptSegment,
    ) -> "TranscriptExtendedSegment":
        return cls(
            start=voice_separator_segment.start,
            end=voice_separator_segment.end,
            speaker=voice_separator_segment.speaker,
            filename=voice_separator_segment.filename,
            id=voice_separator_segment.id,
            transcript=transcript,
        )

