import whisper
import json
import torch
from ..types.convert import (
    VoiceSeparator,
    TranscriptResult,
    TranscriptExtendedSegment,
    VoiceTranscript,
    VoiceTranscriptSegment,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("turbo", download_root="./models/whisper", device=DEVICE)


def transcript(voice_separator_json_output, whisper_json_output):
    raw_json_input = json.loads(open(voice_separator_json_output).read())
    json_input = VoiceSeparator.model_validate(raw_json_input)
    json_output = TranscriptResult(
        audio_files=json_input.audio_files,
        speakers_time=json_input.speakers_time,
        all_segments=[],
    )
    for segment in json_input.all_segments:
        # load audio and pad/trim it to fit 30 seconds
        result = model.transcribe(
            segment.filename,
            language="japanese",
            # verbose=True,
        )
        print(
            f"{segment.id}/{len(json_input.all_segments)} {segment.filename}: {result['language']} {result['text']}"
        )
        result_segment: list[VoiceTranscript] = []
        for i in result["segments"]:
            # result_segment.append(
            #     {
            #         "start": i["start"],  # type: ignore
            #         "end": i["end"],  # type: ignore
            #         "text": i["text"],  # type: ignore
            #         "id": i["id"],  # type: ignore
            #         "seek": i["seek"],  # type: ignore
            #     }
            # )
            result_segment.append(
                VoiceTranscript(
                    start=i["start"],  # type: ignore
                    end=i["end"],  # type: ignore
                    text=i["text"],  # type: ignore
                    id=i["id"],  # type: ignore
                    seek=i["seek"],  # type: ignore
                )
            )

        # json_input["all_segments"][segment.id]["transcript"] = {
        #     "text": result["text"],
        #     "language": result["language"],
        #     "segments": result_segment,
        # }
        json_output.all_segments.append(
            TranscriptExtendedSegment.from_voice_separator_segment(
                segment,
                transcript=VoiceTranscriptSegment(
                    text=result["text"], # type: ignore
                    language=result["language"], # type: ignore
                    segments=result_segment,
                ),
            )
        )
        # json_output.all_segments[-1].transcript = {
        #     "text": result["text"],
        #     "language": result["language"],
        #     "segments": [vt.model_dump() for vt in result_segment],
        # }

    with open(whisper_json_output, "w") as f:
        json.dump(json_output.model_dump(), f, indent=2)
    return whisper_json_output
