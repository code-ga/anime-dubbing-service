import whisper
import json
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("turbo", download_root="./models/whisper", device=DEVICE)


def transcript(voice_separator_json_output, whisper_json_output):
    json_output = json.loads(open(voice_separator_json_output).read())
    for segment in json_output["all_segments"]:
        # load audio and pad/trim it to fit 30 seconds
        result = model.transcribe(
            segment["filename"],
            language="japanese",
            # verbose=True,
        )
        print(f"{segment['id']}/{len(json_output['all_segments'])} {segment['filename']}: {result['language']} {result['text']}")
        result_segment = []
        for i in result["segments"]:
            result_segment.append(
                {
                    "start": i["start"], # type: ignore
                    "end": i["end"], # type: ignore
                    "text": i["text"], # type: ignore
                    "id": i["id"], # type: ignore
                    "seek": i["seek"], # type: ignore
                }
            )
        json_output["all_segments"][segment["id"]]["transcript"] = {
            "text": result["text"],
            "language": result["language"],
            "segments": result_segment,
        }

    with open(whisper_json_output, "w") as f:
        json.dump(json_output, f, indent=2)
    return whisper_json_output
