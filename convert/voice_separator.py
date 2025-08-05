import os
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
import json

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN") or None,
    cache_dir=os.path.join(os.path.abspath("./models"), "voice_separator"),
).to(DEVICE)


def voice_separator(vocal_path: str, output_path: str):
    original_audio = AudioSegment.from_wav(vocal_path)

    # apply pretrained pipeline
    diarization = pipeline(vocal_path)
    speakers_time = {}
    audio_files: list[str] = []
    all_segments = []
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(turn, speaker)
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        start_time_ms = int(turn.start * 1000)
        end_time_ms = int(turn.end * 1000)

        # Extract the audio segment for the current turn
        segment = original_audio[start_time_ms:end_time_ms]
        if not os.path.exists(os.path.join(output_path, speaker)):
            os.makedirs(os.path.join(output_path, speaker))
        segment.export(
            os.path.join(output_path, speaker, f"{turn.start:.1f}-{turn.end:.1f}.wav"),
            format="wav",
        )
        speakers_time[speaker] = (
            speakers_time.get(speaker, 0) + end_time_ms - start_time_ms
        )
        audio_files.append(
            os.path.join(output_path, speaker, f"{turn.start:.1f}-{turn.end:.1f}.wav")
        )
        all_segments.append(
            {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "filename": os.path.join(
                    output_path, speaker, f"{turn.start:.1f}-{turn.end:.1f}.wav"
                ),
                "id": len(all_segments),
            }
        )
    result = {
        "audio_files": audio_files,
        "speakers_time": speakers_time,
        "all_segments": all_segments,
    }
    output_file = os.path.join(output_path, "output.json")
    file = open(output_file, mode="w")
    file.write(str(json.dumps(result)))
    return output_file
    # start=0.2s stop=1.5s speaker_0
    # start=1.8s stop=3.9s speaker_1
    # start=4.2s stop=5.7s speaker_0
    # ...
