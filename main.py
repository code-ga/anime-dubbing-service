#!/usr/bin/env python3
import json
import os
from dotenv import load_dotenv

load_dotenv("./.env")

from convert.mp4_wav import convert_mp4_to_wav
from convert.separate_audio import separate_audio
from transcription.whisper import transcript

# from transcription.emotion import emotional_transcript
import numpy as np
import soundfile as sf


tmp_path = os.path.abspath("./tmp")

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)


if __name__ == "__main__":
    # input_path = "./anime.mp4"
    # raw_audio_path = os.path.join(tmp_path, "raw_audio.wav")
    # convert_mp4_to_wav(input_path, raw_audio_path)

    # print(f"Successfully converted {input_path} to {raw_audio_path}")
    # # Perform the separation on specific audio files without reloading the model
    # instrumental_output_files, vocals_output_files = separate_audio(
    #     raw_audio_path, tmp_path
    # )
    # print(
    #     f"Separation complete! Output file(s): {instrumental_output_files}, {vocals_output_files}"
    # )
    audio_transcript = transcript(
        "./tmp/vocals_output.wav", os.path.join(tmp_path, "transcript")
    )
    # print(audio_transcript)
    for seg in audio_transcript["segments"]:
        if "speaker" not in seg:
            print(seg)
    with open(os.path.join(tmp_path, "transcript", "transcript.json"), "w") as outfile:
        json.dump(audio_transcript, outfile)
