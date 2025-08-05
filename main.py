#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv("./.env")

# from convert.mp4_wav import convert_mp4_to_wav
# from convert.separate_audio import separate_audio
# from convert.voice_separator import voice_separator
# from transcription.whisper import transcript
from transcription.emotion import emotional_transcript
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


tmp_path = os.path.abspath("./tmp")

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

models_path = "./models"

if not os.path.exists(models_path):
    os.makedirs(models_path)


# if __name__ == "__main__":
#     input_path = "./anime.mp4"
#     raw_audio_path = os.path.join(tmp_path, "raw_audio.wav")
#     convert_mp4_to_wav(input_path, raw_audio_path)

#     print(f"Successfully converted {input_path} to {raw_audio_path}")
#     # Perform the separation on specific audio files without reloading the model
#     instrumental_output_files, vocals_output_files = separate_audio(
#         raw_audio_path, tmp_path
#     )
#     print(
#         f"Separation complete! Output file(s): {instrumental_output_files}, {vocals_output_files}"
#     )

#     voice_output = voice_separator(
#         vocals_output_files, os.path.join(tmp_path, "characters")
#     )

#     print(f"Voice separation complete! Output file: {voice_output}")

#     transcript_output = transcript(
#         voice_output,
#         os.path.join(tmp_path, "transcript.json"),
#     )

#     print(f"Transcription complete! Output file: {transcript_output}")

#     emotional_transcript_output = emotional_transcript(
#         transcript_output,
#         os.path.join(tmp_path, "emotional_transcript.json"),
#     )

#     print(f"Emotional transcription complete! Output file: {emotional_transcript_output}")

# if __name__ == "__main__":
#     voice_separator("./tmp/vocals_output.wav.wav", os.path.join(tmp_path, "characters"))

# if __name__ == "__main__":
#     transcript(
#         os.path.join(tmp_path, "characters", "output.json"),
#         os.path.join(tmp_path, "transcript.json"),
#     )

if __name__ == "__main__":
    emotional_transcript(os.path.join(tmp_path, "transcript.json"), os.path.join(tmp_path, "emotional_transcript.json"))
