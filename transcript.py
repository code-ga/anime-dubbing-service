import whisper 
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("turbo").to(DEVICE)
result = model.transcribe("./vocals_output.wav.wav")

for segment in result["segments"]: 
  print(segment)
