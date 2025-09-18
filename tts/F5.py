import torch
import torchaudio
from f5_tts.api import F5TTS
import numpy as np


def generate_tts_custom(
    text: str,
    output_path: str,
    ref_audio_path: str,
    ref_text: str,
    checkpoint_path: str = "checkpoints/f5_tts_multilingual.pth",
):
    """
    Generate TTS using a custom fine-tuned F5-TTS model.

    :param checkpoint_path: Path to the custom model checkpoint (e.g., ckpt.pth).
    :param text: Input text to synthesize.
    :param output_path: Path to save the generated WAV audio file.
    :param ref_audio_path: Path to reference audio for voice cloning.
    :param ref_text: Reference text corresponding to the reference audio.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    if checkpoint_path == "checkpoints/f5_tts_multilingual.pth":
        model = F5TTS(device=device, model="F5TTS_Base")
    else:
        model = F5TTS(ckpt_file=checkpoint_path, device=device)

    # Perform inference
    audio, sr, spec = model.infer(ref_audio_path, ref_text=ref_text, gen_text=text)

    # Save output
    if audio is None:
        raise ValueError("TTS inference returned None audio; check model or inputs.")
    audio_tensor = torch.from_numpy(audio)
    audio_tensor = audio_tensor.unsqueeze(0)
    torchaudio.save(output_path, audio_tensor, sr)
