import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.utils import load_checkpoint

def generate_tts_custom(checkpoint_path: str = "checkpoints/f5_tts_multilingual.pth", text: str, output_path: str, ref_audio_path: str, ref_text: str):
    """
    Generate TTS using a custom fine-tuned F5-TTS model.
    
    :param checkpoint_path: Path to the custom model checkpoint (e.g., ckpt.pth).
    :param text: Input text to synthesize.
    :param output_path: Path to save the generated WAV audio file.
    :param ref_audio_path: Path to reference audio for voice cloning.
    :param ref_text: Reference text corresponding to the reference audio.
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    if checkpoint_path == "checkpoints/f5_tts_multilingual.pth":
        model = F5TTS(device=device)
    else:
        model = F5TTS.from_pretrained(checkpoint_path, device=device)
    
    # Load reference audio
    ref_audio, sr = torchaudio.load(ref_audio_path)
    if sr != 22050:  # F5-TTS typically uses 22kHz
        resampler = torchaudio.transforms.Resample(sr, 22050)
        ref_audio = resampler(ref_audio)
    
    # Perform inference
    audio = model.infer(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
    
    # Save output
    torchaudio.save(output_path, audio.unsqueeze(0), 22050)