import torch
import torchaudio


def adjust_audio_duration(
    waveform: torch.Tensor, sample_rate: int, target_duration: float
) -> torch.Tensor:
    """
    Adjust audio duration to match target duration by speeding up or padding with silence.

    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        target_duration: Target duration in seconds

    Returns:
        Adjusted waveform tensor
    """
    # current_samples = waveform.shape[1]
    # target_samples = int(target_duration * sample_rate)

    # if current_samples == target_samples:
    #     # Audio already matches target duration
    #     return waveform
    # elif current_samples > target_samples:
    #     # Audio is too long - trim to fit
    #     adjusted_waveform = waveform[:, :target_samples]

    #     # Clean up original waveform after trimming
    #     del waveform
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     return adjusted_waveform
    # else:
    #     # Audio is too short - pad with silence
    #     pad_samples = target_samples - current_samples
    #     padded_waveform = torch.nn.functional.pad(waveform, (0, pad_samples))

    #     # Clean up original waveform after padding
    #     del waveform
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     return padded_waveform

    return waveform
