import os
from audio_separator.separator import Separator
from datetime import datetime
from utils.metadata import save_stage_result
import shutil
import gc
import torch


def separate(tmp_path, metadata_path, inputs_data, **kwargs) -> dict:
    full_wav_path = inputs_data["convert_mp4_to_wav"]["full_wav_path"]
    audio_path = os.path.join(tmp_path, full_wav_path)
    output_dir = tmp_path

    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator(
        model_file_dir=os.path.join(os.path.abspath("./models"), "audio_separator"),
        output_dir=output_dir,
    )

    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')

    separator.load_model()
    instrumental_files, vocals_files = separator.separate(
        audio_path,
        custom_output_names={
            "Vocals": "vocals_output",
            "Instrumental": "instrumental_output",
        },
    )

    vocals_path = os.path.join(output_dir, vocals_files)
    instrumental_path = os.path.join(output_dir, instrumental_files)

    # Copy to standard names
    final_vocals = os.path.join(tmp_path, "vocals.wav")
    final_instrumental = os.path.join(tmp_path, "accompaniment.wav")
    shutil.copy(vocals_path, final_vocals)
    shutil.copy(instrumental_path, final_instrumental)

    # Get total duration from previous stage
    total_duration = inputs_data["convert_mp4_to_wav"]["duration"]

    stage_data = {
        "stage": "separate_audio",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "errors": [],
        "vocals_path": "vocals.wav",
        "instrumental_path": "accompaniment.wav",
        "separation_method": "audio_separator",
        "metadata": {"total_duration": total_duration},
    }

    # Clean up model and free memory
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stage_data
