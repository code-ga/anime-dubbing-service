import os
from audio_separator.separator import Separator


def separate(audio_path: str, output_dir: str) -> tuple[str, str]:
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

    return os.path.join(output_dir, instrumental_files), os.path.join(
        output_dir, vocals_files
    )
