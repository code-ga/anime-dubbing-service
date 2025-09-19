import ffmpeg
import os
import torchaudio
from datetime import datetime
from utils.metadata import save_stage_result


def convert_mp4_to_wav(
    tmp_path, metadata_path, inputs_data, input_file, **kwargs
) -> dict:
    """
    Convert an MP4 file to WAV format using ffmpeg-python.

    Args:
        tmp_path: Path to temporary directory.
        metadata_path: Path to metadata.
        inputs_data: Dict of previous stage data (empty for first stage).
        input_file: Path to input MP4.
        **kwargs: Additional arguments.

    Returns:
        Stage data with full_wav_path, duration, sample_rate, channels.
    """
    output_file = os.path.join(tmp_path, "full.wav")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    try:
        # Set up the input stream
        stream = ffmpeg.input(input_file)

        # Configure the output stream with no video and specific audio settings
        stream = ffmpeg.output(
            stream,
            output_file,
            vn=True,  # No video
            acodec="pcm_s16le",  # Audio codec to PCM signed 16-bit little endian
            ar=22050,  # Audio sampling rate to 22050 Hz
            ac=1,  # Audio channels to mono
            format="wav",  # Explicitly specify the output format as WAV
        )

        # Run the conversion
        ffmpeg.run(stream)
        print(f"Successfully converted '{input_file}' to '{output_file}'.")

        # Get metadata
        waveform, sr = torchaudio.load(output_file)
        duration = waveform.shape[1] / sr
        channels = waveform.shape[0]

        stage_data = {
            "stage": "convert_mp4_to_wav",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "errors": [],
            "full_wav_path": "full.wav",
            "duration": duration,
            "sample_rate": sr,
            "channels": channels,
            "output_file": output_file,
        }

        return stage_data
    except ffmpeg.Error as e:
        error_message = e.stderr.decode("utf-8") if e.stderr else str(e)
        raise RuntimeError(f"Error during conversion: {error_message}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


# if __name__ == "__main__":
#     input_path = "../anime.mp4"
#     output_path = "../output.wav"
#     convert_mp4_to_wav(input_path, output_path)
