import ffmpeg
import os

def convert_mp4_to_wav(input_file, output_file):
    """
    Convert an MP4 file to WAV format using ffmpeg-python.
    
    Args:
        input_file (str): Path to the input MP4 file.
        output_file (str): Path to the output WAV file.
    
    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    try:
        # Set up the input stream
        stream = ffmpeg.input(input_file)
        
        # Configure the output stream with no video and specific audio settings
        stream = ffmpeg.output(
            stream,
            output_file,
            vn=True,  # No video
            acodec='pcm_s16le',  # Audio codec to PCM signed 16-bit little endian
            ar=44100,  # Audio sampling rate to 44100 Hz
            ac=2,  # Audio channels to stereo
            format='wav'  # Explicitly specify the output format as WAV
        )
        
        # Run the conversion
        ffmpeg.run(stream)
        print(f"Successfully converted '{input_file}' to '{output_file}'.")
        return True
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"Error during conversion: {error_message}")
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

# if __name__ == "__main__":
#     input_path = "../anime.mp4"
#     output_path = "../output.wav"
#     convert_mp4_to_wav(input_path, output_path)
