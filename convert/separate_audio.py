import os
import subprocess
import shutil

def separate(audio_path: str, tmp_path: str) -> tuple[str, str]:
    os.makedirs(tmp_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_subdir = os.path.join(tmp_path, base_name)
    subprocess.run(['spleeter', 'separate', '-p', 'spleeter:2stems', '-o', output_subdir, audio_path], check=True)
    vocals_src = os.path.join(output_subdir, 'vocals.wav')
    instr_src = os.path.join(output_subdir, 'accompaniment.wav')
    vocals_path = os.path.join(tmp_path, 'vocals.wav')
    instr_path = os.path.join(tmp_path, 'accompaniment.wav')
    shutil.move(vocals_src, vocals_path)
    shutil.move(instr_src, instr_path)
    # Clean up subdir if empty
    try:
        shutil.rmtree(output_subdir)
    except OSError:
        pass
    return vocals_path, instr_path
