from dotenv import load_dotenv
load_dotenv("./.env")
from pyannote.audio import Pipeline
import torch
import torchaudio.transforms as T
import gc

import os
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import scipy.io.wavfile
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AUDIO_PATH = "./test.wav"
sampling_rate = torchaudio.info(AUDIO_PATH).sample_rate
print("sample_rate:", sampling_rate)

HF_TOKEN = os.environ.get("HF_TOKEN")


def VAD(audio_path, sampling_rate, tmpDir=os.path.join("./tmp", "vad")):
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=HF_TOKEN,
    ).to(DEVICE)
    with torch.inference_mode():
        output = pipeline(audio_path)

    result = []
    for speech in output.get_timeline().support():
        # extract speech
        start_sample = int(speech.start * sampling_rate)
        end_sample = int(speech.end * sampling_rate)
        num_frames = max(0, end_sample - start_sample)
        if num_frames <= 0:
            continue
        segment, _sr = torchaudio.load(
            audio_path, frame_offset=start_sample, num_frames=num_frames
        )
        savePath = os.path.join(tmpDir, f"{speech.start:.2f}_{speech.end:.2f}.wav")
        # Save as 16-bit PCM for broad player compatibility
        torchaudio.save(
            savePath, segment, sampling_rate, encoding="PCM_S", bits_per_sample=16
        )
        result.append({"start": speech.start, "end": speech.end, "audio": savePath})
    del pipeline
    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


chunks = VAD(AUDIO_PATH, sampling_rate)
print("DONE VAD", len(chunks))
gc.collect()

def speech_separation(
    audioPath: str,
    start: float,
    end: float,
    tmpDir=os.path.join("./tmp", "speech_separation"),
    ami_pipeline=None,
    diarization_pipeline=None,
):
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)

    try:
        result = []
        # audio, sampling_rate = torchaudio.load(audioPath)
        # new_sample_rate = 16000
        # audio_to_process = audio
        # if sampling_rate != new_sample_rate:
        #     audio_to_process = torchaudio.transforms.Resample(
        #         sampling_rate, new_sample_rate
        #     )(audio)
        with torch.inference_mode():
            if ami_pipeline is not None:
                diarization, sources = ami_pipeline(audioPath)
            else:
                print("ami_pipeline is None, cannot perform speech separation.")
                return result
        print(diarization)

        for s, speaker in enumerate(diarization.labels()):
            startTime = start
            endTime = end
            print(endTime - startTime, sources.data[:, s].shape[0])
            saveDir = os.path.join(
                tmpDir,
                f"{startTime:.1f}_{endTime:.1f}",
            )
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            saving_path = os.path.join(
                saveDir,
                f"{speaker}.wav",
            )
            # Ensure 16-bit PCM output for compatibility
            wave = sources.data[:, s]
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            if np.issubdtype(wave.dtype, np.floating):
                wave16 = (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                wave16 = wave.astype(np.int16)
            scipy.io.wavfile.write(saving_path, 16000, wave16)
            # torchaudio.save(saving_path, sources.data[:, s], 16000)
            # print(diarization, speaker, s)
            result.append(
                {
                    "start": startTime,
                    "end": endTime,
                    "speaker": speaker,
                    "audio": saving_path,
                    "duration": endTime - startTime,
                    "model": "pyannote/speech-separation-ami-1.0",
                }
            )
        del diarization
        del sources
        gc.collect()
        return result
    except Exception as e:
        result = []
        audio, sampling_rate = torchaudio.load(audioPath)
        print(f"{e} falling back using pyannote/speaker-diarization-3.1")

        # run the pipeline on an audio file
        if diarization_pipeline is not None:
            diarization = diarization_pipeline(
                {"waveform": audio, "sample_rate": sampling_rate}
            )
        else:
            print("diarization_pipeline is None, cannot perform diarization.")
            return result
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(turn, speaker)
            startTime = turn.start + start
            endTime = turn.end + start
            print(f"start={startTime:.1f}s stop={endTime:.1f}s speaker_{speaker}")
            saveDir = os.path.join(tmpDir, f"{startTime:.1f}_{endTime:.1f}")
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            saving_path = os.path.join(saveDir, f"speaker_{speaker}.wav")
            relative_start = int(turn.start * sampling_rate)
            relative_end = int(turn.end * sampling_rate)
            torchaudio.save(
                saving_path,
                audio[:, relative_start:relative_end],
                sampling_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            result.append(
                {
                    "start": startTime,
                    "end": endTime,
                    "speaker": speaker,
                    "audio": saving_path,
                    "duration": endTime - startTime,
                    "model": "pyannote/speaker-diarization-3.1",
                }
            )
        del diarization
        del audio
        gc.collect()
        return result


def vadToSpeechSeparation(chunks: list):
    speech_separation_result = []
    ami_pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=HF_TOKEN,
    ).to(DEVICE)
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    ).to(DEVICE)

    for c in chunks:
        print(c["audio"])
        result = speech_separation(
            audioPath=c["audio"],
            start=c["start"],
            end=c["end"],
            ami_pipeline=ami_pipeline,
            diarization_pipeline=diarization_pipeline,
        )
        speech_separation_result += result

    del ami_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return speech_separation_result


gc.collect()
speech_separation_result = vadToSpeechSeparation(chunks)
print("DONE speech_separation", speech_separation_result)


def ensure_min_duration_wav(path, min_seconds=1.0):
    """Ensure audio file has at least min_seconds duration by zero-padding if needed.
    Returns a path to a file that satisfies the minimum duration (may be the same path).
    """
    try:
        info = torchaudio.info(path)
        if info.sample_rate > 0 and info.num_frames / info.sample_rate >= min_seconds:
            return path
    except Exception as e:
        print(f"torchaudio.info failed on {path}: {e}")
    # Load and pad
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as e:
        print(f"Failed to load {path} for padding: {e}")
        return path
    target_len = int(min_seconds * sr)
    if waveform.size(1) >= target_len:
        return path
    pad_len = target_len - waveform.size(1)
    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    tmp_dir = "./tmp/embedding"
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.basename(path)
    out_path = os.path.join(
        tmp_dir, f"{os.path.splitext(base)[0]}_pad{int(min_seconds*1000)}ms.wav"
    )
    torchaudio.save(out_path, waveform, sr, encoding="PCM_S", bits_per_sample=16)
    return out_path


# This function will correct and link the speaker over chunks
def speakerEmbedding(
    inference: EncoderClassifier, speech_separation_result, speakersList: dict
):
    # audio_path = ensure_min_duration_wav(speech_separation_result["audio"], 1.0)
    audio_path = speech_separation_result["audio"]
    signal, fs = torchaudio.load(audio_path)

    # Ensure mono (model expects [batch, time])
    if signal.dim() == 2 and signal.size(0) > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Resample to 16k for speechbrain ecapa model
    try:
        if fs != 16000:
            signal = T.Resample(orig_freq=fs, new_freq=16000)(signal)
            fs = 16000
    except Exception as e:
        print(f"Resample failed for {audio_path}: {e}")

    try:
        emb = inference.encode_batch(signal)
    except Exception as e:
        print(f"Embedding failed for {audio_path}: {e}")
        return speech_separation_result

    # Convert to a single 1 x D embedding vector consistently
    if isinstance(emb, tuple):
        emb = emb[0]
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)

    # Reduce any extra dimensions by averaging until we get a 1D vector
    e = emb
    while e.ndim > 1:
        e = e.mean(dim=0)
    e = e.detach().cpu().float()
    emb2d = e.unsqueeze(0).numpy()  # shape (1, D)

    # Convert to torch tensor for cosine similarity
    emb_tensor = torch.from_numpy(emb2d)

    # loop through speakers and compare the embeddings
    areInList = False
    for key, value in speakersList.items():
        ref = torch.as_tensor(value, dtype=torch.float32)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        # If for any reason dims mismatch, skip this ref to avoid runtime error
        if ref.size(1) != emb_tensor.size(1):
            continue
        distance = torch.nn.functional.cosine_similarity(emb_tensor, ref, dim=-1, eps=1e-6).item()
        if distance > 0.5:
            speech_separation_result["speaker"] = key
            areInList = True
            break
    if not areInList:
        # if the speaker is not in the list we create a new one
        speakersList[len(speakersList)] = emb2d
        speech_separation_result["speaker"] = len(speakersList) - 1
    return speech_separation_result


def voiceSeparatorToEmbedding(speech_separation_result: list):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
    # inference = Inference(model, window="whole").to(DEVICE)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    if classifier is None:
        raise Exception("Failed to load classifier")

    speakersList = {}
    voice_embedding_result = []
    for c in speech_separation_result:
        print(c["audio"])
        voice_embedding_result.append(
            speakerEmbedding(
                inference=classifier,
                speech_separation_result=c.copy(),
                speakersList=speakersList,
            )
        )

    print("DONE voice_embedding", voice_embedding_result)

    return voice_embedding_result

voice_embedding_result = voiceSeparatorToEmbedding(speech_separation_result)

import json
with open("test.json", mode="w") as f:
    f.write(json.dumps(voice_embedding_result))
