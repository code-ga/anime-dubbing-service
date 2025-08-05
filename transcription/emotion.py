import json
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    cache_dir="./models/emotional_transcript",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    cache_dir="./models/emotional_transcript",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
)
id2label = model.config.id2label


def emotional_transcript(
    whisper_transcript_output_file, emotion_transcript_output_file
):
    json_output = json.loads(open(whisper_transcript_output_file).read())
    for segment in json_output["all_segments"]:
        audio_path = segment["filename"]
        for i in segment["transcript"]["segments"]:
            start_time = i["start"]
            end_time = i["end"]
            duration = end_time - start_time
            predicted_emotion = predict_emotion(
                audio_path, model, feature_extractor, id2label, start_time, duration
            )
            print(f"Predicted Emotion: {predicted_emotion}")
            i["emotion"] = predicted_emotion
    with open(emotion_transcript_output_file, "w") as f:
        json.dump(json_output, f, indent=2)
    return emotion_transcript_output_file


def preprocess_audio(
    audio_path, feature_extractor, start_time=0.0, duration=None, max_duration=30.0
):
    audio_array, sampling_rate = librosa.load(
        audio_path, sr=feature_extractor.sampling_rate
    )

    start_sample = int(start_time * sampling_rate)
    if duration is not None:
        end_sample = int((start_time + duration) * sampling_rate)
        if end_sample > len(audio_array):
            end_sample = len(audio_array)
        audio_array = audio_array[start_sample:end_sample]
        print(
            f"Extracted segment: {start_time}s to {start_time + duration}s, samples {start_sample} to {end_sample}"
        )
    else:
        audio_array = audio_array[start_sample:]
        print(
            f"Extracted segment from {start_time}s onwards, starting at sample {start_sample}"
        )

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    elif len(audio_array) < max_length:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def predict_emotion(
    audio_path,
    model,
    feature_extractor,
    id2label,
    start_time=0.0,
    duration=None,
    max_duration=30.0,
):
    print(f"Processing audio segment: start_time={start_time}, duration={duration}")
    inputs = preprocess_audio(
        audio_path, feature_extractor, start_time, duration, max_duration
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(
            f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}"
        )
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    print(f"Predicted Emotion for segment starting at {start_time}s: {predicted_label}")

    return predicted_label
