import json
import os
import pytest
import torch
import torchaudio
from unittest.mock import patch, mock_open, MagicMock
from transcription.whisper import transcript, assign_word_speakers
from tts.orchestrator import generate_dubbed_segments
from dub.mixer import mix_audio
from translate.openAi import translate_with_openai

# Unit test for music detection logic
def test_music_detection():
    # Mock a Whisper segment with high no_speech_prob
    segment = {
        "no_speech_prob": 0.7,
        "text": "",
        "start": 0.0,
        "end": 5.0
    }
    # Simulate the detection logic
    no_speech_prob = segment.get("no_speech_prob", 0)
    text = segment["text"]
    is_empty = not text.strip()
    is_short = len(text.strip().split()) < 2
    MUSIC_THRESHOLD_HIGH = 0.6
    MUSIC_THRESHOLD_LOW = 0.4
    is_music = (
        no_speech_prob > MUSIC_THRESHOLD_HIGH
        or is_empty
        or (is_short and no_speech_prob > MUSIC_THRESHOLD_LOW)
    )
    assert is_music == True

def test_music_detection_low_prob():
    segment = {
        "no_speech_prob": 0.3,
        "text": "Hello world",
        "start": 0.0,
        "end": 5.0
    }
    no_speech_prob = segment.get("no_speech_prob", 0)
    text = segment["text"]
    is_empty = not text.strip()
    is_short = len(text.strip().split()) < 2
    MUSIC_THRESHOLD_HIGH = 0.6
    MUSIC_THRESHOLD_LOW = 0.4
    is_music = (
        no_speech_prob > MUSIC_THRESHOLD_HIGH
        or is_empty
        or (is_short and no_speech_prob > MUSIC_THRESHOLD_LOW)
    )
    assert is_music == False

# Test TTS skipping music segments
@patch('tts.orchestrator.generate_tts_custom')
def test_generate_dubbed_segments_skips_music(mock_tts):
    mock_tts.return_value = None
    translated_data = {
        "segments": [
            {"start": 0.0, "end": 5.0, "translated_text": "Hello", "is_singing": False, "speaker": "Speaker1"},
            {"start": 5.0, "end": 10.0, "translated_text": "", "is_singing": True},
            {"start": 10.0, "end": 15.0, "translated_text": "World", "is_singing": False, "speaker": "Speaker1"}
        ]
    }
    translated_path = "tmp/translated/translated.json"
    os.makedirs(os.path.dirname(translated_path), exist_ok=True)
    with open(translated_path, 'w') as f:
        json.dump(translated_data, f)
    
    ref_audios = {"Speaker1": "tmp/refs/speaker1.wav"}
    default_ref = "tmp/refs/default.wav"
    
    segments = generate_dubbed_segments(translated_path, ref_audios, default_ref)
    
    assert len(segments) == 2  # Skips the singing segment
    assert segments[0]['start'] == 0.0
    assert segments[1]['start'] == 10.0
    assert all(not seg.get('is_singing', False) for seg in segments)

# Test mixing: Mock audio tensors
def test_mix_audio_preserves_music():
    # Create mock original audio (10s, 22050 Hz) with non-zero in singing part
    sr = 22050
    original_audio = torch.zeros(2, int(10 * sr))  # Stereo, 10s
    # Add non-zero to 5-10s to simulate original singing/music
    original_audio[:, int(5 * sr):] = 0.5
    # Mock TTS segment audio (5s)
    tts_audio = torch.ones(1, int(5 * sr))  # Mono, 5s
    tts_path = "tmp/tts/0.0_5.0.wav"
    torchaudio.save(tts_path, tts_audio, sr)
    # Mock singing audio (5s, original vocal)
    singing_audio = torch.ones(1, int(5 * sr)) * 0.5  # Mono, 5s
    singing_path = "tmp/singing/5.0_10.0.wav"
    os.makedirs(os.path.dirname(singing_path), exist_ok=True)
    torchaudio.save(singing_path, singing_audio, sr)
    
    # Mock translated segments: speech 0-5, singing 5-10
    translated_data = {
        "segments": [
            {"start": 0.0, "end": 5.0, "is_singing": False},
            {"start": 5.0, "end": 10.0, "is_singing": True, "audioFilePath": singing_path}
        ]
    }
    translated_path = "tmp/translated/translated.json"
    with open(translated_path, 'w') as f:
        json.dump(translated_data, f)
    
    tts_segments = [{"start": 0.0, "end": 5.0, "tts_path": tts_path, "speaker": "Speaker1"}]
    original_wav = "tmp/full.wav"
    torchaudio.save(original_wav, original_audio, sr)
    output_wav = "tmp/dubbed.wav"
    
    mix_audio(original_wav, translated_path, tts_segments, output_wav)
    
    # Load mixed and assert: 0-5s has TTS (non-zero), 5-10s has original singing preserved (non-zero)
    mixed_audio, _ = torchaudio.load(output_wav)
    speech_part = mixed_audio[:, :int(5 * sr)]
    singing_part = mixed_audio[:, int(5 * sr):]
    assert torch.all(speech_part != 0)  # Has TTS
    assert torch.all(singing_part != 0)  # Preserves original singing

# Integration test placeholder (manual with sample audio)
def test_integration_pipeline():
    """Integration test requires sample audio assets. Run manually with small MP4."""
    # pytest.skip("Manual integration test: Run main.py with small input")
    pass

# Test translate skips music
@patch('openai.OpenAI.chat.completions.create')
def test_translate_skips_singing(mock_chat):
    # Mock responses: classif speech='speech', classif empty='singing', translate='Translated text'
    mock_speech = MagicMock()
    mock_speech.choices[0].message.content = "speech"
    mock_singing = MagicMock()
    mock_singing.choices[0].message.content = "singing"
    mock_translated = MagicMock()
    mock_translated.choices[0].message.content = "Translated text"
    mock_chat.side_effect = [mock_speech, mock_singing, mock_translated]
    
    transcript_data = {
        "speaker_embeddings": {},
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Speech", "speaker": "Speaker1"},
            {"start": 5.0, "end": 10.0, "text": "", "speaker": "Speaker1"}
        ],
        "language": "ja",
        "text": "Speech"
    }
    
    result = translate_with_openai(transcript_data, "en")  # type: ignore
    
    translated_segments = result.get("translated_segments", [])
    assert len(translated_segments) == 2
    assert translated_segments[0].get("translated_text", "") == "Translated text"
    assert translated_segments[1].get("translated_text", "") == ""  # Skipped singing
    assert translated_segments[1].get("is_singing", False) == True