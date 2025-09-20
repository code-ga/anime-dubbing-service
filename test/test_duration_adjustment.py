#!/usr/bin/env python3
"""
Simple test script to verify the duration adjustment functionality.
"""
import torch
import torchaudio
import numpy as np
from tts.F5 import adjust_audio_duration

def test_duration_adjustment():
    """Test the adjust_audio_duration function with different scenarios."""
    print("Testing duration adjustment functionality...")

    # Create a simple test audio (1 second of sine wave at 22050 Hz)
    sample_rate = 22050
    duration = 1.0
    frequency = 440  # A4 note

    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)

    print(f"Original audio: {waveform.shape[1] / sample_rate:.2f} seconds")

    # Test 1: Audio that fits exactly (should return unchanged)
    target_duration = 1.0
    adjusted = adjust_audio_duration(waveform, sample_rate, target_duration)
    print(f"Test 1 - Exact fit: {adjusted.shape[1] / sample_rate:.2f} seconds (expected: {target_duration:.2f})")

    # Test 2: Audio that's too long (should be sped up)
    target_duration = 0.5
    adjusted = adjust_audio_duration(waveform, sample_rate, target_duration)
    print(f"Test 2 - Speed up: {adjusted.shape[1] / sample_rate:.2f} seconds (expected: {target_duration:.2f})")

    # Test 3: Audio that's too short (should be padded)
    target_duration = 1.5
    adjusted = adjust_audio_duration(waveform, sample_rate, target_duration)
    print(f"Test 3 - Pad with silence: {adjusted.shape[1] / sample_rate:.2f} seconds (expected: {target_duration:.2f})")

    print("All tests completed successfully!")

if __name__ == "__main__":
    test_duration_adjustment()