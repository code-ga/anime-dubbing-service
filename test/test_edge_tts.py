#!/usr/bin/env python3
"""
Test script for edge-tts implementation.
This script tests the basic functionality of the edge-tts TTS module.
"""

import asyncio
import os
import sys
import tempfile
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts.edge_tts import (
    generate_tts_for_speaker,
    get_voice_for_speaker,
    list_available_voices,
    test_voice_quality,
    generate_tts_audio_async
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_basic_tts():
    """Test basic TTS functionality."""
    print("Testing basic edge-tts functionality...")

    # Test text
    test_text = "Hello, this is a test of the edge-tts text-to-speech system."

    # Test voice
    test_voice = "en-US-GuyNeural"

    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        output_path = tmp_file.name

    try:
        # Generate TTS
        success = await generate_tts_audio_async(
            text=test_text,
            output_path=output_path,
            voice=test_voice
        )

        if success:
            print(f"✓ TTS generation successful: {output_path}")
            print(f"  File size: {os.path.getsize(output_path)} bytes")

            # Check if file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print("✓ Generated audio file is valid")
            else:
                print("✗ Generated audio file is empty or missing")
                return False
        else:
            print("✗ TTS generation failed")
            return False

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)

    return True

def test_voice_selection():
    """Test voice selection functionality."""
    print("\nTesting voice selection...")

    # Test different speakers
    test_speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "unknown_speaker"]

    for speaker in test_speakers:
        voice = get_voice_for_speaker(speaker, "en", "neutral")
        print(f"  {speaker} -> {voice}")

    print("✓ Voice selection test completed")
    return True

def test_available_voices():
    """Test listing available voices."""
    print("\nTesting available voices listing...")

    try:
        voices = list_available_voices("en")
        print(f"✓ Found {len(voices)} English voices")
        if voices:
            print(f"  First 5 voices: {voices[:5]}")
        return True
    except Exception as e:
        print(f"✗ Error listing voices: {e}")
        return False

async def test_voice_quality():
    """Test voice quality testing."""
    print("\nTesting voice quality...")

    test_voice = "en-US-GuyNeural"
    test_text = "This is a quality test."

    try:
        success = test_voice_quality(test_voice, test_text)
        if success:
            print(f"✓ Voice quality test passed for {test_voice}")
            return True
        else:
            print(f"✗ Voice quality test failed for {test_voice}")
            return False
    except Exception as e:
        print(f"✗ Error testing voice quality: {e}")
        return False

async def test_speaker_generation():
    """Test TTS generation for speaker segments."""
    print("\nTesting speaker-based TTS generation...")

    # Mock segments data
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "translated_text": "Hello, this is the first segment.",
            "speaker": "SPEAKER_00"
        },
        {
            "start": 2.0,
            "end": 4.0,
            "translated_text": "This is the second segment from a different speaker.",
            "speaker": "SPEAKER_01"
        }
    ]

    # Mock reference data
    ref_audios_by_speaker = {
        "SPEAKER_00": "refs/SPEAKER_00.wav",
        "SPEAKER_01": "refs/SPEAKER_01.wav"
    }
    default_ref = "refs/default.wav"

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Test TTS generation
            tts_segments = generate_tts_for_speaker(
                segments=segments,
                speaker="SPEAKER_00",
                ref_audios_by_speaker=ref_audios_by_speaker,
                default_ref=default_ref,
                tmp_path=tmp_dir,
                target_sr=24000,
                language="en"
            )

            print(f"✓ Generated {len(tts_segments)} TTS segments")
            for i, seg in enumerate(tts_segments):
                print(f"  Segment {i}: {seg['start']".1f"}s - {seg['end']".1f"}s, speaker: {seg['speaker']}")

            return len(tts_segments) > 0

        except Exception as e:
            print(f"✗ Error in speaker generation test: {e}")
            return False

async def main():
    """Run all tests."""
    print("Starting edge-tts implementation tests...\n")

    tests = [
        ("Basic TTS", test_basic_tts),
        ("Voice Selection", test_voice_selection),
        ("Available Voices", test_available_voices),
        ("Voice Quality", test_voice_quality),
        ("Speaker Generation", test_speaker_generation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                print(f"\n✗ {test_name} FAILED")

        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")

    print(f"\n{'='*50}")
    print("TEST SUMMARY"
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)