#!/usr/bin/env python3
"""
Test script for build_speaker_refs function with re-transcription
"""
import json
import os
import sys
from tts.orchestrator import build_speaker_refs
from utils.metadata import load_metadata

def test_build_speaker_refs():
    """Test the build_speaker_refs function in isolation"""
    print("Testing build_speaker_refs function...")

    # Use existing tmp directory and metadata
    tmp_path = "./tmp"
    metadata_path = os.path.join(tmp_path, "metadata.json")

    # Load metadata
    metadata = load_metadata(metadata_path)
    if not metadata:
        print("ERROR: Could not load metadata")
        return False

    # Load previous results
    try:
        from utils.metadata import load_previous_result
        transcribe_data = load_previous_result(metadata_path, "transcribe")
        separate_data = load_previous_result(metadata_path, "separate_audio")

        if not transcribe_data or not separate_data:
            print("ERROR: Could not load previous results")
            return False

        print(f"Loaded transcribe data with {len(transcribe_data.get('segments', []))} segments")
        print(f"Loaded separate data with vocals path: {separate_data.get('vocals_path')}")

        # Test the function
        inputs_data = {
            "separate_audio": separate_data
        }

        result = build_speaker_refs(tmp_path, metadata_path, inputs_data)

        print("Function executed successfully!")
        print(f"Result keys: {list(result.keys())}")
        print(f"Number of speakers: {len(result.get('refs_by_speaker', {}))}")
        print(f"Default ref: {result.get('default_ref')}")
        print(f"Extraction criteria: {result.get('extraction_criteria')}")

        # Validate the structure
        if "refs_by_speaker" not in result:
            print("ERROR: refs_by_speaker not found in result")
            return False

        if "default_ref" not in result:
            print("ERROR: default_ref not found in result")
            return False

        if "ref_text" not in result:
            print("ERROR: ref_text not found in result")
            return False

        # Check if reference files were created
        refs_dir = os.path.join(tmp_path, "refs")
        if os.path.exists(refs_dir):
            ref_files = os.listdir(refs_dir)
            print(f"Reference files created: {ref_files}")
        else:
            print("WARNING: refs directory not found")

        return True

    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_build_speaker_refs()
    if success:
        print("\n✅ build_speaker_refs test PASSED")
    else:
        print("\n❌ build_speaker_refs test FAILED")
    sys.exit(0 if success else 1)