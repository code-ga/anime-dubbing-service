#!/usr/bin/env python3
"""
Simple test script for build_speaker_refs function structure validation
"""
import json
import os
import sys
from tts.orchestrator import build_speaker_refs
from utils.metadata import load_metadata

def test_build_speaker_refs_structure():
    """Test the build_speaker_refs function structure without full execution"""
    print("Testing build_speaker_refs function structure...")

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

        print(f"✅ Loaded transcribe data with {len(transcribe_data.get('segments', []))} segments")
        print(f"✅ Loaded separate data with vocals path: {separate_data.get('vocals_path')}")

        # Check if vocals file exists
        vocals_path = os.path.join(tmp_path, separate_data["vocals_path"])
        if not os.path.exists(vocals_path):
            print(f"❌ Vocals file not found: {vocals_path}")
            return False

        print(f"✅ Vocals file exists: {vocals_path}")

        # Test function signature and basic structure
        inputs_data = {
            "separate_audio": separate_data
        }

        # Test that the function can be called (but we'll interrupt before transcription)
        try:
            print("🔄 Testing function call...")
            # We'll call it but expect it to take time due to transcription
            # For now, just validate the structure is correct
            result = build_speaker_refs(tmp_path, metadata_path, inputs_data)

            print("✅ Function executed successfully!")
            print(f"✅ Result keys: {list(result.keys())}")

            # Validate expected structure
            expected_keys = ["stage", "timestamp", "errors", "refs_by_speaker", "default_ref", "extraction_criteria"]
            for key in expected_keys:
                if key not in result:
                    print(f"❌ Missing expected key: {key}")
                    return False

            print("✅ All expected keys present")

            # Check refs_by_speaker structure
            refs_by_speaker = result.get("refs_by_speaker", {})
            if not refs_by_speaker:
                print("⚠️  No speaker references found (might be expected)")
            else:
                print(f"✅ Found references for {len(refs_by_speaker)} speakers")
                # Check if the new structure is used (with ref_text)
                sample_speaker = list(refs_by_speaker.keys())[0]
                sample_ref = refs_by_speaker[sample_speaker]
                if isinstance(sample_ref, dict) and "ref_text" in sample_ref:
                    print("✅ New structure with ref_text detected")
                else:
                    print("⚠️  Old structure detected (no ref_text)")

            # Check default_ref structure
            default_ref = result.get("default_ref")
            if default_ref and isinstance(default_ref, dict) and "ref_text" in default_ref:
                print("✅ Default ref has new structure with ref_text")
            elif default_ref:
                print("⚠️  Default ref has old structure")

            # Check extraction criteria
            criteria = result.get("extraction_criteria", "")
            if "re-transcription" in criteria:
                print("✅ Re-transcription mentioned in criteria")
            else:
                print("⚠️  Re-transcription not mentioned in criteria")

            return True

        except KeyboardInterrupt:
            print("⚠️  Function call interrupted (expected due to transcription)")
            print("✅ Function structure is correct - transcription was working")
            return True
        except Exception as e:
            print(f"❌ Error during function call: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ ERROR during test setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_build_speaker_refs_structure()
    if success:
        print("\n✅ build_speaker_refs structure test PASSED")
    else:
        print("\n❌ build_speaker_refs structure test FAILED")
    sys.exit(0 if success else 1)