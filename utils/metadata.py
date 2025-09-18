import json
import os
from datetime import datetime

STAGES_ORDER = [
    "convert_mp4_to_wav", "separate_audio", "transcribe", "translate",
    "build_refs", "generate_tts", "mix_audio", "mux_video"
]

def load_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        metadata = json.load(f)
    # Validate input_file match would be done in main.py with args
    return metadata

def create_metadata(path, input_file, output_file, tmp_dir, target_lang):
    metadata = {
        "input_file": input_file,
        "output_file": output_file,
        "tmp_dir": tmp_dir,
        "target_lang": target_lang,
        "current_stage": "convert_mp4_to_wav",
        "completed_stages": [],
        "timestamps": {
            "workflow_start": datetime.utcnow().isoformat() + "Z"
        },
        "errors": {
            "overall": None,
            "per_stage": {}
        }
    }
    # Atomic write
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    os.replace(temp_path, path)
    return metadata

def update_success(path, stage, start_time, outputs):
    metadata = load_metadata(path)
    if metadata is None:
        raise ValueError("Metadata file not found")
    
    current_stage = metadata["current_stage"]
    if current_stage != stage:
        raise ValueError(f"Expected current_stage '{stage}', but got '{current_stage}'")
    
    end_time = datetime.utcnow().isoformat() + "Z"
    entry = {
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "output_files": outputs or {},
        "error": None
    }
    metadata["completed_stages"].append(entry)
    
    # Advance current_stage
    idx = STAGES_ORDER.index(stage)
    if idx + 1 < len(STAGES_ORDER):
        metadata["current_stage"] = STAGES_ORDER[idx + 1]
    else:
        metadata["current_stage"] = "complete"
        metadata["timestamps"]["workflow_end"] = end_time
    
    # Atomic write
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    os.replace(temp_path, path)
    return metadata

def update_failure(path, stage, start_time, error_msg, outputs=None):
    metadata = load_metadata(path)
    if metadata is None:
        raise ValueError("Metadata file not found")
    
    current_stage = metadata["current_stage"]
    if current_stage != stage:
        raise ValueError(f"Expected current_stage '{stage}', but got '{current_stage}'")
    
    end_time = datetime.utcnow().isoformat() + "Z"
    entry = {
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "output_files": outputs or {},
        "error": error_msg
    }
    if metadata["completed_stages"] and metadata["completed_stages"][-1]["stage"] == stage:
        # Update last entry if already started
        metadata["completed_stages"][-1] = entry
    else:
        metadata["completed_stages"].append(entry)
    
    metadata["errors"]["per_stage"][stage] = error_msg
    # Don't advance current_stage on failure
    
    # Atomic write
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    os.replace(temp_path, path)
    return metadata

def set_overall_error(path, error_msg):
    metadata = load_metadata(path)
    if metadata is None:
        raise ValueError("Metadata file not found")
    metadata["errors"]["overall"] = error_msg
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    os.replace(temp_path, path)

def is_complete(metadata):
    return metadata.get("current_stage") == "complete" and not metadata["errors"]["overall"]