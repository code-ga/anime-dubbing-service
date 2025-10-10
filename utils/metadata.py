import json
import os
from datetime import datetime

STAGES_ORDER = [
    "convert_mp4_to_wav",
    "separate_audio",
    "transcribe",
    "translate",
    "build_refs",
    "generate_tts",
    "mix_audio",
    "mux_video",
]


def load_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
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
        "timestamps": {"workflow_start": datetime.utcnow().isoformat() + "Z"},
        "stage_results": {},
        "errors": {"overall": None, "per_stage": {}},
    }
    # Direct write for step-by-step execution
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def update_success(path, stage, start_time, outputs, stage_data=None):
    metadata = load_metadata(path)
    if metadata is None:
        raise ValueError("Metadata file not found")

    current_stage = metadata["current_stage"]
    if current_stage != stage:
        raise ValueError(f"Expected current_stage '{stage}', but got '{current_stage}'")
    if stage_data:
        try:
            save_stage_result(path, stage, stage_data)
            metadata = load_metadata(path)
            if metadata is None:
                raise ValueError("Metadata file not found after saving stage result")
        except Exception as save_e:
            print(f"Error saving stage result for {stage}: {save_e}")
            raise

    end_time = datetime.utcnow().isoformat() + "Z"
    output_files = stage_data.get("output_files", {}) if stage_data else {}
    entry = {
        "stage": stage,
        "start_time": start_time,
        "end_time": end_time,
        "output_files": output_files,
        "error": None,
    }
    metadata["completed_stages"].append(entry)

    # Advance current_stage
    idx = STAGES_ORDER.index(stage)
    if idx + 1 < len(STAGES_ORDER):
        metadata["current_stage"] = STAGES_ORDER[idx + 1]
    else:
        metadata["current_stage"] = "complete"
        metadata["timestamps"]["workflow_end"] = end_time
    print(
        f"✅ Stage '{stage}' completed successfully. Next stage: '{metadata}' write to file {path}'"
    )
    # Direct write for step-by-step execution
    with open(path, "w") as f:
        # json.dump(metadata, f, indent=4)
        json_string = json.dumps(metadata, indent=4)
        print(f"Writing updated metadata to {path}:\n{json_string}")
        f.write(json_string)
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
        "error": error_msg,
    }
    if (
        metadata["completed_stages"]
        and metadata["completed_stages"][-1]["stage"] == stage
    ):
        # Update last entry if already started
        metadata["completed_stages"][-1] = entry
    else:
        metadata["completed_stages"].append(entry)

    metadata["errors"]["per_stage"][stage] = error_msg
    # Don't advance current_stage on failure

    # Direct write for step-by-step execution
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def set_overall_error(path, error_msg):
    metadata = load_metadata(path)
    if metadata is None:
        raise ValueError("Metadata file not found")
    metadata["errors"]["overall"] = error_msg
    # Direct write for step-by-step execution
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def save_stage_result(metadata_path, stage, data):
    metadata = load_metadata(metadata_path)
    if metadata is None:
        raise ValueError("Metadata file not found")

    tmp_dir = metadata["tmp_dir"]
    results_dir = os.path.join(tmp_dir, f"{stage}_results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f"{stage}.json")

    # Add common fields if not present
    if "stage" not in data:
        data["stage"] = stage
    if "timestamp" not in data:
        data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    if "errors" not in data:
        data["errors"] = []

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    if "stage_results" not in metadata:
        metadata["stage_results"] = {}
    metadata["stage_results"][stage] = {
        "json_path": json_path,
        "output_files": data.get("output_files", {}),
    }

    # Direct write for step-by-step execution
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_previous_result(metadata_path, prev_stage):
    metadata = load_metadata(metadata_path)
    if metadata is None:
        raise ValueError("Metadata file not found")

    if "stage_results" not in metadata or prev_stage not in metadata["stage_results"]:
        raise ValueError(f"Previous stage '{prev_stage}' result not found in metadata")

    json_path = metadata["stage_results"][prev_stage]["json_path"]
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Stage result file not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def is_complete(metadata):
    return (
        metadata.get("current_stage") == "complete"
        and not metadata["errors"]["overall"]
    )
