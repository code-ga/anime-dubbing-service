from openai import OpenAI
import os
import logging
from typing import List, TypedDict, cast, Any, Optional
from transcription.whisper import AssignWordSpeakersResult, DiarizationResult

client = OpenAI(
    base_url="https://ai.hackclub.com",
    api_key=os.getenv("OPENAI_API_KEY", "INVALID_KEY"),
)

PROMPT = """You are an expert in anime. Classify this transcript segment as 'singing' if it appears to be lyrics (repetitive, poetic, song-like, e.g., 'la la la', chorus), or 'speech' for dialogue. Respond only with 'singing' or 'speech' without explanation. Text: {text}"""


class TranslatedDiarizationResult(DiarizationResult, total=False):
    translated_text: str
    is_singing: bool


class TranslateResult(TypedDict, total=False):
    speaker_embeddings: dict[str, list[float]]
    segments: List[dict[str, Any]]
    language: str
    text: str
    translated_segments: List[TranslatedDiarizationResult]
    target_language: str

def translate_with_openai(
    transcript: AssignWordSpeakersResult, tmp_path: str, target_language: str = "English", singing_model: Optional[str] = "openai/gpt-oss-120b"
) -> TranslateResult:
    # Classify singing segments
    for segment in transcript["segments"]:
        text = segment.get("text", "")
        try:
            model_to_use = singing_model or "openai/gpt-oss-120b"
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                max_tokens=10,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            classification = content.strip().lower() if content else ""
            segment["is_singing"] = classification == 'singing'
        except Exception as e:
            logging.warning(f"Classification error for segment '{text}': {e}")
            # Heuristic fallback
            text_stripped = text.strip()
            is_short = len(text_stripped) < 10
            words = text_stripped.lower().split()
            has_repetition = len(words) > 0 and max(words.count(w) for w in set(words)) >= 3
            segment["is_singing"] = is_short or has_repetition

    # Update transcript.json with is_singing flags
    import json
    import os
    transcript_dir = os.path.join(tmp_path, "transcript")
    os.makedirs(transcript_dir, exist_ok=True)
    transcript_path = os.path.join(transcript_dir, "transcript.json")
    to_save_trans = {
        "segments": transcript["segments"],
        "language": transcript["language"],
        "text": transcript["text"],
    }
    with open(transcript_path, "w") as f:
        json.dump(to_save_trans, f, indent=2)

    translated_segments: List[TranslatedDiarizationResult] = []
    source_language = transcript.get("language", "")
    if len(transcript["segments"]) > 20:
        logging.warning(
            "Transcript longer than 20 segments; limiting context to last 10 previous segments to avoid token limits."
        )
    for i, segment in enumerate(transcript["segments"]):
        speaker = segment.get("speaker")
        text = segment.get("text", "")
        is_singing = segment.get("is_singing", False)

        if is_singing:
            translated_text = ""
        elif not text.strip():
            translated_text = ""
        else:
            if i > 0:
                context_start = 0 if len(transcript["segments"]) <= 20 else max(0, i - 10)
                prev_segments = transcript["segments"][context_start:i]
                # Filter out singing segments from context
                prev_context = [seg for seg in prev_segments if not seg.get("is_singing", False)]
                context_str = "\n".join(
                    [f"{seg.get('speaker', 'Unknown')}: {seg.get('text', '')}" for seg in prev_context]
                )
                user_content = f"Previous dialogue context:\n{context_str}\n\nNow, translate this line from {source_language} to {target_language}, spoken by {speaker}, keeping it natural and true to the character's voice in anime dialogue: {text}"
            else:
                user_content = f"Now, translate this line from {source_language} to {target_language}, spoken by {speaker}, keeping it natural and true to the character's voice in anime dialogue: {text}"
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert in translating anime dialogue from {source_language} to natural, engaging {target_language}. Respond ONLY with the translated text. Do not include any explanations, introductions, or additional commentary.",
                        },
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=500,
                    temperature=0.7,
                )
                content = response.choices[0].message.content
                translated_text = content.strip() if content else ""
                # Fallback for model variance: take the first line if the response is multi-line
                if "\n" in translated_text:
                    translated_text = translated_text.split("\n")[0].strip()
            except Exception as e:
                print(f"Translation error for segment '{text}': {e}")
                translated_text = text  # Fallback to original if error
        print(f"Translated {text} to {translated_text}" if not is_singing else f"Skipped singing segment {text}")
        translated_segment = {**segment, "translated_text": translated_text, "is_singing": is_singing}
        translated_segments.append(
            cast(TranslatedDiarizationResult, translated_segment)
        )

    # Save translated.json
    import json
    import os
    translated_dir = os.path.join(tmp_path, "translated")
    os.makedirs(translated_dir, exist_ok=True)
    translated_path = os.path.join(translated_dir, "translated.json")
    to_save = {
        "segments": [{"translated_text": seg.get("translated_text", ""), "speaker": seg.get("speaker"), "start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", ""), "is_singing": seg.get("is_singing", False)} for seg in translated_segments],
        "language": source_language,
        "text": " ".join([seg.get("translated_text", "") for seg in translated_segments if not seg.get("is_singing", False)]),
        "target_language": target_language,
    }
    with open(translated_path, "w") as f:
        json.dump(to_save, f, indent=2)

    return cast(TranslateResult, {
        **transcript,
        "translated_segments": translated_segments,
        "target_language": target_language,
    })
