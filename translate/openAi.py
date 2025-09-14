from openai import OpenAI
import os
import logging
from typing import List, TypedDict, cast, Any
from transcription.whisper import DiarizationResult

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class TranslatedDiarizationResult(DiarizationResult, total=False):
    translated_text: str


def translate_with_openai(transcript: List[DiarizationResult], target_language: str = "English") -> List[TranslatedDiarizationResult]:
    translated_segments = []
    if len(transcript) > 20:
        logging.warning("Transcript longer than 20 segments; limiting context to last 10 previous segments to avoid token limits.")
    for i, segment in enumerate(transcript):
        speaker = segment["speaker"]
        text = segment["text"]

        if i > 0:
            context_start = 0 if len(transcript) <= 20 else max(0, i - 10)
            prev_segments = transcript[context_start:i]
            context_str = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in prev_segments])
            user_content = f"Previous dialogue context:\n{context_str}\n\nNow, translate this line spoken by {speaker} into {target_language}, keeping it natural and true to the character's voice in anime dialogue: {text}"
        else:
            user_content = f"Now, translate this line spoken by {speaker} into {target_language}, keeping it natural and true to the character's voice in anime dialogue: {text}"
        if not text.strip():
            translated_text = ""
        else:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in translating anime dialogue to natural, engaging {target_language}. Respond ONLY with the translated text. Do not include any explanations, introductions, or additional commentary."},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                translated_text = content.strip() if content else ""
                # Fallback for model variance: take the first line if the response is multi-line
                if '\n' in translated_text:
                    translated_text = translated_text.split('\n')[0].strip()
            except Exception as e:
                print(f"Translation error for segment '{text}': {e}")
                translated_text = text  # Fallback to original if error

        translated_segment = {**segment, "translated_text": translated_text}
        translated_segments.append(cast(TranslatedDiarizationResult, translated_segment))

    return translated_segments
