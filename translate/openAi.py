from openai import OpenAI
import json

from types.convert import TranscriptResult

client = OpenAI(
    base_url="https://ai.hackclub.com"
)


def translate_with_openai(transcript_output_file:str):
    json_raw = json.loads(open(transcript_output_file).read())

    json_input = TranscriptResult.model_validate(json_raw)
    client.chat.completions.create(messages=[
        {
            "role":"system",
            "content": json.dumps(json_input)
        }
    ],model="gpt-3.5-turbo",temperature=0.5)
    for segment in json_input.all_segments:
        for transcript_segment in segment.transcript.segments:
            start_time = transcript_segment.start
            end_time = transcript_segment.end
            duration = end_time - start_time
            text = transcript_segment.text
