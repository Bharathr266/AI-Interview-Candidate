import whisper
import tempfile
import os

model = whisper.load_model("tiny")

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    result = model.transcribe(temp_path)
    os.remove(temp_path)
    return result["text"]
