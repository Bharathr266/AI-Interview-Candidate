import os
from groq import Groq
import tempfile
import dotenv
dotenv.load_dotenv()

import streamlit as st

def transcribe_audio_groq(audio_bytes):
    # Save audio to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    # Init Groq client
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    with open(temp_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            response_format="json",  
            language="en"
        )
    os.remove(temp_path)
    return transcription.text  
