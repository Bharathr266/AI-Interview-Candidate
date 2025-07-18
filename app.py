import streamlit as st
from resume_parser import parse_resume
from audio_to_text import transcribe_audio
from groq_api import query_groq_chain
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Interview Responder", layout="centered")
st.title("🎤 AI Interview Responder")

# Upload Resume
resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])
resume_text = ""
if resume_file:
    resume_text = parse_resume(resume_file)
    st.success("✅ Resume uploaded and parsed.")

st.markdown("### 📝 Add Optional Project Details or Experience")

user_notes = st.text_area(
    "Tell us more about your project, challenges you faced, experiences you want to highlight, etc.",
    placeholder="E.g. I built a predictive model that reduced churn by 15%, or faced challenges with data imbalance..."
)

# Record Question
st.markdown("### 🎙️ Record your interview question")
# audio_bytes = st.audiorecorder("Click to record your question")  # st.audio_recorder is custom, use JS if needed
audio_bytes = audio_recorder("Click to record your question")
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

question_text = ""
if audio_bytes:
    question_text = transcribe_audio(audio_bytes)
    st.write(f"**Transcribed Question:** {question_text}")

# Ask AI
if st.button("🧠 Get Answer") and resume_text and question_text:
    with st.spinner("Thinking..."):
        full_context = resume_text
        if user_notes.strip():
            full_context += f"\n\nAdditional Notes from Candidate:\n{user_notes.strip()}"
        response = query_groq_chain(full_context, question_text)
        st.markdown("### ✅ Your Suggested Answer")
        st.write(response)
