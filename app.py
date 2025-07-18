import streamlit as st
from resume_parser import parse_resume
from groq_api import query_groq_chain
from audio_to_text import transcribe_audio_groq
from audio_recorder_streamlit import audio_recorder  # Or any other suitable recorder component

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

st.markdown("### 🎙️ Record your interview question")
audio_bytes = audio_recorder("Click to record your question")
question_text = ""


if audio_bytes and resume_text:
    question_text = transcribe_audio_groq(audio_bytes)
    st.write(f"**Transcribed Question:** {question_text}")

    if question_text:
        with st.spinner("Thinking..."):
            full_context = resume_text
            if user_notes.strip():
                full_context += f"\n\nAdditional Notes from Candidate:\n{user_notes.strip()}"
            response = query_groq_chain(full_context, question_text)
            st.markdown("### ✅ Your Suggested Answer")
            st.write(response)
