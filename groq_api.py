from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
import os
import dotenv
import streamlit as st
dotenv.load_dotenv()
llm = ChatGroq(
    groq_api_key= st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

prompt_template = ChatPromptTemplate.from_template("""
You are an AI interview assistant helping candidates in answering their interview questions.
Use the provided resume context to inform your answer when relevant.
If the resume contains useful information related to the question, incorporate it thoughtfully.
If the resume does not contain relevant information, you are allowed to answer using your own general knowledge — but do not make up any facts about the candidate that are not provided.
Ensure your answer is detailed, professional, and aligned with best practices for interview responses.

{context}

Question: {input}
""")

chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    prompt=prompt_template
)

def query_groq_chain(context_text, user_question):
    docs = [Document(page_content=context_text)]
    inputs = {
        "context": context_text,
        "input": user_question
    }
    return chain.run(input_documents=docs, **inputs)
